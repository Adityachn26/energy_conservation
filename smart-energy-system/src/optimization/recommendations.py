"""Phase 14 — Rule-based energy-saving recommendations.

The engine looks at (1) predicted consumption, (2) the meter's peak threshold and
(3) its historical patterns, and produces human-readable ADVICE. It does not control any
appliance. Every rule and threshold is listed below so the logic is transparent and easy to
explain (and easy to tune).

"potential_saving_kwh" is an estimate of energy above a reference level (peak threshold,
typical usage, ...). Shifting load to another time reduces the PEAK but not the total kWh;
actually reducing usage lowers both. Messages say which one applies.
"""

from dataclasses import asdict, dataclass

import pandas as pd

# ---- Rule parameters (tunable) ----
ABOVE_TYPICAL_RATIO = 1.30  # prediction > 130% of usual for that hour of day
RECENT_TREND_RATIO = 1.20  # last 7 days > 120% of the previous 28 days
OVERNIGHT_HOURS = range(0, 5)  # 00:00–05:00
OVERNIGHT_BASELOAD_RATIO = 0.80  # overnight mean > 80% of the overall mean (little day/night difference)
DAYS_PER_MONTH = 30


@dataclass
class Recommendation:
    recommendation_type: str
    severity: str  # info | warning | critical
    message: str
    timestamp: pd.Timestamp | None = None
    potential_saving_kwh: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _format_windows(hours: list[int]) -> list[str]:
    """[18, 19, 20, 23] -> ['18:00–21:00', '23:00–00:00']"""
    windows, hours = [], sorted(set(hours))
    for hour in hours:
        if windows and hour == windows[-1][1] + 1:
            windows[-1][1] = hour
        else:
            windows.append([hour, hour])
    return [f"{start:02d}:00–{(end + 1) % 24:02d}:00" for start, end in windows]


def _predicted_peak_rule(predictions, threshold) -> list[Recommendation]:
    above = [(ts, kwh) for ts, kwh in predictions if kwh > threshold]
    if not above:
        return []
    excess = sum(kwh - threshold for _, kwh in above)
    windows = ", ".join(_format_windows([ts.hour for ts, _ in above]))
    peak_value = max(kwh for _, kwh in above)
    return [
        Recommendation(
            recommendation_type="predicted_peak",
            severity="critical" if peak_value > 1.5 * threshold else "warning",
            message=(
                f"High consumption is predicted during {windows} (up to {peak_value:.2f} kWh in an hour; "
                f"peak threshold {threshold:.2f} kWh). Consider shifting non-essential loads such as "
                "washing machines, water pumps or ironing outside this window."
            ),
            timestamp=above[0][0],
            potential_saving_kwh=round(excess, 3),
        )
    ]


def _above_typical_rule(history, predictions) -> list[Recommendation]:
    typical_by_hour = history.groupby(history["timestamp"].dt.hour)["kwh"].mean()
    recommendations = []
    for ts, kwh in predictions[:1]:  # only the evaluated next-hour prediction
        typical = typical_by_hour.get(ts.hour)
        if typical and typical > 0 and kwh > ABOVE_TYPICAL_RATIO * typical:
            pct = 100 * (kwh / typical - 1)
            recommendations.append(
                Recommendation(
                    recommendation_type="above_typical",
                    severity="warning",
                    message=(
                        f"Predicted consumption for {ts:%H:00} ({kwh:.2f} kWh) is {pct:.0f}% higher than "
                        f"this meter's typical {typical:.2f} kWh at that hour. Check for appliances left running."
                    ),
                    timestamp=ts,
                    potential_saving_kwh=round(kwh - typical, 3),
                )
            )
    return recommendations


def _recurring_peak_window_rule(history, threshold) -> list[Recommendation]:
    peaks = history.loc[history["kwh"] > threshold]
    if len(peaks) < 5:
        return []
    by_hour = peaks.groupby(peaks["timestamp"].dt.hour).size().sort_values(ascending=False)
    # Smallest set of hours that together cover MORE than half of all peak events.
    top_hours, covered = [], 0
    for hour, count in by_hour.items():
        top_hours.append(int(hour))
        covered += count
        if covered > 0.5 * len(peaks):
            break
    share = 100 * covered / len(peaks)
    return [
        Recommendation(
            recommendation_type="recurring_peak_window",
            severity="info",
            message=(
                f"Historically, {share:.0f}% of this meter's peak hours fall in "
                f"{', '.join(_format_windows(top_hours))}. Reduce use of high-load appliances "
                "(AC, geysers, heaters) during these periods where possible."
            ),
        )
    ]


def _overnight_baseload_rule(history) -> list[Recommendation]:
    overall_mean = history["kwh"].mean()
    night = history.loc[history["timestamp"].dt.hour.isin(OVERNIGHT_HOURS), "kwh"]
    if night.empty or not overall_mean or overall_mean <= 0:
        return []
    night_mean = night.mean()
    if night_mean <= OVERNIGHT_BASELOAD_RATIO * overall_mean:
        return []
    reducible_per_hour = night_mean - OVERNIGHT_BASELOAD_RATIO * overall_mean
    monthly_kwh = reducible_per_hour * len(OVERNIGHT_HOURS) * DAYS_PER_MONTH
    return [
        Recommendation(
            recommendation_type="high_overnight_baseload",
            severity="info",
            message=(
                f"Overnight consumption (00:00–05:00) averages {night_mean:.2f} kWh/h — "
                f"{100 * night_mean / overall_mean:.0f}% of the all-day average. Check standby and "
                f"always-on devices; reducing this could save roughly {monthly_kwh:.0f} kWh per month (estimate)."
            ),
            potential_saving_kwh=round(monthly_kwh, 2),
        )
    ]


def _recent_trend_rule(history) -> list[Recommendation]:
    end = history["timestamp"].max()
    recent = history.loc[history["timestamp"] > end - pd.Timedelta(days=7), "kwh"]
    previous = history.loc[
        (history["timestamp"] <= end - pd.Timedelta(days=7)) & (history["timestamp"] > end - pd.Timedelta(days=35)),
        "kwh",
    ]
    if len(recent) < 24 or len(previous) < 24 * 7 or previous.mean() <= 0:
        return []
    ratio = recent.mean() / previous.mean()
    if ratio <= RECENT_TREND_RATIO:
        return []
    return [
        Recommendation(
            recommendation_type="rising_consumption",
            severity="warning",
            message=(
                f"Average consumption over the last 7 days is {100 * (ratio - 1):.0f}% higher than "
                "the previous four weeks. Review recent changes in appliance usage."
            ),
            timestamp=end,
            potential_saving_kwh=round((recent.mean() - previous.mean()) * len(recent), 2),
        )
    ]


def generate_recommendations(
    history: pd.DataFrame,
    peak_threshold: float,
    predictions: list[tuple[pd.Timestamp, float]] | None = None,
) -> list[Recommendation]:
    """Apply all rules. ``history`` has columns timestamp, kwh (one meter, complete hours)."""
    predictions = predictions or []
    recommendations: list[Recommendation] = []
    recommendations += _predicted_peak_rule(predictions, peak_threshold)
    if not history.empty:
        recommendations += _above_typical_rule(history, predictions)
        recommendations += _recurring_peak_window_rule(history, peak_threshold)
        recommendations += _overnight_baseload_rule(history)
        recommendations += _recent_trend_rule(history)

    if not recommendations:
        recommendations.append(
            Recommendation(
                recommendation_type="normal",
                severity="info",
                message="Consumption is within the normal range for this meter. No action needed.",
            )
        )
    return recommendations
