"""Phase 12 — Peak demand detection (configurable percentile threshold, per meter).

Definition used in this project:
    An hour is a PEAK for a meter if its consumption is above that meter's P-th percentile
    of hourly consumption (default P = 90, configurable in system settings).

Why per meter? A large shop and a small flat have completely different scales. A single global
threshold would flag every hour of the big consumer and none of the small one.

Severity (how far above the threshold):
    moderate: <= 1.25 × threshold · high: <= 1.5 × threshold · critical: > 1.5 × threshold
"""

import numpy as np
import pandas as pd

SEVERITY_LEVELS = [(1.25, "moderate"), (1.5, "high"), (np.inf, "critical")]


def validate_percentile(percentile: float) -> float:
    if not 50 <= percentile < 100:
        raise ValueError("peak percentile must be in [50, 100)")
    return float(percentile)


def compute_thresholds(df: pd.DataFrame, percentile: float, value_col: str = "kwh") -> pd.Series:
    """Per-meter threshold = P-th percentile of that meter's hourly consumption."""
    validate_percentile(percentile)
    return df.groupby("meter", observed=True)[value_col].quantile(percentile / 100).rename("peak_threshold")


def classify_severity(value: float, threshold: float) -> str | None:
    if threshold <= 0 or value <= threshold:
        return None
    ratio = value / threshold
    return next(label for limit, label in SEVERITY_LEVELS if ratio <= limit)


def detect_peaks(df: pd.DataFrame, thresholds: pd.Series, value_col: str = "kwh") -> pd.DataFrame:
    """Add peak_threshold, is_peak and severity columns."""
    out = df.copy()
    out["peak_threshold"] = out["meter"].map(thresholds).astype(float)
    # A threshold of 0 (mostly idle meter) has no meaningful peak, and severity needs a ratio.
    out["is_peak"] = (out["peak_threshold"] > 0) & (out[value_col] > out["peak_threshold"])
    out["severity"] = [
        classify_severity(v, t) if p else None
        for v, t, p in zip(out[value_col], out["peak_threshold"], out["is_peak"], strict=True)
    ]
    return out


def peak_summary(peaks: pd.DataFrame, value_col: str = "kwh") -> dict:
    """Aggregate statistics for the dashboard: when do peaks happen and how often."""
    events = peaks.loc[peaks["is_peak"]]
    by_hour = events.groupby(events["timestamp"].dt.hour).size().reindex(range(24), fill_value=0)
    by_weekday = events.groupby(events["timestamp"].dt.dayofweek).size().reindex(range(7), fill_value=0)
    return {
        "total_hours": int(len(peaks)),
        "peak_hours": int(len(events)),
        "peak_frequency_pct": float(100 * len(events) / len(peaks)) if len(peaks) else 0.0,
        "max_kwh": float(events[value_col].max()) if len(events) else None,
        "by_hour_of_day": {int(h): int(n) for h, n in by_hour.items()},
        "by_day_of_week": {int(d): int(n) for d, n in by_weekday.items()},
        "most_common_peak_hours": [int(h) for h in by_hour.sort_values(ascending=False).index[:3] if by_hour[h] > 0],
    }
