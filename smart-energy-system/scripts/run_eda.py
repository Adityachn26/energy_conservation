"""Exploratory Data Analysis (Phase 5) — saves figures to reports/figures/ and a summary JSON.

Usage (after the data pipeline):
    python -m scripts.run_eda

Every figure follows the same rules as the dashboard: one y-axis per chart, thin lines,
fixed colour roles, single-hue scale for magnitude, diverging blue↔red (grey midpoint) for
correlation.
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import (  # noqa: E402
    DATA_QUALITY_REPORT,
    FIGURES_DIR,
    HOURLY_ALL_PARQUET,
    HOURLY_CLEAN_PARQUET,
    REPORTS_DIR,
    ensure_directories,
    setup_logging,
)
from src.data.loader import load_parquet  # noqa: E402

logger = logging.getLogger("eda")

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
INK_2, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a8a", "#b02a2a"])
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "lines.linewidth": 2,
        "figure.dpi": 130,
    }
)


def save(fig, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure -> %s", path)


def main() -> None:
    setup_logging()
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    ensure_directories()
    df = load_parquet(HOURLY_CLEAN_PARQUET)
    grid = load_parquet(HOURLY_ALL_PARQUET)
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dow"] >= 5
    source = json.loads(DATA_QUALITY_REPORT.read_text()) if DATA_QUALITY_REPORT.exists() else {}
    tag = " (SYNTHETIC)" if source.get("is_synthetic") else ""

    # 1. Average daily profile: weekday vs weekend
    profile = df.groupby(["is_weekend", "hour"])["kwh"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(profile.index, profile[False], color=BLUE, label="Weekday")
    if True in profile:
        ax.plot(profile.index, profile[True], color=ORANGE, label="Weekend")
    ax.set(title=f"Average hourly consumption by hour of day{tag}", xlabel="Hour of day", ylabel="kWh per hour")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save(fig, "01_hourly_profile_weekday_weekend")

    # 2. Daily total (all meters) — days with missing hours flagged
    daily = df.groupby(df["timestamp"].dt.date).agg(kwh=("kwh", "sum"), meter_hours=("kwh", "size"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(daily.index), daily["kwh"], color=BLUE, linewidth=1.5)
    ax.set(title=f"Daily total consumption, all meters{tag}", ylabel="kWh per day")
    ax.set_ylim(bottom=0)
    save(fig, "02_daily_total")

    # 3. Monthly consumption, normalised per meter-hour (fair despite missing hours)
    monthly = df.groupby(df["timestamp"].dt.to_period("M"))["kwh"].agg(["sum", "mean"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(monthly.index.astype(str), monthly["mean"], color=BLUE, width=0.7)
    ax.set(title=f"Average consumption per meter-hour by month{tag}", ylabel="kWh per meter-hour")
    ax.grid(axis="x", visible=False)
    save(fig, "03_monthly_average")

    # 4. Day of week
    by_dow = df.groupby("dow")["kwh"].mean().reindex(range(7))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(WEEKDAYS, by_dow.to_numpy(), color=BLUE, width=0.7)
    ax.set(title=f"Average consumption by day of week{tag}", ylabel="kWh per meter-hour")
    ax.grid(axis="x", visible=False)
    save(fig, "04_day_of_week")

    # 5. Weekday × hour heatmap
    pivot = df.pivot_table(index="dow", columns="hour", values="kwh", aggfunc="mean").reindex(range(7))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=SEQUENTIAL)
    ax.set_yticks(range(7), WEEKDAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.grid(False)
    ax.set(title=f"Average kWh by weekday and hour{tag}", xlabel="Hour of day")
    fig.colorbar(im, ax=ax, label="kWh per hour")
    save(fig, "05_weekday_hour_heatmap")

    # 6. Meter-level average consumption
    per_meter = df.groupby("meter")["kwh"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.22 * len(per_meter))))
    ax.barh(per_meter.index.astype(str), per_meter.to_numpy(), color=BLUE, height=0.7)
    ax.set(title=f"Average hourly consumption per meter{tag}", xlabel="kWh per hour")
    ax.grid(axis="y", visible=False)
    save(fig, "06_per_meter_average")

    # 7. Distribution of hourly consumption
    fig, ax = plt.subplots(figsize=(8, 4))
    upper = df["kwh"].quantile(0.995)
    ax.hist(df["kwh"].clip(upper=upper), bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    for q, label in ((0.5, "median"), (0.9, "P90")):
        value = df["kwh"].quantile(q)
        ax.axvline(value, color=RED if q == 0.9 else INK_2, linewidth=1)
        ax.text(value, ax.get_ylim()[1] * 0.95, f" {label} {value:.2f}", color=INK_2, fontsize=9)
    ax.set(
        title=f"Distribution of hourly consumption{tag}",
        xlabel="kWh per hour (clipped at P99.5)",
        ylabel="Meter-hours",
    )
    save(fig, "07_distribution")

    # 8. Correlation matrix (kWh, voltage, current, frequency)
    corr = df[["kwh", "voltage", "current", "frequency"]].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    labels = ["kWh", "Voltage", "Current", "Frequency"]
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) > 0.6 else "#0b0b0b"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    ax.set_title(f"Correlation (Pearson){tag}")
    fig.colorbar(im, ax=ax)
    save(fig, "08_correlation")

    # 9. Current vs energy (sample) — physical relationship P ≈ V·I
    sample = df.sample(min(len(df), 20_000), random_state=0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(sample["current"], sample["kwh"], s=4, alpha=0.25, color=BLUE)
    ax.set(title=f"Hourly energy vs average current{tag}", xlabel="Average current (A)", ylabel="kWh per hour")
    save(fig, "09_current_vs_kwh")

    # 10. Data coverage: share of empty hours per meter
    coverage = grid.groupby("meter").agg(
        hours=("readings_per_hour", "size"), empty=("readings_per_hour", lambda s: int((s == 0).sum()))
    )
    coverage["empty_pct"] = 100 * coverage["empty"] / coverage["hours"]
    coverage = coverage.sort_values("empty_pct")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.22 * len(coverage))))
    ax.barh(coverage.index.astype(str), coverage["empty_pct"], color=MUTED, height=0.7)
    ax.set(title=f"Hours with no readings, per meter{tag}", xlabel="% of the meter's hours")
    ax.grid(axis="y", visible=False)
    save(fig, "10_missing_hours_per_meter")

    # Outliers (IQR rule, per meter) + summary
    def iqr_outliers(s: pd.Series) -> int:
        q1, q3 = s.quantile([0.25, 0.75])
        return int((s > q3 + 1.5 * (q3 - q1)).sum())

    outliers = df.groupby("meter")["kwh"].apply(iqr_outliers)
    summary = {
        "source_file": source.get("source_file"),
        "is_synthetic": source.get("is_synthetic"),
        "complete_meter_hours": int(len(df)),
        "kwh_describe": {k: float(v) for k, v in df["kwh"].describe().items()},
        "peak_hour_of_day": int(profile.mean(axis=1).idxmax()),
        "weekday_mean_kwh": float(df.loc[~df["is_weekend"], "kwh"].mean()),
        "weekend_mean_kwh": float(df.loc[df["is_weekend"], "kwh"].mean()),
        "monthly_mean_kwh_per_meter_hour": {str(k): float(v) for k, v in monthly["mean"].items()},
        "correlation_with_kwh": {k: float(v) for k, v in corr["kwh"].items()},
        "iqr_high_outliers_total": int(outliers.sum()),
        "iqr_high_outliers_pct": float(100 * outliers.sum() / len(df)),
        "empty_hours_pct_by_meter": {str(k): round(float(v), 2) for k, v in coverage["empty_pct"].items()},
        "notes": [
            "IQR 'outliers' are kept: high consumption hours are real and are exactly the peaks we want to detect.",
            "Monthly values are means per meter-hour so months with more missing hours are not under-counted.",
        ],
    }
    out = REPORTS_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("EDA summary -> %s", out)


if __name__ == "__main__":
    main()
