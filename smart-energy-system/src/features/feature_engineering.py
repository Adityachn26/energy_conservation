"""Phase 6 — Feature engineering (time, lag and rolling features), leakage-safe.

Two rules make these features safe:

1. **Computed on the regular hourly grid.** Input must contain every hour for each meter,
   with missing hours as NaN. Then ``shift(1)`` really means "one hour ago". If missing
   hours were deleted first, ``shift(1)`` could silently return a value from days earlier.

2. **Only the past is used.** Lags are past values by definition. Rolling statistics are
   computed on the series shifted by one hour, so the window for time T is T-w ... T-1 and
   never includes the value we are trying to predict.

All calculations are grouped by meter, so one meter's history never leaks into another's.

Same-hour voltage/current are NOT used as features: kWh ≈ V × I × power-factor × time, so the
same-hour current practically *is* the answer (leakage), and it is unknown when forecasting.
"""

import logging
import math

import pandas as pd

from src.config import TARGET_COLUMN

logger = logging.getLogger(__name__)

TIME_FEATURES = ["hour", "day_of_week", "month", "day_of_month", "is_weekend"]
LAG_HOURS = [1, 2, 3, 24, 168]
LAG_FEATURES = [f"lag_{h}h" for h in LAG_HOURS]
ROLLING_MEAN_WINDOWS = [3, 6, 24, 168]
ROLLING_FEATURES = [f"rolling_mean_{w}h" for w in ROLLING_MEAN_WINDOWS] + [
    "rolling_std_24h",
    "rolling_max_24h",
]
FEATURE_COLUMNS = TIME_FEATURES + LAG_FEATURES + ROLLING_FEATURES

# How much history a meter needs before every feature can be computed.
MAX_LOOKBACK_HOURS = max(max(LAG_HOURS), max(ROLLING_MEAN_WINDOWS))


def _min_periods(window: int) -> int:
    """A rolling statistic needs more than half of its window present (rest may be gaps)."""
    return math.floor(window / 2) + 1


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """hour (0-23), day_of_week (0=Mon..6=Sun), month, day_of_month, is_weekend (Sat/Sun=1)."""
    ts = df["timestamp"]
    return df.assign(
        hour=ts.dt.hour.astype("int64"),
        day_of_week=ts.dt.dayofweek.astype("int64"),
        month=ts.dt.month.astype("int64"),
        day_of_month=ts.dt.day.astype("int64"),
        is_weekend=(ts.dt.dayofweek >= 5).astype("int64"),
    )


def check_regular_hourly_grid(df: pd.DataFrame) -> None:
    """Raise if any meter's timestamps are not consecutive hours (lags would be wrong)."""
    gaps = df.groupby("meter", observed=True)["timestamp"].diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Input is not a regular hourly grid per meter. Use the output of "
            "aggregate_to_hourly() (missing hours as NaN rows), not the filtered clean data."
        )


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-meter lag and (shifted) rolling features on a regular hourly grid."""
    check_regular_hourly_grid(df)
    df = df.sort_values(["meter", "timestamp"]).reset_index(drop=True)
    by_meter = df.groupby("meter", observed=True)[TARGET_COLUMN]

    new_columns = {f"lag_{h}h": by_meter.shift(h) for h in LAG_HOURS}

    def rolling(stat: str, window: int) -> pd.Series:
        return by_meter.transform(
            lambda s: getattr(s.shift(1).rolling(window, min_periods=_min_periods(window)), stat)()
        )

    for window in ROLLING_MEAN_WINDOWS:
        new_columns[f"rolling_mean_{window}h"] = rolling("mean", window)
    new_columns["rolling_std_24h"] = rolling("std", 24)
    new_columns["rolling_max_24h"] = rolling("max", 24)

    return df.assign(**new_columns)


def compute_features(hourly_grid: pd.DataFrame) -> pd.DataFrame:
    """All features for every row of the hourly grid (no rows removed)."""
    return add_lag_and_rolling_features(add_time_features(hourly_grid))


def select_model_rows(featured: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep rows usable for training/evaluation, and report exactly why others were dropped.

    A row is usable if (a) its target hour is complete (20/20 readings) and
    (b) every feature is available (enough history, no gap inside the required lags).
    """
    complete = featured["is_complete"]
    has_features = featured[FEATURE_COLUMNS].notna().all(axis=1)
    usable = featured.loc[complete & has_features].reset_index(drop=True)

    drop_log = {
        "grid_rows": int(len(featured)),
        "dropped_incomplete_target_hour": int((~complete).sum()),
        "dropped_missing_history": int((complete & ~has_features).sum()),
        "model_rows": int(len(usable)),
    }
    logger.info(
        "Feature rows: %s usable of %s grid rows (incomplete target hours: %s, insufficient history / gap in lags: %s)",
        f"{drop_log['model_rows']:,}",
        f"{drop_log['grid_rows']:,}",
        f"{drop_log['dropped_incomplete_target_hour']:,}",
        f"{drop_log['dropped_missing_history']:,}",
    )
    return usable, drop_log
