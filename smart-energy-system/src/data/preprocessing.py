"""Phase 3 — Data quality checks and minimal, *logged* cleaning of the raw 3-minute data.

Philosophy: never silently discard data. Every removed row is counted and written to the
data-quality report so it can be explained in the project report.
"""

import logging

import pandas as pd

from src.config import READING_INTERVAL_MINUTES

logger = logging.getLogger(__name__)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Describe the raw data without modifying it."""
    on_grid = (df["timestamp"].dt.minute % READING_INTERVAL_MINUTES == 0) & (df["timestamp"].dt.second == 0)
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "meters": int(df["meter"].nunique()),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "missing_values": {col: int(n) for col, n in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_meter_timestamps": int(df.duplicated(subset=["meter", "timestamp"]).sum()),
        "negative_kwh": int((df["kwh"] < 0).sum()),
        "timestamps_off_3min_grid": int((~on_grid).sum()),
        "kwh_stats": {k: float(v) for k, v in df["kwh"].describe().items()},
        "voltage_stats": {k: float(v) for k, v in df["voltage"].describe().items()},
        "frequency_stats": {k: float(v) for k, v in df["frequency"].describe().items()},
    }
    logger.info(
        "Quality: %s rows | missing=%s | duplicate rows=%s | duplicate (meter,timestamp)=%s | negative kWh=%s",
        f"{report['rows']:,}",
        sum(report["missing_values"].values()),
        report["duplicate_rows"],
        report["duplicate_meter_timestamps"],
        report["negative_kwh"],
    )
    return report


def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply safety-net cleaning rules and return (clean_df, log of what was removed).

    On the real 2020 Bareilly file none of these rules remove anything (0 missing, 0 duplicates),
    but the pipeline must stay correct if a different year/city file is used.

    Rules:
      1. Drop exact duplicate rows.
      2. For repeated (meter, timestamp) pairs, keep the first reading.
      3. Drop rows where kWh is missing (cannot be used as a target).
      4. Negative kWh is physically impossible for consumption -> dropped.
    """
    removed = {}
    n0 = len(df)

    df = df.drop_duplicates()
    removed["exact_duplicates"] = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates(subset=["meter", "timestamp"], keep="first")
    removed["duplicate_meter_timestamps"] = n1 - len(df)

    n2 = len(df)
    df = df.dropna(subset=["kwh"])
    removed["missing_kwh"] = n2 - len(df)

    n3 = len(df)
    df = df.loc[df["kwh"] >= 0]
    removed["negative_kwh"] = n3 - len(df)

    removed["total_removed"] = n0 - len(df)
    if removed["total_removed"]:
        logger.warning("Cleaning removed %s rows: %s", removed["total_removed"], removed)
    else:
        logger.info("Cleaning removed 0 rows (data already clean)")

    df = df.sort_values(["meter", "timestamp"]).reset_index(drop=True)
    return df, removed
