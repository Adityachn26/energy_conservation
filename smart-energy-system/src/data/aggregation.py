"""Phase 4 — Aggregate 3-minute readings to hourly values on a regular per-meter hourly grid.

Output has ONE row per (meter, hour) between each meter's first and last reading, including
hours that had no readings at all. Those empty hours are kept as rows with ``kwh = NaN``
(missing), NOT 0 — so that later lag features can see the gap instead of jumping over it.
"""

import logging

import pandas as pd

from src.config import READINGS_PER_HOUR

logger = logging.getLogger(__name__)


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """3-min -> hourly. kWh: SUM, voltage/current/frequency: MEAN, readings_per_hour: COUNT.

    Returns columns: meter, timestamp (hour start), kwh, voltage, current, frequency,
    readings_per_hour, is_complete.
    """
    work = df.assign(hour_start=df["timestamp"].dt.floor("h"))
    hourly = work.groupby(["meter", "hour_start"], observed=True).agg(
        kwh=("kwh", "sum"),
        voltage=("voltage", "mean"),
        current=("current", "mean"),
        frequency=("frequency", "mean"),
        readings_per_hour=("kwh", "size"),
    )

    # Build the full hourly grid for each meter (first hour -> last hour), so that hours
    # with no readings appear explicitly instead of silently disappearing.
    spans = hourly.reset_index().groupby("meter", observed=True)["hour_start"].agg(["min", "max"])
    grid_parts = [
        pd.MultiIndex.from_product(
            [[meter], pd.date_range(row["min"], row["max"], freq="h")],
            names=["meter", "hour_start"],
        )
        for meter, row in spans.iterrows()
    ]
    full_index = grid_parts[0].append(grid_parts[1:]) if len(grid_parts) > 1 else grid_parts[0]
    hourly = hourly.reindex(full_index)

    # Hours absent from the raw data -> 0 readings; their kWh/V/I/Hz stay NaN (missing).
    hourly["readings_per_hour"] = hourly["readings_per_hour"].fillna(0).astype("int64")
    hourly["is_complete"] = hourly["readings_per_hour"] == READINGS_PER_HOUR

    # Partial hours (1..19 readings) would under-count the hourly kWh sum, so their values are
    # also treated as missing for modelling purposes. They remain visible via readings_per_hour.
    incomplete = ~hourly["is_complete"]
    hourly.loc[incomplete, ["kwh", "voltage", "current", "frequency"]] = float("nan")

    hourly = hourly.reset_index().rename(columns={"hour_start": "timestamp"})
    hourly["meter"] = hourly["meter"].astype("string")

    counts = hourly_counts(hourly)
    logger.info(
        "Hourly grid: %s meter-hours | complete=%s | empty=%s | partial=%s",
        f"{counts['hourly_rows']:,}",
        f"{counts['complete_hours']:,}",
        f"{counts['empty_hours']:,}",
        f"{counts['partial_hours']:,}",
    )
    return hourly


def hourly_counts(hourly: pd.DataFrame) -> dict:
    """Count complete / empty / partial meter-hours (used for logging and verification)."""
    rph = hourly["readings_per_hour"]
    return {
        "hourly_rows": int(len(hourly)),
        "complete_hours": int((rph == READINGS_PER_HOUR).sum()),
        "empty_hours": int((rph == 0).sum()),
        "partial_hours": int(((rph > 0) & (rph < READINGS_PER_HOUR)).sum()),
    }


def select_complete_hours(hourly: pd.DataFrame) -> pd.DataFrame:
    """The clean modelling dataset: only hours with all 20 readings."""
    clean = hourly.loc[hourly["is_complete"]].reset_index(drop=True)
    logger.info("Clean hourly dataset: %s complete meter-hours", f"{len(clean):,}")
    return clean
