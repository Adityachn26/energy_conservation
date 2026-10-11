"""Shared pytest fixtures: small synthetic datasets and an API client backed by in-memory SQLite."""

import os

os.environ.setdefault("DATABASE_URL", "sqlite://")  # tests never touch the real PostgreSQL

import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_readings


def make_raw(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Tiny 3-minute frame in the *renamed* schema: (meter, timestamp, kwh)."""
    df = pd.DataFrame(rows, columns=["meter", "timestamp", "kwh"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["meter"] = df["meter"].astype("string")
    df["voltage"], df["current"], df["frequency"] = 230.0, 1.0, 50.0
    return df


def full_hour(meter: str, hour_start: str, kwh_per_reading: float = 0.01) -> list[tuple]:
    start = pd.Timestamp(hour_start)
    return [(meter, start + pd.Timedelta(minutes=3 * i), kwh_per_reading) for i in range(20)]


def hourly_frame(values: dict[str, list[float | None]], start: str = "2020-01-06 00:00") -> pd.DataFrame:
    """Regular hourly grid per meter; None = missing hour (NaN, incomplete)."""
    frames = []
    for meter, series in values.items():
        ts = pd.date_range(start, periods=len(series), freq="h")
        kwh = pd.Series(series, dtype="float64")
        frames.append(
            pd.DataFrame(
                {
                    "meter": pd.Series([meter] * len(series), dtype="string"),
                    "timestamp": ts,
                    "kwh": kwh,
                    "readings_per_hour": [0 if pd.isna(v) else 20 for v in series],
                    "is_complete": kwh.notna(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_raw_original() -> pd.DataFrame:
    """Synthetic data with the ORIGINAL CEEW column names (2 meters, 60 days)."""
    return generate_synthetic_readings(n_meters=2, days=60, missing_block_count=2, seed=7)
