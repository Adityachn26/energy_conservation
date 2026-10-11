"""Phase 2 — Data ingestion: read the raw CEEW CSV, validate its schema, save as Parquet."""

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_COLUMN_MAP

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the input file does not look like the CEEW smart-meter dataset."""


def load_raw_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load the raw 3-minute smart-meter CSV and return a typed DataFrame.

    Columns are renamed to short snake_case names (see ``RAW_COLUMN_MAP``):
    timestamp, kwh, voltage, current, frequency, meter.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {csv_path}. Put the CEEW CSV in data/raw/ "
            "or generate synthetic data with `python -m scripts.generate_sample_data`."
        )

    logger.info("Reading raw CSV: %s", csv_path)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [col for col in RAW_COLUMN_MAP if col not in header]
    if missing:
        raise DataValidationError(f"Missing expected columns {missing}. Found: {header}")

    df = pd.read_csv(
        csv_path,
        usecols=list(RAW_COLUMN_MAP),
        dtype={
            "t_kWh": "float64",
            "z_Avg Voltage (Volt)": "float64",
            "z_Avg Current (Amp)": "float64",
            "y_Freq (Hz)": "float64",
            "meter": "string",
        },
    )
    df = df.rename(columns=RAW_COLUMN_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["meter"] = df["meter"].str.strip()

    logger.info(
        "Loaded %s rows, %s meters, %s -> %s",
        f"{len(df):,}",
        df["meter"].nunique(),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    return df[["meter", "timestamp", "kwh", "voltage", "current", "frequency"]]


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as Parquet (creating parent folders if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s rows -> %s", f"{len(df):,}", path)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file produced by an earlier pipeline stage."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the data pipeline first: python -m scripts.run_pipeline")
    df = pd.read_parquet(path)
    logger.info("Loaded %s rows <- %s", f"{len(df):,}", path)
    return df
