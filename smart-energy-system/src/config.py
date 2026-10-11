"""Central configuration: file paths, column names, constants and environment settings.

Every other module imports paths/settings from here, so there is exactly one place to change them.
Secrets (database password etc.) are read from environment variables / the local `.env` file —
never hardcoded.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load `.env` if it exists. Inside Docker the variables come from docker-compose instead,
# and this call is a harmless no-op.
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"

DEFAULT_RAW_CSV = RAW_DATA_DIR / "CEEW - Smart meter data Bareilly 2020.csv"
SYNTHETIC_RAW_CSV = RAW_DATA_DIR / "synthetic_sample.csv"

RAW_3MIN_PARQUET = INTERIM_DATA_DIR / "raw_3min.parquet"
HOURLY_ALL_PARQUET = INTERIM_DATA_DIR / "hourly_all.parquet"
HOURLY_CLEAN_PARQUET = PROCESSED_DATA_DIR / "hourly_clean.parquet"
FEATURES_PARQUET = PROCESSED_DATA_DIR / "features.parquet"
DATA_QUALITY_REPORT = REPORTS_DIR / "data_quality.json"

BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.parquet"
EXPERIMENT_LOG_PATH = EXPERIMENTS_DIR / "experiments.csv"
MODEL_COMPARISON_PATH = EXPERIMENTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Raw dataset schema (CEEW smart-meter CSV)
# ---------------------------------------------------------------------------
# Original column name -> short snake_case name used everywhere in the code.
RAW_COLUMN_MAP = {
    "x_Timestamp": "timestamp",
    "t_kWh": "kwh",
    "z_Avg Voltage (Volt)": "voltage",
    "z_Avg Current (Amp)": "current",
    "y_Freq (Hz)": "frequency",
    "meter": "meter",
}
READING_INTERVAL_MINUTES = 3
READINGS_PER_HOUR = 60 // READING_INTERVAL_MINUTES  # = 20

# Numbers measured on the real 2020 Bareilly file (Colab analysis). The pipeline logs a
# warning if a run on the real file does not reproduce them.
EXPECTED_REAL_DATA = {
    "raw_rows": 6_627_360,
    "meters": 46,
    "hourly_rows": 378_624,
    "empty_hours": 47_256,
    "complete_hours": 331_368,
}

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TARGET_COLUMN = "kwh"
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15  # test = the remaining 0.15
MAX_FORECAST_HORIZON_HOURS = 24

# ---------------------------------------------------------------------------
# Application defaults (overridable at runtime via the system_settings table)
# ---------------------------------------------------------------------------
DEFAULT_TARIFF_PER_KWH = float(os.getenv("DEFAULT_TARIFF_PER_KWH", "7.0"))
DEFAULT_PEAK_PERCENTILE = float(os.getenv("DEFAULT_PEAK_PERCENTILE", "90"))
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Kolkata")
CURRENCY = os.getenv("CURRENCY", "INR")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_database_url() -> str:
    """Build the SQLAlchemy database URL from environment variables.

    `DATABASE_URL` wins if set (handy for tests, e.g. ``sqlite://``). Otherwise the URL is
    assembled from the individual POSTGRES_* variables so the password never appears in code.
    """
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        return explicit_url

    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")
    if not (user and password and database):
        raise RuntimeError(
            "Database credentials missing. Copy .env.example to .env and set "
            "POSTGRES_USER, POSTGRES_PASSWORD and POSTGRES_DB (or set DATABASE_URL)."
        )
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once for scripts, the API and the dashboard."""
    logging.basicConfig(
        level=(level or LOG_LEVEL).upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    """Create output directories if they don't exist yet."""
    for directory in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        EXPERIMENTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
