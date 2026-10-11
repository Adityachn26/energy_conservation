"""Data pipeline: raw CSV -> quality report -> hourly aggregation -> features (Phases 2–6).

Usage:
    python -m scripts.run_pipeline                                  # real CEEW file in data/raw/
    python -m scripts.run_pipeline --input data/raw/synthetic_sample.csv
"""

import argparse
import json
import logging
from pathlib import Path

from src.config import (
    DATA_QUALITY_REPORT,
    DEFAULT_RAW_CSV,
    EXPECTED_REAL_DATA,
    FEATURES_PARQUET,
    HOURLY_ALL_PARQUET,
    HOURLY_CLEAN_PARQUET,
    RAW_3MIN_PARQUET,
    ensure_directories,
    setup_logging,
)
from src.data.aggregation import aggregate_to_hourly, hourly_counts, select_complete_hours
from src.data.loader import load_raw_csv, save_parquet
from src.data.preprocessing import clean_raw_data, data_quality_report
from src.features.feature_engineering import compute_features, select_model_rows

logger = logging.getLogger("pipeline")


def verify_against_colab(report: dict) -> None:
    """On the real file, the counts must match what was measured in Colab."""
    measured = {
        "raw_rows": report["raw"]["rows"],
        "meters": report["raw"]["meters"],
        **{k: report["hourly"][k] for k in ("hourly_rows", "empty_hours", "complete_hours")},
    }
    for key, expected in EXPECTED_REAL_DATA.items():
        status = "OK" if measured[key] == expected else "MISMATCH"
        log = logger.info if status == "OK" else logger.warning
        log("Check %-15s expected=%s measured=%s %s", key, f"{expected:,}", f"{measured[key]:,}", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV, help="raw 3-minute CSV")
    args = parser.parse_args()

    setup_logging()
    ensure_directories()
    is_synthetic = "synthetic" in args.input.name.lower()
    if is_synthetic:
        logger.warning("Running on SYNTHETIC data — results are for testing only.")

    raw = load_raw_csv(args.input)
    report = {"source_file": args.input.name, "is_synthetic": is_synthetic, "raw": data_quality_report(raw)}

    clean_raw, removed = clean_raw_data(raw)
    report["cleaning_removed"] = removed
    save_parquet(clean_raw, RAW_3MIN_PARQUET)

    hourly = aggregate_to_hourly(clean_raw)
    report["hourly"] = hourly_counts(hourly)
    save_parquet(hourly, HOURLY_ALL_PARQUET)
    save_parquet(select_complete_hours(hourly), HOURLY_CLEAN_PARQUET)

    featured = compute_features(hourly)
    model_rows, drop_log = select_model_rows(featured)
    report["features"] = drop_log
    save_parquet(model_rows, FEATURES_PARQUET)

    if not is_synthetic:
        verify_against_colab(report)

    DATA_QUALITY_REPORT.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Data quality report -> %s", DATA_QUALITY_REPORT)


if __name__ == "__main__":
    main()
