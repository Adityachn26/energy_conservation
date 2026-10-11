"""Generate a SYNTHETIC CSV with the CEEW schema (development/testing only — not real data).

Usage:
    python -m scripts.generate_sample_data                # 6 meters x 120 days
    python -m scripts.generate_sample_data --meters 10 --days 365
"""

import argparse
import logging

from src.config import SYNTHETIC_RAW_CSV, ensure_directories, setup_logging
from src.data.synthetic import generate_synthetic_readings

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meters", type=int, default=6)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    ensure_directories()
    df = generate_synthetic_readings(n_meters=args.meters, days=args.days, seed=args.seed)
    df.to_csv(SYNTHETIC_RAW_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
    logger.warning("SYNTHETIC data written (NOT real CEEW data): %s rows -> %s", f"{len(df):,}", SYNTHETIC_RAW_CSV)


if __name__ == "__main__":
    main()
