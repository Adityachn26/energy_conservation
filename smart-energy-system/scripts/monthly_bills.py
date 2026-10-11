"""Estimated monthly electricity bill per meter with UPPCL urban domestic slab rates.

Usage (after the data pipeline):
    python -m scripts.monthly_bills                  # contracted load estimated per meter
    python -m scripts.monthly_bills --load-kw 2      # same contracted load for every meter

Output: reports/monthly_bills.csv. Only complete hours are summed, so a month with missing
data is under-counted; `data_coverage_pct` shows how much of each month was actually measured.
Every amount is an ESTIMATE (see src/optimization/cost_estimation.py).
"""

import argparse
import logging

import pandas as pd

from src.config import HOURLY_CLEAN_PARQUET, REPORTS_DIR, setup_logging
from src.data.loader import load_parquet
from src.optimization.cost_estimation import (
    UPPCL_BILL_DISCLAIMER,
    UPPCL_TARIFF_SOURCE,
    estimate_contracted_load_kw,
    uppcl_monthly_bill,
)

logger = logging.getLogger("monthly_bills")
MONTHLY_BILLS_CSV = REPORTS_DIR / "monthly_bills.csv"


def monthly_bills(hourly: pd.DataFrame, load_kw: float | None = None) -> pd.DataFrame:
    meter = hourly["meter"].astype(str)
    month = hourly["timestamp"].dt.to_period("M")
    bills = (
        hourly.groupby([meter, month])
        .agg(kwh=("kwh", "sum"), measured_hours=("kwh", "size"))
        .reset_index()
        .rename(columns={"timestamp": "month"})
    )
    if load_kw is None:
        max_hourly = hourly.groupby(meter)["kwh"].max()
        bills["contracted_load_kw"] = bills["meter"].map(max_hourly).map(estimate_contracted_load_kw)
    else:
        bills["contracted_load_kw"] = load_kw

    hours_in_month = bills["month"].dt.days_in_month * 24
    bills["data_coverage_pct"] = (100 * bills["measured_hours"] / hours_in_month).round(1)
    bills["kwh"] = bills["kwh"].round(2)
    charges = pd.DataFrame(
        [uppcl_monthly_bill(k, load) for k, load in zip(bills["kwh"], bills["contracted_load_kw"], strict=True)]
    )
    bills["fixed_charge_inr"] = charges["fixed_charge"]
    bills["energy_charge_inr"] = charges["energy_charge"]
    bills["estimated_bill_inr"] = charges["total"]
    bills["month"] = bills["month"].astype(str)
    return bills


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-kw", type=float, default=None, help="contracted load for every meter (kW)")
    args = parser.parse_args()
    setup_logging()

    bills = monthly_bills(load_parquet(HOURLY_CLEAN_PARQUET), args.load_kw)
    bills.to_csv(MONTHLY_BILLS_CSV, index=False)
    logger.info("%d meter-month bills (%s) -> %s", len(bills), UPPCL_TARIFF_SOURCE, MONTHLY_BILLS_CSV)
    logger.info("%s", UPPCL_BILL_DISCLAIMER)


if __name__ == "__main__":
    main()
