"""Phase 13 — Estimated electricity cost.

    estimated_cost = consumption_kWh × tariff_per_kWh

IMPORTANT: The CEEW dataset contains no tariff or billing information. The tariff is a
configurable assumption (system settings), so every cost in this system is an ESTIMATE —
it ignores fixed charges, slabs, taxes and time-of-day pricing found on real bills.
"""

import math

import pandas as pd

ESTIMATED_COST_LABEL = "Estimated Cost"
COST_DISCLAIMER = (
    "Estimated using a flat, configurable tariff. Not an actual bill: excludes fixed charges, "
    "slab rates, taxes and time-of-day pricing."
)
MAX_TARIFF_PER_KWH = 1000.0  # sanity limit for input validation


def validate_tariff(tariff_per_kwh: float) -> float:
    if not 0 <= tariff_per_kwh <= MAX_TARIFF_PER_KWH:
        raise ValueError(f"tariff_per_kwh must be between 0 and {MAX_TARIFF_PER_KWH}")
    return float(tariff_per_kwh)


def estimate_cost(kwh: float, tariff_per_kwh: float) -> float:
    if kwh < 0:
        raise ValueError("consumption cannot be negative")
    return round(kwh * validate_tariff(tariff_per_kwh), 4)


def cost_breakdown(df: pd.DataFrame, tariff_per_kwh: float, freq: str = "D") -> pd.DataFrame:
    """Total kWh and estimated cost per period (D = day, W = week, MS = month)."""
    validate_tariff(tariff_per_kwh)
    grouped = df.set_index("timestamp")["kwh"].resample(freq).sum(min_count=1).dropna()
    return pd.DataFrame(
        {
            "period": grouped.index,
            "kwh": grouped.to_numpy(),
            "estimated_cost": (grouped * tariff_per_kwh).round(2).to_numpy(),
        }
    )


# ---------------------------------------------------------------------------
# UPPCL slab billing (monthly bills)
# ---------------------------------------------------------------------------
# Source: UPERC Tariff Order for UP State Discoms FY 2020-21 (11 Nov 2020), Rate Schedule LMV-1,
# section (c) "Other metered domestic consumers" (urban), page 552. The order kept the FY 2019-20
# tariff unchanged, so these rates apply to all of calendar year 2020.
# Slabs are telescopic: each rate applies only to the units inside its slab.
UPPCL_DOMESTIC_URBAN_SLABS = [(150, 5.50), (300, 6.00), (500, 6.50), (float("inf"), 7.00)]  # (upper kWh, Rs/kWh)
UPPCL_FIXED_CHARGE_PER_KW = 110.0  # Rs per kW of contracted load per month
UPPCL_TARIFF_SOURCE = "UPERC Tariff Order FY 2020-21, Rate Schedule LMV-1(c), urban domestic metered"
UPPCL_BILL_DISCLAIMER = (
    "Estimated with UPPCL urban domestic slab rates (FY 2020-21). Contracted load is estimated from the "
    "meter's highest hourly use. Excludes electricity duty, regulatory surcharge, FPPCA and rebates."
)


def uppcl_energy_charge(kwh: float) -> float:
    """Energy charge for one month's consumption under the UPPCL telescopic slabs."""
    if kwh < 0:
        raise ValueError("consumption cannot be negative")
    charge, lower = 0.0, 0.0
    for upper, rate in UPPCL_DOMESTIC_URBAN_SLABS:
        if kwh <= lower:
            break
        charge += (min(kwh, upper) - lower) * rate
        lower = upper
    return round(charge, 2)


def estimate_contracted_load_kw(max_hourly_kwh: float) -> int:
    """Contracted load is not in the dataset: use the highest hourly use (≈ average kW over that
    hour), rounded up to a whole kW, minimum 1 kW."""
    return max(1, math.ceil(max_hourly_kwh))


def uppcl_monthly_bill(kwh: float, contracted_load_kw: float) -> dict:
    fixed = round(contracted_load_kw * UPPCL_FIXED_CHARGE_PER_KW, 2)
    energy = uppcl_energy_charge(kwh)
    return {"fixed_charge": fixed, "energy_charge": energy, "total": round(fixed + energy, 2)}
