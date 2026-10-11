"""SYNTHETIC smart-meter data with the exact CEEW CSV schema — for development and tests only.

This is NOT real data. It exists so the full pipeline, API and dashboard can be built and tested
before the real CEEW file is available. All results produced from it are meaningless for the
report; re-run everything on the real CSV before presenting results.

The generator mimics the broad shape of household consumption: a daily cycle with morning and
evening peaks, weekend differences, a summer cooling bump, per-meter scale differences, random
noise, and blocks of missing hours (meter offline) like the real data.
"""

import numpy as np
import pandas as pd

from src.config import READING_INTERVAL_MINUTES


def generate_synthetic_readings(
    n_meters: int = 6,
    start: str = "2020-01-01",
    days: int = 120,
    missing_block_count: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with the ORIGINAL CEEW column names, at 3-minute resolution."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start, periods=days * 24 * (60 // READING_INTERVAL_MINUTES), freq=f"{READING_INTERVAL_MINUTES}min"
    )
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    day_of_year = timestamps.dayofyear.to_numpy()
    is_weekend = (timestamps.dayofweek.to_numpy() >= 5).astype(float)

    # Daily profile (kW): low at night, morning bump ~8h, larger evening peak ~20h.
    daily_profile = 0.15 + 0.25 * np.exp(-((hour - 8) ** 2) / 4) + 0.55 * np.exp(-((hour - 20) ** 2) / 5)
    seasonal = 1 + 0.5 * np.exp(-((day_of_year - 160) ** 2) / (2 * 40**2))  # summer cooling load
    weekend = 1 + 0.15 * is_weekend

    frames = []
    for meter_index in range(n_meters):
        scale = rng.uniform(0.5, 2.0)
        noise = rng.lognormal(mean=0, sigma=0.35, size=len(timestamps))
        power_kw = scale * daily_profile * seasonal * weekend * noise
        kwh = power_kw * READING_INTERVAL_MINUTES / 60  # energy in one 3-minute interval

        voltage = rng.normal(230, 6, size=len(timestamps))
        current = power_kw * 1000 / (voltage * 0.9)  # P = V * I * power factor
        frequency = rng.normal(50, 0.05, size=len(timestamps))

        frame = pd.DataFrame(
            {
                "x_Timestamp": timestamps,
                "t_kWh": kwh.round(4),
                "z_Avg Voltage (Volt)": voltage.round(2),
                "z_Avg Current (Amp)": current.round(3),
                "y_Freq (Hz)": frequency.round(3),
                "meter": f"SYN_{meter_index + 1:02d}",
            }
        )

        # Remove random blocks of whole hours (meter offline) -> become "empty hours".
        keep = np.ones(len(frame), dtype=bool)
        readings_per_hour = 60 // READING_INTERVAL_MINUTES
        for _ in range(missing_block_count):
            block_hours = int(rng.integers(2, 30))
            start_hour = int(rng.integers(200, days * 24 - block_hours - 1))
            keep[start_hour * readings_per_hour : (start_hour + block_hours) * readings_per_hour] = False
        frames.append(frame.loc[keep])

    return pd.concat(frames, ignore_index=True)
