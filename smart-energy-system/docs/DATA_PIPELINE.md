# Data Pipeline

Command: `python -m scripts.run_pipeline [--input <csv>]` · Code: `src/data/`, `src/features/`

## Stages

| # | Stage | Code | Output | Rows (real file) |
|---|---|---|---|---|
| 1 | Load & validate schema | `loader.load_raw_csv` | in memory | 6,627,360 |
| 2 | Quality report | `preprocessing.data_quality_report` | `reports/data_quality.json` | — |
| 3 | Safety-net cleaning (logged) | `preprocessing.clean_raw_data` | `data/interim/raw_3min.parquet` | 6,627,360 (0 removed) |
| 4 | 3-min → hourly on a per-meter grid | `aggregation.aggregate_to_hourly` | `data/interim/hourly_all.parquet` | 378,624 |
| 5 | Complete hours only | `aggregation.select_complete_hours` | `data/processed/hourly_clean.parquet` | 331,368 |
| 6 | Features + usable rows | `feature_engineering.compute_features` / `select_model_rows` | `data/processed/features.parquet` | logged |

On the real file, step 6's log and `reports/data_quality.json → features` state exactly how many rows were dropped
because the target hour was incomplete vs because a required lag/rolling window fell inside a gap.

The pipeline compares its counts with the Colab numbers (`EXPECTED_REAL_DATA` in `src/config.py`) and logs
`OK` / `MISMATCH` for each.

## Column renaming

| Original | Internal |
|---|---|
| `x_Timestamp` | `timestamp` |
| `t_kWh` | `kwh` |
| `z_Avg Voltage (Volt)` | `voltage` |
| `z_Avg Current (Amp)` | `current` |
| `y_Freq (Hz)` | `frequency` |
| `meter` | `meter` |

Short snake_case names avoid quoting spaces/brackets everywhere in code, SQL and the API.

## Aggregation rules (3-min → hourly)

| Column | Rule | Why |
|---|---|---|
| `kwh` | SUM | energy accumulates over the hour |
| `voltage`, `current`, `frequency` | MEAN | instantaneous quantities |
| `readings_per_hour` | COUNT | data-completeness check (20 = complete) |

## Missing-hour policy

1. The hourly grid spans each meter's first to last reading, so hours with no readings appear **as rows**
   (`readings_per_hour = 0`, `kwh = NaN`). This reproduces the 47,256 empty hours found in Colab.
2. Empty hours are **missing data**, never zero consumption: not interpolated, not zero-filled, not training targets.
3. Partial hours (1–19 readings) would under-count the sum, so their values are also treated as missing.
   (The real 2020 file has none.)
4. Lags/rolling features are computed on this grid, so a gap produces `NaN` features instead of silently
   reaching back to an older reading. Rows with missing features are dropped and **counted**.

## Synthetic data

`python -m scripts.generate_sample_data` writes `data/raw/synthetic_sample.csv` with the identical schema
(6 meters × 120 days, with offline blocks). Any run on it is tagged `is_synthetic: true` all the way to the
dashboard banner. It exists only so the system can be built and tested without the real file.
