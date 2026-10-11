# Modelling

Command: `python -m scripts.train` · Code: `src/models/`, `src/features/feature_engineering.py`

## Problem

Predict `kwh` (hourly energy) for a meter at hour **T** using only information available at the end of hour **T−1**.

## Features (16)

| Group | Features | Notes |
|---|---|---|
| Time | `hour`, `day_of_week` (0 = Mon), `month`, `day_of_month`, `is_weekend` | calendar effects |
| Lags | `lag_1h`, `lag_2h`, `lag_3h`, `lag_24h`, `lag_168h` | per meter, by **time** (gaps → NaN) |
| Rolling | `rolling_mean_3h/6h/24h/168h`, `rolling_std_24h`, `rolling_max_24h` | window = T−w … T−1 (`shift(1)` first); needs > 50% of window present |

**Deliberately excluded:** same-hour voltage/current/frequency. kWh ≈ V × I × pf × t, so same-hour current is
almost the answer itself (EDA correlation ≈ 1.0) — leakage — and it is unknown when forecasting. Their lagged
versions duplicate `lag_1h` and would block multi-step forecasting. Meter ID is not a feature: the
lag/rolling-level features already describe each meter's scale, and the model then works for new meters too.

## Leakage safeguards (each has a unit test in `tests/test_features.py`)

- Lags are computed by hour on the full grid, never across gaps or across meters.
- Rolling windows exclude the current hour.
- Changing the value at T leaves every feature at T unchanged.
- The predictor overwrites the target hour with NaN before computing features.

## Split

Chronological, global calendar cutoffs: first ~70% of rows → train, next ~15% → validation, last ~15% → test.
All meters share the cutoffs, so the test set is one unseen future period. A random split is not used because
neighbouring hours share almost the same lag values, so the model would effectively see the test data.

## Models

| Model | Idea | Tuning |
|---|---|---|
| Persistence baseline | ŷ(T) = y(T−1) | none |
| Seasonal-naive baseline | ŷ(T) = y(T−24) | none |
| Random Forest | average of many decorrelated trees | 2 configs (depth/leaf size) compared on validation |
| XGBoost | boosted trees, each correcting the last | 2 configs; early stopping on validation (≤ 2000 rounds) |

`random_state = 42` everywhere. Candidate grids live in `random_forest.py` / `xgboost_model.py`.
LSTM/GRU were not added: decide after seeing real-data results (ROADMAP decision point).

## Evaluation

| Metric | Meaning | Limitation |
|---|---|---|
| MAE (kWh) | average absolute error | treats a peak-hour miss like any other |
| RMSE (kWh) | penalises large errors | sensitive to outliers |
| R² | variance explained | can look high while absolute errors are large |
| NMAE (%) | MAE / mean kWh | scale-free; replaces MAPE |

**MAPE is not used**: many hours are at or near 0 kWh, and MAPE divides by the actual value.

**Selection:** best model = lowest **validation RMSE** (RMSE because large peak-hour errors matter most).
The test set is evaluated once and only reported. Losing candidates are never scored on test.

## Outputs

| File | Content |
|---|---|
| `artifacts/models/best_model.joblib` | trained best model |
| `artifacts/models/model_metadata.json` | features, hyperparameters, metrics, split cutoffs, data source, version |
| `artifacts/models/test_predictions.parquet` | test-period predictions of every model (dashboard back-test) |
| `reports/experiments/experiments.csv` | append-only log of every run & candidate |
| `reports/experiments/model_comparison.csv` | the comparison table |
| `reports/experiments/per_meter_test_metrics.csv` | does the model work for every meter? |
| `reports/experiments/feature_importance.csv` | tree-model feature importance |

## Results

Real CEEW 2020 data. Chronological split: train 196,332 (< 2020-09-04 06:00) · validation 42,090 · test 42,090
(>= 2020-11-02 22:00).

| Model | Val RMSE | Test MAE | Test RMSE | Test R² | Test NMAE % |
|---|---|---|---|---|---|
| Persistence (lag 1 h) | 0.275 | 0.125 | 0.277 | 0.238 | 52.9 |
| Seasonal naive (lag 24 h) | 0.312 | 0.133 | 0.288 | 0.177 | 56.3 |
| Random Forest | 0.220 | 0.109 | 0.222 | 0.509 | 46.1 |
| **XGBoost** (selected) | **0.219** | 0.110 | 0.222 | 0.508 | 46.2 |

Both tree models beat both baselines by ~20 % RMSE. XGBoost was selected on validation RMSE (and trains ~10×
faster); on test it is tied with Random Forest. The drop from validation R² ≈ 0.76 to test R² ≈ 0.51 is
seasonal shift: the test period is Nov–Dec (winter), unlike the summer-dominated training data.

## Forecast horizons

- **Next hour**: the evaluated task.
- **2–24 h**: recursive — each prediction is fed back as the next step's `lag_1h`. Errors accumulate; the API
  and dashboard say so explicitly. No accuracy claim is made for these horizons.
