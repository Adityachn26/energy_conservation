# Development Roadmap

Each phase ends with: what was built · how to test it · what's next. We stop for review after each **milestone**.

Status: ✅ done · 🔄 in progress · ⏳ not started

## Milestone A — Foundation
| # | Phase | Key deliverables | Status |
|---|---|---|---|
| 1 | Project setup | Folder structure, README skeleton, DEPENDENCIES.md, ARCHITECTURE.md, roadmap, `.gitignore`, `.env.example`, pinned requirements | ✅ |

## Milestone B — Data pipeline (reproduces the Colab work as scripts)
| # | Phase | Key deliverables | Status |
|---|---|---|---|
| 2 | Data ingestion | Create venv, install core packages; `src/config.py`; `src/data/loader.py` (schema/dtype validation, logging); raw CSV → `data/interim/raw_3min.parquet` | ✅ |
| 3 | Data preprocessing | Check duplicates, missing values, value ranges (negative kWh, voltage outliers), timestamps on the 3-min grid; first pytest tests | ✅ |
| 4 | Hourly aggregation | `aggregation.py`: sum kWh, mean V/I/Hz, `readings_per_hour`; verify **378,624 / 47,256 / 331,368**; write `hourly_all` + `hourly_clean` | ✅ |
| 5 | EDA | Hourly/daily/weekly/monthly patterns, weekday vs weekend, per-meter, distribution, outliers, V/I/Hz correlations → `reports/figures/` | ✅ |

## Milestone C — Modelling
| # | Phase | Key deliverables | Status |
|---|---|---|---|
| 6 | Feature engineering | Time features; **gap-aware** per-meter lags (1, 2, 3, 24, 168 h) and shifted rolling stats; leakage unit tests | ✅ |
| 7 | Train/val/test split | Chronological split by **global timestamp cutoffs** (70/15/15) | ✅ |
| 8 | Baseline | Persistence (`lag_1h`) and seasonal-naive (`lag_24h`) baselines | ✅ |
| 9 | Random Forest | Train, tune on validation set, log experiment | ✅ |
| 10 | XGBoost | Train with early stopping on validation set, log experiment | ✅ |
| 11 | Model comparison | MAE / RMSE / R² table on test set, per-meter error, best model saved + metadata JSON | ✅ |
| — | *(Decision point)* | Is an LSTM/GRU worth adding? Decide based on results & time left | ⏳ |

## Milestone D — Analytics modules
| # | Phase | Key deliverables | Status |
|---|---|---|---|
| 12 | Peak detection | Configurable percentile threshold (per meter, computed on training period) | ✅ |
| 13 | Cost estimation | `cost = kWh × tariff`, configurable, always labelled **"Estimated Cost"** | ✅ |
| 14 | Recommendation engine | Rule-based recommendations from predictions, peaks and historical averages | ✅ |

## Milestone E — Application
| # | Phase | Key deliverables | Status |
|---|---|---|---|
| 15 | PostgreSQL | Schema (7 tables), indexes, Alembic migration, bulk loader script | ✅ |
| 16 | FastAPI backend | All endpoints, Pydantic schemas, model loaded once at start-up, logging, error handling | ✅ |
| 17 | Streamlit dashboard | Sidebar filters, KPIs, Plotly charts, all 7 sections | ✅ |

## Milestone F — Quality & delivery
| # | Phase | Key deliverables | Status |
|---|---|---|---|
| 18 | Testing | Full pytest suite incl. API tests, coverage report | ✅ |
| 19 | Dockerization | Dockerfile + `docker-compose.yml` (db, backend, dashboard) | 🔄 files written & validated; image build blocked by Docker Hub network errors |
| 20 | Documentation | Complete README + all docs/, final figures | ✅ |
| 21 | Final deployment | Clean-machine run-through, demo script, presentation assets | ⏳ after real-data run |

---

## ⚠️ Technical decisions recorded up front

These affect correctness. They're written down now so we don't forget them later.

### 1. Lag features must respect the 47,256 missing hours (Phase 6)
If lags are computed on `clean_hourly_df` (missing hours already removed) with
`groupby("meter")["t_kWh"].shift(1)`, then **"1 hour ago" can actually be days ago** whenever a
gap was removed. The model would be trained on mislabelled features.

**Correct approach:** compute lags/rolling stats on the **complete hourly grid per meter**
(all 378,624 meter-hours, with missing hours set to **NaN — not 0**), *then* keep only rows that are
(a) complete hours and (b) have the required lags available. Rows dropped this way are **counted and
logged**, never silently discarded.

> Please check this in your Colab lag code — it is the most common bug in this kind of project.

### 2. Rolling features use only the past (Phase 6)
`rolling_mean_24h` at time *T* = mean of *T−24 … T−1*, i.e. `shift(1)` **before** `.rolling(24)`.
We'll write a unit test that fails if the value at *T* leaks into its own feature.

### 3. Split by calendar time, not row count per meter (Phase 7)
All 46 meters share the same cutoff dates, so the test set is a genuinely unseen *future period* for every meter.
Lag/rolling features of validation/test rows may use earlier (training-period) values — that is legitimate,
because that history would exist in real life at prediction time.

### 4. Anything "learned" from data is learned from training data only
Peak thresholds (percentiles), any scalers or encoders → fitted on the training period only.

### 5. Context: 2020 was not a normal year
India's COVID-19 lockdown began on 25 March 2020. EDA should check for this shift in behaviour. It's a
**limitation to state in the report**: models trained on 2020 may not generalise to normal years.

### 6. Prediction horizon
Version 1 supports **next-hour** prediction (uses `lag_1h`). Next-24-hour forecasting requires either a
recursive strategy or models without short lags; we'll decide after Phase 11 and will not overclaim accuracy.

## Current state (2026-10-06)

Everything is implemented and verified end-to-end on **synthetic** data with the CEEW schema
(46 tests passing, 93% coverage; API and dashboard checked in a browser). Remaining:

1. Put the real CSV in `data/raw/` and run: `run_pipeline → run_eda → train → load_to_db`.
2. Check the pipeline's `OK/MISMATCH` lines against the Colab counts.
3. Fill the results table in README / MODELING from `reports/experiments/model_comparison.csv`.
4. Decide on LSTM/GRU based on those results.
5. Build the Docker images once Docker Hub is reachable.

## Open questions (to resolve during the phases)
- Timezone of `x_Timestamp` — assumed IST (`Asia/Kolkata`); verify in Phase 2.
- Does the CEEW metadata provide meter locations/types (residential/commercial)? Fills `meters.location`.
- Is a verified Bareilly/UP tariff available? If not, keep the configurable placeholder and say so.
