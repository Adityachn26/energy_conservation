# AI-Based Smart Energy Consumption Prediction & Optimization System

> Final-year B.Tech (AI & Data Science) capstone project.
> **Status:** all phases run end-to-end on the **real** CEEW Bareilly 2020 file (6,627,360 readings, 46 meters);
> pipeline counts match the Colab analysis exactly (378,624 / 47,256 / 331,368).
> Progress: [docs/ROADMAP.md](docs/ROADMAP.md)

## 1. Project overview

A software-based AI energy-management system that uses historical smart-meter data to:

1. **Predict** next-hour electricity consumption per meter (and a clearly-labelled recursive 1–24 h forecast)
2. **Analyse** historical consumption patterns (hourly, daily, weekly, monthly, weekday vs weekend, per meter)
3. **Detect peak demand** with a configurable per-meter percentile threshold
4. **Estimate cost** — monthly bills per meter with **UPPCL urban domestic slab rates** (FY 2020-21), plus a
   configurable flat tariff for hourly forecasts; always labelled **"Estimated"**, never a real bill
5. **Recommend** energy-saving actions (rule-based, transparent)
6. **Compare models** objectively: two baselines vs Random Forest vs XGBoost on a chronological test set
7. Serve everything through a **FastAPI** backend, **PostgreSQL** database and **Streamlit** dashboard

**Out of scope:** physical smart-meter deployment, IoT hardware, direct appliance control. The system only
*recommends*; the API is designed so that live smart-meter data could be ingested later (future scope).

## 2. Architecture

```
CEEW CSV ─► data pipeline (pandas) ─► Parquet (interim / processed)
         ─► features (per-meter lags & shifted rolling stats)
         ─► Baseline / Random Forest / XGBoost ─► evaluation ─► best model (.joblib + metadata.json)
                                                                     │
        PostgreSQL ◄──── scripts/load_to_db ◄─────────────────────────┤
            ▲                                                         ▼
            └──────────── FastAPI (predict · peaks · cost · recommendations) ─► Streamlit dashboard
```

Details and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## 3. Dataset

**CEEW – Smart meter data, Bareilly 2020** (`CEEW - Smart meter data Bareilly 2020.csv`)

| Property | Value |
|---|---|
| Rows / columns | 6,627,360 / 6 |
| Meters | 46 |
| Interval | 3 minutes (20 readings per hour) |
| Period | 2020-01-01 00:00 → 2020-12-31 23:57 |
| Columns | `x_Timestamp`, `t_kWh`, `z_Avg Voltage (Volt)`, `z_Avg Current (Amp)`, `y_Freq (Hz)`, `meter` |
| Hourly meter-hours | 378,624 — of which 47,256 have **no readings** (kept as *missing*, never as 0 kWh) |
| Clean hourly records | **331,368** (all 20 readings present) |

The pipeline automatically checks these numbers when run on the real file. Details: [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)

## 4. Quick start with the real CEEW data

```bash
cd smart-energy-system
source .venv/bin/activate                    # see §5 for first-time setup

# 1. Put the CSV here (git-ignored):
#    data/raw/CEEW - Smart meter data Bareilly 2020.csv
python -m scripts.run_pipeline               # raw -> hourly -> features   (logs the 378,624 / 47,256 / 331,368 checks)
python -m scripts.run_eda                    # figures -> reports/figures/
python -m scripts.train                      # baselines, RF, XGBoost -> artifacts/models/
python -m scripts.load_to_db                 # replaces the synthetic demo data in PostgreSQL

uvicorn api.main:app --reload                # terminal 1 -> http://localhost:8000/docs
streamlit run dashboard/app.py               # terminal 2 -> http://localhost:8501
```

The synthetic banner in the dashboard disappears automatically once the model is trained on the real file.

## 5. Installation (first time)

Prerequisites (macOS): Python 3.12, Homebrew.

```bash
brew install libomp                          # OpenMP runtime required by XGBoost on macOS
brew install postgresql@17                   # local database (or use Docker, §9)
brew services start postgresql@17

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt          # runtime + testing/EDA tools
```

Every dependency is explained in [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## 6. Environment variables

```bash
cp .env.example .env                         # then set a real POSTGRES_PASSWORD
```

| Variable | Purpose | Default |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Database credentials (never hardcoded) | — |
| `POSTGRES_HOST` / `POSTGRES_PORT` | Database location | `localhost` / `5432` |
| `DATABASE_URL` | Optional full URL; overrides the above (tests use `sqlite://`) | — |
| `DEFAULT_TARIFF_PER_KWH` | Initial tariff for estimated cost (placeholder, not verified) | `7.0` |
| `DEFAULT_PEAK_PERCENTILE` | Initial peak percentile | `90` |
| `APP_TIMEZONE` | Timezone of the meter timestamps | `Asia/Kolkata` |
| `API_BASE_URL` | Where the dashboard finds the API | `http://localhost:8000` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `DOCKER_DB_PORT` | Host port of the Docker database | `5433` |

## 7. Database setup (local PostgreSQL)

```bash
set -a; source .env; set +a
psql -d postgres -c "CREATE ROLE $POSTGRES_USER LOGIN PASSWORD '$POSTGRES_PASSWORD';"
psql -d postgres -c "CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER;"
alembic upgrade head                         # creates the 7 tables
python -m scripts.load_to_db                 # loads hourly data, model results, peaks, recommendations
```

Schema and design: [docs/DATABASE.md](docs/DATABASE.md)

## 8. Running locally

| Task | Command |
|---|---|
| Generate synthetic demo data (no CSV needed) | `python -m scripts.generate_sample_data` |
| Data pipeline (real file) | `python -m scripts.run_pipeline` |
| Data pipeline (synthetic) | `python -m scripts.run_pipeline --input data/raw/synthetic_sample.csv` |
| EDA figures | `python -m scripts.run_eda` |
| Train & compare models | `python -m scripts.train` |
| Load database | `python -m scripts.load_to_db` |
| Estimated monthly UPPCL bill per meter | `python -m scripts.monthly_bills [--load-kw 2]` → `reports/monthly_bills.csv` |
| Start backend | `uvicorn api.main:app --reload` → <http://localhost:8000/docs> |
| Start dashboard | `streamlit run dashboard/app.py` → <http://localhost:8501> |
| Run tests | `pytest` (add `--cov=src --cov=api --cov=database` for coverage) |
| Lint / format | `ruff check .` / `ruff format .` |

## 9. Docker

```bash
docker compose up -d --build                                   # db + backend + dashboard
docker compose run --rm backend python -m scripts.load_to_db   # first time: load data
```

- Dashboard: <http://localhost:8501> · API: <http://localhost:8000/docs> · DB on host port `5433`
- The ML pipeline runs on the host (not containerised on purpose); `artifacts/` and `data/processed/`
  are mounted read-only into the containers.
- ⚠️ The Compose file is validated (`docker compose config`), but the images were **not yet built** on the
  development machine because Docker Hub downloads were failing from the Colima VM (network issue).
  Re-test with `docker compose up --build` on a working network.

## 10. API documentation

Interactive Swagger docs at `/docs`. Endpoint reference: [docs/API.md](docs/API.md)

`GET /health` · `GET /meters` · `GET /consumption` · `GET /consumption/by-meter` · `GET /overview` ·
`POST /predict` · `GET /predictions` · `GET /peaks` · `GET /peaks/summary` · `GET /recommendations` ·
`GET /model-performance` · `GET /settings` · `PUT /settings`

## 11. Model methodology & evaluation

- **Target:** hourly kWh per meter, **next-hour** forecast
- **Features (16):** hour, day of week, month, day of month, weekend flag; lags 1/2/3/24/168 h;
  rolling mean 3/6/24/168 h, rolling std & max 24 h — all per meter, computed on the full hourly grid
  so gaps stay gaps, and rolling windows end at T−1 (no leakage)
- **Split:** chronological 70 / 15 / 15 with global calendar cutoffs
- **Selection:** hyperparameters and best model chosen on **validation RMSE**; the **test** set is used once, for reporting
- **Metrics:** MAE, RMSE, R², NMAE (MAE as % of mean). MAPE is not used (near-zero hours make it explode)

Test set (Nov–Dec 2020, 42,090 meter-hours, kWh per hour):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline — persistence (lag 1 h) | 0.125 | 0.277 | 0.238 |
| Baseline — seasonal naive (lag 24 h) | 0.133 | 0.288 | 0.177 |
| Random Forest | 0.109 | 0.222 | 0.509 |
| **XGBoost (selected on validation RMSE)** | 0.110 | 0.222 | 0.508 |

Validation R² was ~0.76 but test R² is ~0.51: the test period is winter, when consumption is lower and
less regular than the summer-heavy training period. Source: `reports/experiments/model_comparison.csv`.
Details: [docs/MODELING.md](docs/MODELING.md)

## 12. Limitations

- One city (Bareilly), one year (2020), 46 meters — and 2020 includes the COVID-19 lockdown (from 25 March),
  so patterns may not generalise to normal years.
- 47,256 meter-hours have no readings; they are excluded, not imputed. Rows whose lag/rolling features fall in a
  gap are also excluded from training (counts are logged in `reports/data_quality.json`).
- Monthly bills use UPPCL slab rates + fixed charge (UPERC Tariff Order FY 2020-21, LMV-1(c), p. 552), but the
  contracted load is **estimated** (highest hourly kWh, rounded up, min 1 kW) because the dataset doesn't include it,
  and electricity duty, regulatory surcharge, FPPCA and rebates are excluded. Months with missing hours are under-counted.
- API/dashboard costs for hourly forecasts use a flat, configurable tariff (slabs only make sense per month).
- Only next-hour accuracy is measured; 2–24 h forecasts are recursive and less accurate.
- Peak thresholds use each meter's full stored history (descriptive analytics), not only the training period.
- Recommendations are rule-based heuristics, not optimised schedules.

## 13. Future scope

- Live IoT / smart-meter ingestion endpoint feeding the same database, model and API
- Deep-learning forecasters (LSTM/GRU) and direct multi-horizon models
- Weather and holiday features; time-of-day tariffs; demand-response scheduling
- Authentication for the settings endpoint; model monitoring and scheduled retraining

## 14. Project structure

```
smart-energy-system/
├── src/            pipeline code: data/ · features/ · models/ · prediction/ · optimization/ · config.py
├── api/            FastAPI: main.py · routes/ · schemas/ · services.py · dependencies.py
├── database/       models.py (tables) · database.py (engine) · crud.py (queries) · migrations/ (Alembic)
├── dashboard/      app.py · charts.py · api_client.py
├── scripts/        generate_sample_data · run_pipeline · run_eda · train · load_to_db
├── tests/          46 pytest tests (pipeline, leakage guards, models, optimisation, API)
├── data/           raw/ · interim/ · processed/           (git-ignored)
├── artifacts/      trained model + metadata               (git-ignored)
├── reports/        figures/ · experiments/ · data_quality.json · eda_summary.json
└── docs/           ARCHITECTURE · DEPENDENCIES · DATA_PIPELINE · MODELING · API · DATABASE · ROADMAP
```
