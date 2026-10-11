# Architecture

## 1. Big picture

The system has two clearly separated halves:

1. **Offline ML pipeline** (run occasionally, by a developer): raw CSV → clean hourly data → features → train & evaluate models → save the best model.
2. **Online application** (always running): PostgreSQL + FastAPI + Streamlit. It **loads** the saved model; it never retrains on start-up.

```
 ┌──────────────────────── OFFLINE: ML PIPELINE (scripts/, src/) ────────────────────────┐
 │                                                                                         │
 │  data/raw/CEEW ... Bareilly 2020.csv   (6.6M rows, 3-min, 46 meters)                    │
 │        │  src/data/loader.py         – read + validate schema & dtypes                  │
 │        ▼                                                                                │
 │  data/interim/raw_3min.parquet       – same data, fast columnar format                  │
 │        │  src/data/aggregation.py    – 3-min → hourly (sum kWh, mean V/I/Hz, count)     │
 │        ▼                                                                                │
 │  data/interim/hourly_all.parquet     – 378,624 meter-hours (incl. 47,256 empty hours)   │
 │        │  src/data/preprocessing.py  – mark empty hours as MISSING (NaN), never zero    │
 │        ▼                                                                                │
 │  data/processed/hourly_clean.parquet – 331,368 complete meter-hours (20/20 readings)    │
 │        │  src/features/feature_engineering.py – time + per-meter lag/rolling features   │
 │        ▼                                                                                │
 │  data/processed/features.parquet                                                         │
 │        │  src/models/train.py  – chronological split → Baseline / RF / XGBoost          │
 │        ▼                                                                                │
 │  artifacts/models/  best_model.joblib + model_metadata.json (features, metrics, date)   │
 │  reports/experiments/experiments.csv   reports/figures/*.png                            │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                         │ scripts/load_to_db.py (clean hourly data + metadata)
                         ▼
 ┌──────────────────────── ONLINE: APPLICATION (Docker Compose) ─────────────────────────┐
 │                                                                                         │
 │   ┌──────────────┐    SQL     ┌──────────────────────────────────────────┐             │
 │   │ PostgreSQL   │◀──────────▶│ FastAPI backend  (api/)                   │             │
 │   │ meters,      │            │  ├─ predictor  (loads .joblib once)       │             │
 │   │ consumption, │            │  ├─ peak detection                        │             │
 │   │ predictions, │            │  ├─ cost estimation (configurable tariff) │             │
 │   │ peaks, recs, │            │  └─ recommendation engine (rule-based)    │             │
 │   │ settings ... │            └──────────────────┬───────────────────────┘             │
 │   └──────────────┘                          HTTP/JSON                                   │
 │                                                  ▼                                      │
 │                                     Streamlit dashboard (dashboard/)                    │
 └──────────────────────────────────────────────────┬──────────────────────────────────────┘
                                                    ▼
                                                END USER

 Future scope: IoT smart meter → ingestion endpoint (e.g. POST /readings) → same DB / model / API.
```

**Key design rule:** the dashboard talks **only to the API**, never directly to the database.
That keeps business logic (peaks, cost, recommendations) in one place, and a future mobile app or IoT
gateway could reuse the same API.

## 2. Technology stack

| Layer | Technology | Why this, not something else |
|---|---|---|
| Language | Python 3.12 | One language for data, ML, API and dashboard. |
| Data processing | pandas + Parquet (pyarrow) | 6.6M rows fit comfortably in memory; Parquet is fast and keeps dtypes. Spark/Dask would be over-engineering. |
| ML | scikit-learn, XGBoost | Tree models are the right fit for tabular lag features; easy to explain. |
| Model storage | joblib + JSON metadata | Simple, standard; MLflow unnecessary at this scale. |
| Database | PostgreSQL 17 | See §3. |
| DB access | SQLAlchemy 2.x + psycopg 3 (+ Alembic) | Industry-standard ORM; safe parameterised queries; migrations. |
| API | FastAPI + Pydantic + uvicorn | Automatic validation and Swagger docs at `/docs`. |
| Dashboard | Streamlit + Plotly | Python-only interactive dashboard; React not justified. |
| Testing | pytest | De-facto Python standard. |
| Deployment | Docker Compose (3 services) | Reproducible one-command start-up. |

## 3. Why PostgreSQL (for the application)

The application needs data that is **shared, queried, and updated** by several components at once:

- **Concurrent access:** the API writes predictions while the dashboard reads history — a database handles that safely; files don't.
- **Relationships & integrity:** foreign keys guarantee a prediction always refers to a real meter and model.
  `UNIQUE (meter_id, timestamp)` stops duplicate readings.
- **Fast filtered queries:** "meter 12, March 2020, hourly" uses the `(meter_id, timestamp)` index instead of scanning a file.
- **Mutable settings:** the tariff and peak percentile are changed via `PUT /settings` and must persist.
- **Production realism:** PostgreSQL is free, open-source, widely used, and has excellent time/date support.
  (SQLite would work for a demo but handles concurrent writes poorly and is less representative of production.)

## 4. Why raw data stays in CSV/Parquet during ML experimentation

- **Load-once, scan-everything workload:** training reads the *entire* dataset in one go. `pd.read_parquet()`
  is much faster than pulling millions of rows through a database driver.
- **Raw data is immutable:** we never update raw readings, so a database's transactions/updates add nothing.
- **Reproducibility:** `data/raw/` is never modified; every derived file can be regenerated by a script.
- **Size:** 6.6M raw rows would bloat the DB, backups and Docker volumes, while the app only needs the
  **hourly** granularity (331k rows) — 20× smaller.
- **Separation of concerns:** experimenting with new features never risks breaking the live application.

So: **files for the ML lab, PostgreSQL for the running application.**

## 5. Changes from the originally suggested structure

| Change | Reason |
|---|---|
| `models/saved/` → **`artifacts/models/`** | The word "models" otherwise means three things: ML code (`src/models/`), ORM tables (`database/models.py`) and binary files. `artifacts/` makes it obvious these are generated outputs. |
| Added **`data/interim/`** | Clear stages: `raw` (never touched) → `interim` (converted/aggregated, incl. empty hours) → `processed` (model-ready). Makes the "47,256 missing hours" step visible and auditable instead of silent. |
| Added **`reports/experiments/`** | Simple CSV/JSON experiment log (instead of MLflow). |
| Added **`pyproject.toml`** (tool config only) | Lets pytest import `src`, `api`, `database` from the project root; holds ruff settings. |
| Split **`requirements.txt`** / **`requirements-dev.txt`** | Docker images install only runtime packages; pytest, matplotlib, notebooks stay out of production. |
| Added **`docs/ROADMAP.md`** | Tracks the 21 phases and their status. |
| Added **`api/services.py`**, **`api/dependencies.py`**, **`database/crud.py`** | Routes stay thin; logic that combines DB + model + optimisation lives in services, reusable queries in `crud.py` (also used by `scripts/load_to_db.py`). |
| Routes grouped into 4 modules (`health`, `consumption`, `predictions`, `analytics`) | 13 small endpoints don't need 10 files. |
| **`scripts/` is a Python package** (`python -m scripts.<name>`) | Lets scripts `import src...` from the project root without path hacks. |
| **`src/data/synthetic.py`** + `scripts/generate_sample_data.py` | Build and test the whole system before the real CSV is available; synthetic runs are tagged end-to-end. |
| **`dashboard/charts.py`**, **`dashboard/api_client.py`** | One consistent chart style; one place for HTTP and error handling. |
| EDA as **`scripts/run_eda.py`** instead of notebooks | Reproducible figures from one command; notebooks remain optional for exploration. |
| **`.streamlit/config.toml`** | Dashboard theme matches the chart palette. |

## 6. Folder structure

```
smart-energy-system/
├── data/            raw/ (original CSV, read-only) · interim/ · processed/ · external/   [git-ignored]
├── notebooks/       exploration & EDA only — logic is moved into src/ once it works
├── src/             reusable pipeline code (importable as `src.*`)
│   ├── config.py    paths + settings from environment
│   ├── data/        loader.py · preprocessing.py · aggregation.py · synthetic.py
│   ├── features/    feature_engineering.py
│   ├── models/      split.py · baseline.py · random_forest.py · xgboost_model.py · evaluate.py · train.py
│   ├── prediction/  predictor.py
│   └── optimization/ peak_detection.py · cost_estimation.py · recommendations.py
├── api/             FastAPI app: main.py · routes/ · schemas/ · services.py · dependencies.py
├── database/        database.py (engine/session) · models.py (ORM tables) · crud.py · migrations/ (Alembic)
├── dashboard/       Streamlit app: app.py · charts.py · api_client.py
├── artifacts/models/ trained .joblib + metadata JSON                                   [git-ignored]
├── reports/         figures/ (EDA plots) · experiments/ (metrics log)
├── scripts/         CLI entry points: generate_sample_data · run_pipeline · run_eda · train · load_to_db
├── tests/           pytest test suite
└── docs/            ARCHITECTURE · DEPENDENCIES · DATA_PIPELINE · MODELING · API · DATABASE · ROADMAP
```

**`src/` vs `scripts/`:** `src/` holds *functions* (testable, reusable); `scripts/` holds thin *entry points*
that call those functions in order. The same `src/` feature code is used by both training and the API's
predictor — this guarantees training and serving compute features identically (no "training/serving skew").

## 7. Runtime environments

| Environment | Database | How to start |
|---|---|---|
| Local development (verified) | Homebrew PostgreSQL 17 on `localhost:5432` | `uvicorn …` + `streamlit run …` |
| Docker Compose (written, not yet built — see README §9) | `postgres:17-alpine` container, host port 5433 | `docker compose up -d --build` |
| Tests | in-memory SQLite (`DATABASE_URL=sqlite://`) | `pytest` |

The ORM models use only portable types, so the same code runs on PostgreSQL and on SQLite in tests.
