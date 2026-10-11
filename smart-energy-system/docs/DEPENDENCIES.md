# Dependencies

This document explains **every** direct dependency of the project: why it exists, where it is used,
and what would happen without it. It is updated whenever a package is added, removed, or upgraded.

- **Python version:** 3.12 (see [Python version choice](#python-version-choice))
- **Versions verified:** 2026-10-06 against PyPI, with a full `pip install --dry-run` resolution of
  `requirements-dev.txt` on Python 3.12 (95 packages total incl. transitive — no conflicts).
- **Files:**
  - `requirements.txt` — runtime (what the pipeline / API / dashboard need to run)
  - `requirements-dev.txt` — development only (tests, linting, notebooks, static plots). Includes `requirements.txt`.

> **Status legend:** ✅ installed (in `.venv`, Python 3.12.13) & in use

---

## Summary table

| Package | Version | Category | Prod/Dev | Mandatory? | Introduced in | Status |
|---|---|---|---|---|---|---|
| pandas | 3.0.6 | Core data | Prod | Yes | Phase 2 | ✅ |
| numpy | 2.5.3 | Core data | Prod | Yes | Phase 2 | ✅ |
| pyarrow | 25.0.1 | Core data | Prod | Yes | Phase 2 | ✅ |
| scikit-learn | 1.9.1 | ML | Prod | Yes | Phase 8 | ✅ |
| xgboost | 3.4.1 | ML | Prod | Yes | Phase 10 | ✅ |
| joblib | 1.6.0 | ML | Prod | Yes | Phase 9 | ✅ |
| python-dotenv | 1.2.4 | Config | Prod | Yes | Phase 2 | ✅ |
| sqlalchemy | 2.1.3 | Database | Prod | Yes | Phase 15 | ✅ |
| psycopg[binary] | 3.3.6 | Database | Prod | Yes | Phase 15 | ✅ |
| alembic | 1.20.0 | Database | Prod | Recommended | Phase 15 | ✅ |
| fastapi | 0.142.2 | Backend | Prod | Yes | Phase 16 | ✅ |
| uvicorn | 0.54.0 | Backend | Prod | Yes | Phase 16 | ✅ |
| pydantic | 2.13.5 | Backend | Prod | Yes | Phase 16 | ✅ |
| streamlit | 1.65.0 | Dashboard | Prod | Yes | Phase 17 | ✅ |
| plotly | 7.1.0 | Dashboard | Prod | Yes | Phase 17 | ✅ |
| httpx | 0.28.1 | Dashboard/Test | Prod | Yes | Phase 16 | ✅ |
| pytest | 9.1.1 | Testing | Dev | Yes | Phase 3 | ✅ |
| pytest-cov | 7.1.0 | Testing | Dev | Optional | Phase 18 | ✅ |
| ruff | 0.16.10 | Code quality | Dev | Optional | Phase 2 | ✅ |
| matplotlib | 3.11.2 | EDA | Dev | Yes (for report figures) | Phase 5 | ✅ |
| ipykernel | 7.4.0 | Notebooks | Dev | Optional | Phase 5 | ✅ |

**Non-Python tools (installed with Homebrew):**

| Tool | Version | Why | Mandatory? |
|---|---|---|---|
| `libomp` | latest | OpenMP runtime; `import xgboost` fails on macOS without it | Yes (macOS) |
| `postgresql@17` | 17.11 | Local database for development (`brew services start postgresql@17`) | Yes, unless using Docker's DB |
| `docker-compose` | 5.6.0 | Compose v2 plugin (the previous one pointed at an uninstalled Docker Desktop; the plugin link in `~/.docker/cli-plugins` was repointed to Homebrew's) | For Phase 19 |
| Docker engine via Colima | existing | Runs containers | For Phase 19 |
| Git | existing | Version control | Recommended |

Inside the Docker image, `libgomp1` (Debian) plays the role of `libomp`.

---

## Core data & ML

### pandas — 3.0.6
- **Why:** Load the 6.6M-row CSV, aggregate 3-min → hourly, build lag/rolling features per meter (`groupby("meter").shift()`), chronological splitting.
- **Used in:** `src/data/*`, `src/features/*`, `src/models/*`, `api/`, `dashboard/`.
- **Without it:** We would hand-write groupby/resample logic in NumPy — slow to write and error-prone.
- **Brings in:** numpy, python-dateutil (date parsing), tzdata.
- **Note:** pandas 3.0 enables **Copy-on-Write** and a dedicated **string dtype** by default. Code written in Colab
  on pandas 2.x mostly works, but chained assignment like `df[df.x > 0]["y"] = 1` silently does nothing — we use `.loc` everywhere.

### numpy — 2.5.3
- **Why:** Fast numerical arrays; the foundation under pandas, scikit-learn and xgboost. Used directly for metrics (RMSE = `np.sqrt(...)`) and percentiles.
- **Used in:** everywhere, mostly indirectly.
- **Without it:** Nothing in the ML stack works — it is a hard requirement of pandas/sklearn/xgboost.
- **Brings in:** nothing.

### pyarrow — 25.0.1
- **Why:** Engine pandas uses to read/write **Parquet** files. Parquet is columnar and compressed: the
  hourly dataset as Parquet is typically ~5–10× smaller than CSV and loads much faster, and it preserves dtypes (timestamps stay timestamps).
- **Used in:** `src/data/*` (save `data/interim/*.parquet`, `data/processed/*.parquet`).
- **Without it:** We'd be limited to CSV — re-parsing 6.6M timestamps on every run (slow) and losing dtypes.
- **Brings in:** nothing significant (large binary wheel, ~40 MB).

### scikit-learn — 1.9.1
- **Why:** `RandomForestRegressor`, metrics (`mean_absolute_error`, `root_mean_squared_error`, `r2_score`).
- **Used in:** `src/models/random_forest.py`, `src/models/evaluate.py`.
- **Without it:** We'd have to implement Random Forest and metrics ourselves — not sensible.
- **Brings in:** **scipy** (scientific computing, ~35 MB), threadpoolctl (controls CPU threads), joblib.

### xgboost — 3.4.1
- **Why:** Gradient-boosted trees — usually the strongest model on tabular lag/rolling features. One of the three required models.
- **Used in:** `src/models/xgboost_model.py`, and at inference time in `src/prediction/predictor.py` (if it is the best model).
- **Without it:** We could use sklearn's `HistGradientBoostingRegressor` instead, but the project brief explicitly asks for XGBoost.
- **Brings in:** numpy, scipy.
- **⚠️ macOS:** XGBoost needs the OpenMP runtime. If `import xgboost` fails with a `libomp.dylib` error, run `brew install libomp`.

### joblib — 1.6.0
- **Why:** Save/load trained models (`joblib.dump(model, "artifacts/models/xgboost_model.joblib")`). Efficient for objects containing large NumPy arrays (Random Forests are large).
- **Used in:** `src/models/train.py` (save), `src/prediction/predictor.py` (load once at API start-up).
- **Without it:** Python's `pickle` works but is slower/larger for big arrays. (Already a scikit-learn dependency — we pin it because we import it directly.)
- **⚠️ Security:** Never load a `.joblib` file from an untrusted source — loading can execute arbitrary code.

---

## Configuration

### python-dotenv — 1.2.4
- **Why:** Reads the local `.env` file into environment variables so credentials are **never hardcoded**.
- **Used in:** `src/config.py` (single place that reads settings).
- **Without it:** You'd need to `export POSTGRES_PASSWORD=...` in every terminal by hand. In Docker, variables are injected by Compose and dotenv is simply a no-op.
- **Brings in:** nothing.
- **Considered and rejected:** `pydantic-settings` — nicer typed settings, but one more dependency for a small config file.

---

## Database (Phase 15)

### sqlalchemy — 2.1.3
- **Why:** ORM + SQL toolkit. We define tables as Python classes (`database/models.py`) and use the modern 2.x `select()` style. Protects against SQL injection via parameter binding.
- **Used in:** `database/`, `api/`, `scripts/load_to_db.py`.
- **Without it:** We'd write raw SQL strings by hand — more error-prone and harder to test.
- **Brings in:** typing-extensions, greenlet (only on some platforms).

### psycopg[binary] — 3.3.6
- **Why:** The actual driver that talks to PostgreSQL. SQLAlchemy uses it underneath (`postgresql+psycopg://...`). psycopg 3 is the modern successor of psycopg2.
- **`[binary]` extra:** ships a pre-compiled `libpq`, so you don't need PostgreSQL installed on your Mac to connect.
- **Used in:** indirectly via SQLAlchemy; also `COPY` for fast bulk loading of ~331k rows.
- **Without it:** SQLAlchemy cannot connect to PostgreSQL at all.
- **Brings in:** psycopg-binary.

### alembic — 1.20.0
- **Why:** Database **migrations** — version-controlled schema changes (`database/migrations/`). If you add a column later, Alembic upgrades the DB without dropping data.
- **Used in:** `database/migrations/`.
- **Without it:** We could use `Base.metadata.create_all()` (fine for a first version), but changing the schema later means dropping and recreating tables.
- **Mandatory?** Recommended, not strictly required. We'll decide definitively in Phase 15.
- **Brings in:** Mako (templating for migration files), MarkupSafe.

---

## Backend API (Phase 16)

### fastapi — 0.142.2
- **Why:** REST API framework. Automatic request validation (via Pydantic) and automatic interactive docs at `/docs`.
- **Used in:** `api/`.
- **Without it:** Flask is the alternative, but needs extra libraries for validation and API docs.
- **Brings in:** **starlette** (the underlying web toolkit), pydantic, anyio, python-multipart.

### uvicorn — 0.54.0
- **Why:** The ASGI **server** that actually runs the FastAPI app (`uvicorn api.main:app`). FastAPI defines the app; uvicorn serves it over HTTP.
- **Used in:** start-up command; Docker `CMD`.
- **Without it:** The FastAPI app has nothing to run it.
- **Brings in:** click, h11.
- **Note:** We install plain `uvicorn`, not `uvicorn[standard]`, to avoid extra packages we don't need.

### pydantic — 2.13.5
- **Why:** Data validation via Python type hints. API request/response schemas (`api/schemas/`) — e.g. rejects a negative tariff or an unknown meter with a clear 422 error.
- **Used in:** `api/schemas/`.
- **Without it:** FastAPI requires it — not optional. We pin it explicitly because we import it directly.
- **Brings in:** pydantic-core (Rust-compiled, fast), annotated-types, typing-inspection.

---

## Dashboard (Phase 17)

### streamlit — 1.65.0
- **Why:** Build the interactive dashboard in pure Python — no HTML/JS/React needed.
- **Used in:** `dashboard/app.py`.
- **Without it:** We'd need a React/Vue frontend — far more work, and not justified for this project.
- **Brings in (notable):** **altair**, pydeck, tornado (its web server), protobuf, pillow, requests. Streamlit is the heaviest dependency (~30 transitive packages).

### plotly — 7.1.0
- **Why:** Interactive charts (hover, zoom, range sliders) for consumption trends, actual-vs-predicted, peak heatmaps. Renders natively in Streamlit with `st.plotly_chart`.
- **Used in:** `dashboard/`.
- **Without it:** Streamlit's built-in charts are too limited (no dual-axis, weak heatmaps).
- **Brings in:** narwhals (dataframe compatibility layer).

### httpx — 0.28.1
- **Why:** Two jobs with one library: (1) the dashboard calls the FastAPI backend over HTTP; (2) FastAPI's `TestClient` requires httpx for API tests.
- **Used in:** `dashboard/`, `tests/test_api*.py`.
- **Without it:** API tests won't run. We'd also need `requests` separately (streamlit already brings `requests`, but using one HTTP library consistently is cleaner).
- **Brings in:** httpcore, anyio, certifi, idna.

---

## Development only

### pytest — 9.1.1
- **Why:** Unit tests for preprocessing, feature engineering (especially **leakage checks**), peak detection, cost estimation, recommendations, API.
- **Used in:** `tests/`.
- **Brings in:** pluggy, iniconfig, packaging.

### pytest-cov — 7.1.0 (optional)
- **Why:** Shows which lines of code the tests actually execute (`pytest --cov=src`). Useful evidence for the capstone report.
- **Brings in:** coverage.

### ruff — 0.16.10 (optional)
- **Why:** Very fast linter + formatter (replaces flake8, isort, black). Catches unused imports, bugs, and keeps style consistent.
- **Brings in:** nothing (single compiled binary).

### matplotlib — 3.11.2
- **Why:** Static, publication-quality EDA figures saved as PNG to `reports/figures/` for the report and slides.
- **Brings in:** pillow, fonttools, kiwisolver, contourpy, cycler, pyparsing.

### ipykernel — 7.4.0 (optional)
- **Why:** Lets notebooks in `notebooks/` run with this project's virtual environment (in VS Code or Jupyter).
- **Without it:** You can keep using Colab for exploration; scripts in `src/` are the source of truth anyway.
- **Brings in:** ipython, jupyter-client, pyzmq, tornado, debugpy.

---

## Considered and deliberately NOT included

| Package | Reason not included |
|---|---|
| **seaborn** | Last release Jan 2024; matplotlib + plotly cover all EDA needs. Avoids a compatibility risk with pandas 3. Can be added if a specific plot needs it. |
| **torch** (PyTorch) | ~700 MB+. Only needed if we build the optional LSTM/GRU, and only after Baseline/RF/XGBoost are done. Decision deferred to after Phase 11. |
| **statsmodels** | Classical forecasting (SARIMA). Not needed for the core comparison. Optional later. |
| **MLflow** | Overkill for ~3–4 models. A CSV/JSON experiment log in `reports/experiments/` is enough. |
| **jupyterlab** | ~60 extra packages. Use Colab or VS Code notebooks (with `ipykernel`) instead. |
| **pydantic-settings** | Extra dependency for little gain over `python-dotenv` + a small `config.py`. |
| **requests** | Not listed directly — `httpx` covers our HTTP needs. (Streamlit still pulls it in transitively.) |
| **psycopg2** | Legacy driver; psycopg 3 is its modern successor. |

---

## Python version choice

| Version | Verdict |
|---|---|
| 3.14 (your system default) | Too new — some scientific wheels lag behind; risky for a deadline-bound project. |
| **3.12** | ✅ Chosen. Every pinned package ships wheels for it; `numpy 2.5` and `xgboost 3.4` require **≥ 3.12**. Already installed at `/opt/homebrew/bin/python3.12`. |
| 3.11 | ❌ Not supported by the pinned numpy/xgboost. |

---

## Security / licensing notes

- All listed packages use permissive licenses (BSD / MIT / Apache-2.0) — fine for an academic project.
  (psycopg is LGPL-3.0, which allows use as a library without affecting your code's license.)
- `joblib`/`pickle` files can execute code when loaded — only load models this project trained.
- Run `pip list --outdated` occasionally; upgrade deliberately, re-run tests, and update this file.

## Changelog

| Date | Change |
|---|---|
| 2026-10-06 | Phase 1: initial dependency plan; versions pinned and resolution-verified on Python 3.12. Nothing installed yet. |
| 2026-10-06 | All Python packages installed in `.venv`; Homebrew `libomp`, `postgresql@17`, `docker-compose` installed. No new Python packages beyond the plan. |

## Known warnings

- `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2` appears when
  running tests. It is harmless today. When upgrading FastAPI/Starlette in future, replace `httpx` with the
  package Starlette recommends and re-run the tests.
