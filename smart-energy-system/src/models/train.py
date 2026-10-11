"""Phases 8–11 — Reproducible training pipeline.

Steps: load features -> chronological split -> train baselines, Random Forest, XGBoost ->
choose hyperparameters on VALIDATION -> pick the best model on VALIDATION RMSE ->
report all models on the untouched TEST set -> save best model + metadata + experiment log.

Why choose on validation and report on test? If we picked the model with the best *test*
score, the test set would become part of model selection and its score would be optimistic.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    DATA_QUALITY_REPORT,
    EXPERIMENT_LOG_PATH,
    EXPERIMENTS_DIR,
    FEATURES_PARQUET,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_PREDICTIONS_PATH,
)
from src.data.loader import load_parquet
from src.features.feature_engineering import FEATURE_COLUMNS
from src.models.baseline import PersistenceBaseline, SeasonalNaiveBaseline
from src.models.evaluate import per_meter_metrics, regression_metrics
from src.models.random_forest import RANDOM_FOREST_CANDIDATES, build_random_forest
from src.models.split import chronological_split
from src.models.xgboost_model import XGBOOST_CANDIDATES, build_xgboost

logger = logging.getLogger(__name__)

SELECTION_METRIC = "rmse"  # lower is better; penalises large (peak-hour) errors


def _fit_and_score(name, model, params, X_train, y_train, X_val, y_val, fit_kwargs=None):  # noqa: N803
    started = time.perf_counter()
    model.fit(X_train, y_train, **(fit_kwargs or {}))
    seconds = time.perf_counter() - started
    val_metrics = regression_metrics(y_val, model.predict(X_val))
    if hasattr(model, "best_iteration") and getattr(model, "best_iteration", None) is not None:
        params = {**params, "best_iteration": int(model.best_iteration)}
    logger.info(
        "%-24s val MAE=%.4f RMSE=%.4f R2=%.4f (%.1fs) params=%s",
        name,
        val_metrics["mae"],
        val_metrics["rmse"],
        val_metrics["r2"],
        seconds,
        params,
    )
    return {"name": name, "model": model, "params": params, "val": val_metrics, "seconds": seconds}


def _best_of(candidates: list[dict]) -> dict:
    return min(candidates, key=lambda c: c["val"][SELECTION_METRIC])


def _read_data_source() -> dict:
    if DATA_QUALITY_REPORT.exists():
        report = json.loads(DATA_QUALITY_REPORT.read_text())
        return {"source_file": report.get("source_file"), "is_synthetic": report.get("is_synthetic")}
    return {"source_file": None, "is_synthetic": None}


def run_training(features_path: Path = FEATURES_PARQUET) -> dict:
    """Train, evaluate and save. Returns a summary dict (also written to disk)."""
    df = load_parquet(features_path).sort_values(["timestamp", "meter"]).reset_index(drop=True)
    train, val, test, cutoffs = chronological_split(df)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]  # noqa: N806
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]  # noqa: N806
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]  # noqa: N806

    results = [
        _fit_and_score("baseline_persistence", PersistenceBaseline(), {}, X_train, y_train, X_val, y_val),
        _fit_and_score("baseline_seasonal_naive", SeasonalNaiveBaseline(), {}, X_train, y_train, X_val, y_val),
    ]

    rf_runs = [
        _fit_and_score("random_forest", build_random_forest(**p), p, X_train, y_train, X_val, y_val)
        for p in RANDOM_FOREST_CANDIDATES
    ]
    results.append(_best_of(rf_runs))

    xgb_runs = [
        _fit_and_score(
            "xgboost",
            build_xgboost(**p),
            p,
            X_train,
            y_train,
            X_val,
            y_val,
            fit_kwargs={"eval_set": [(X_val, y_val)], "verbose": False},
        )
        for p in XGBOOST_CANDIDATES
    ]
    results.append(_best_of(xgb_runs))

    # ---- Final, one-time evaluation on the untouched test set ----
    test_predictions = test[["meter", "timestamp", TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "actual_kwh"})
    for result in results:
        preds = result["model"].predict(X_test)
        result["test"] = regression_metrics(y_test, preds)
        test_predictions[f"pred_{result['name']}"] = preds

    best = _best_of(results)
    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    trained_at = datetime.now().isoformat(timespec="seconds")
    data_source = _read_data_source()

    comparison = pd.DataFrame(
        [
            {
                "model": r["name"],
                "test_mae": r["test"]["mae"],
                "test_rmse": r["test"]["rmse"],
                "test_r2": r["test"]["r2"],
                "test_nmae_pct": r["test"]["nmae_pct"],
                "val_rmse": r["val"]["rmse"],
                "train_seconds": round(r["seconds"], 2),
                "selected": r is best,
            }
            for r in results
        ]
    )
    logger.info("Model comparison (TEST set):\n%s", comparison.to_string(index=False))

    # ---- Save artefacts ----
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best["model"], BEST_MODEL_PATH)
    test_predictions.to_parquet(TEST_PREDICTIONS_PATH, index=False)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    per_meter_metrics(test_predictions, "actual_kwh", f"pred_{best['name']}").to_csv(
        EXPERIMENTS_DIR / "per_meter_test_metrics.csv", index=False
    )
    if hasattr(best["model"], "feature_importances_"):
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": best["model"].feature_importances_}).sort_values(
            "importance", ascending=False
        ).to_csv(EXPERIMENTS_DIR / "feature_importance.csv", index=False)

    metadata = {
        "model_name": best["name"],
        "version": version,
        "trained_at": trained_at,
        "task": "next-hour consumption forecast (kWh) per meter",
        "target": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "hyperparameters": best["params"],
        "selection_metric": f"validation {SELECTION_METRIC}",
        "validation_metrics": best["val"],
        "test_metrics": best["test"],
        "split": {**cutoffs.as_dict(), "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test)},
        "all_models": [
            {"model_name": r["name"], "hyperparameters": r["params"], "validation": r["val"], "test": r["test"]}
            for r in results
        ],
        "random_state": RANDOM_STATE,
        "model_path": str(BEST_MODEL_PATH.relative_to(BEST_MODEL_PATH.parents[2])),
        **data_source,
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))

    log_rows = pd.DataFrame(
        [
            {
                "run_version": version,
                "trained_at": trained_at,
                "model": r["name"],
                "features": "|".join(FEATURE_COLUMNS),
                "hyperparameters": json.dumps(r["params"], default=str),
                # Candidates that lost on validation are never scored on the test set.
                "finalist": "test" in r,
                **{f"val_{k}": v for k, v in r["val"].items()},
                **{f"test_{k}": v for k, v in r.get("test", {}).items()},
                "source_file": data_source["source_file"],
                "is_synthetic": data_source["is_synthetic"],
            }
            for r in results[:2] + rf_runs + xgb_runs
        ]
    )
    log_rows.to_csv(EXPERIMENT_LOG_PATH, mode="a", header=not EXPERIMENT_LOG_PATH.exists(), index=False)

    logger.info("Best model: %s (%s) saved to %s", best["name"], version, BEST_MODEL_PATH)
    return metadata
