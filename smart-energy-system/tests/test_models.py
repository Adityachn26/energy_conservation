"""Phases 7–11: split, baselines, metrics and the training pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

import src.models.train as train_module
from src.data.aggregation import aggregate_to_hourly
from src.data.loader import RAW_COLUMN_MAP
from src.features.feature_engineering import compute_features, select_model_rows
from src.models.baseline import PersistenceBaseline, SeasonalNaiveBaseline
from src.models.evaluate import regression_metrics
from src.models.split import chronological_split


@pytest.fixture(scope="module")
def features(synthetic_raw_original) -> pd.DataFrame:
    raw = synthetic_raw_original.rename(columns=RAW_COLUMN_MAP)
    raw["meter"] = raw["meter"].astype("string")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    rows, _ = select_model_rows(compute_features(aggregate_to_hourly(raw)))
    return rows


def test_chronological_split_has_no_time_overlap(features):
    train, val, test, cutoffs = chronological_split(features)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(val) + len(test) == len(features)
    assert 0.65 < len(train) / len(features) < 0.75
    assert cutoffs.test_start == test["timestamp"].min()


def test_split_rejects_bad_fractions(features):
    with pytest.raises(ValueError):
        chronological_split(features, train_fraction=0.9, validation_fraction=0.2)


def test_baselines_return_lag_columns():
    X = pd.DataFrame({"lag_1h": [1.0, 2.0], "lag_24h": [5.0, 6.0]})  # noqa: N806
    assert PersistenceBaseline().fit(X).predict(X).tolist() == [1.0, 2.0]
    assert SeasonalNaiveBaseline().predict(X).tolist() == [5.0, 6.0]


def test_regression_metrics_known_values():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert metrics["nmae_pct"] == pytest.approx(100 * (2 / 3) / 2)
    assert regression_metrics([1.0, 2.0], [1.0, 2.0])["r2"] == pytest.approx(1.0)


def test_training_pipeline_end_to_end(tmp_path, monkeypatch, features):
    """Small but real run: trains all models, writes model, metadata and experiment log."""
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    for name, value in {
        "ARTIFACTS_DIR": tmp_path / "models",
        "EXPERIMENTS_DIR": tmp_path / "exp",
        "BEST_MODEL_PATH": tmp_path / "models" / "best_model.joblib",
        "MODEL_METADATA_PATH": tmp_path / "models" / "model_metadata.json",
        "TEST_PREDICTIONS_PATH": tmp_path / "models" / "test_predictions.parquet",
        "EXPERIMENT_LOG_PATH": tmp_path / "exp" / "experiments.csv",
        "MODEL_COMPARISON_PATH": tmp_path / "exp" / "comparison.csv",
        "DATA_QUALITY_REPORT": tmp_path / "missing.json",
        "RANDOM_FOREST_CANDIDATES": [{"n_estimators": 20, "max_depth": 8, "min_samples_leaf": 10, "max_features": 0.6}],
        "XGBOOST_CANDIDATES": [{"max_depth": 4, "learning_rate": 0.1, "min_child_weight": 5}],
    }.items():
        monkeypatch.setattr(train_module, name, value)

    metadata = train_module.run_training(features_path)

    assert {m["model_name"] for m in metadata["all_models"]} == {
        "baseline_persistence",
        "baseline_seasonal_naive",
        "random_forest",
        "xgboost",
    }
    assert (tmp_path / "models" / "best_model.joblib").exists()
    saved = json.loads((tmp_path / "models" / "model_metadata.json").read_text())
    assert saved["feature_columns"] == metadata["feature_columns"]
    best_val = min(m["validation"]["rmse"] for m in metadata["all_models"])
    assert metadata["validation_metrics"]["rmse"] == best_val, "best model is chosen on VALIDATION"
    assert len(pd.read_csv(tmp_path / "exp" / "experiments.csv")) == 4
