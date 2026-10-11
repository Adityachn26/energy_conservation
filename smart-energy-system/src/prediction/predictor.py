"""Prediction engine — loads the saved model ONCE and produces forecasts from meter history.

Features are built with exactly the same ``compute_features`` function used in training, so
training and serving cannot drift apart ("training/serving skew").

* ``predict_next_hour``: the main, evaluated use-case (what the model was trained and tested on).
* ``forecast``: next 1–24 hours using a *recursive* strategy — each prediction is fed back in as
  the "previous hour" for the next step. Errors therefore accumulate with the horizon; only the
  next-hour accuracy is measured by the test-set metrics. The API labels this clearly.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, MAX_FORECAST_HORIZON_HOURS, MODEL_METADATA_PATH
from src.features.feature_engineering import MAX_LOOKBACK_HOURS, compute_features

logger = logging.getLogger(__name__)


class InsufficientHistoryError(ValueError):
    """Not enough recent, gap-free history to compute the model's features."""


@dataclass
class HourPrediction:
    timestamp: pd.Timestamp
    predicted_kwh: float
    step: int  # 1 = next hour (evaluated); >1 = recursive multi-step


class EnergyPredictor:
    def __init__(self, model, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.feature_columns: list[str] = metadata["feature_columns"]

    @classmethod
    def load(cls, model_path: Path = BEST_MODEL_PATH, metadata_path: Path = MODEL_METADATA_PATH) -> "EnergyPredictor":
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Run: python -m scripts.train")
        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text())
        logger.info("Loaded model %s (%s) from %s", metadata["model_name"], metadata["version"], model_path)
        return cls(model, metadata)

    @property
    def model_name(self) -> str:
        return self.metadata["model_name"]

    @property
    def version(self) -> str:
        return self.metadata["version"]

    # ------------------------------------------------------------------
    def _build_grid(self, history: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
        """Regular hourly grid ending at ``target`` (whose kWh is unknown = NaN)."""
        start = target - pd.Timedelta(hours=MAX_LOOKBACK_HOURS + 1)
        hours = pd.date_range(start, target, freq="h")
        kwh = (
            history.loc[(history["timestamp"] >= start) & (history["timestamp"] < target)]
            .set_index("timestamp")["kwh"]
            .reindex(hours)
        )
        kwh.iloc[-1] = np.nan  # the value being predicted is never an input
        return pd.DataFrame({"meter": "m", "timestamp": hours, "kwh": kwh.to_numpy()})

    def _predict_at(self, history: pd.DataFrame, target: pd.Timestamp) -> float:
        features = compute_features(self._build_grid(history, target)).iloc[[-1]]
        missing = [c for c in self.feature_columns if pd.isna(features[c].iloc[0])]
        if missing:
            raise InsufficientHistoryError(
                f"Cannot predict {target}: missing features {missing}. The meter needs "
                f"recent history without gaps (up to {MAX_LOOKBACK_HOURS} hours back)."
            )
        prediction = float(self.model.predict(features[self.feature_columns])[0])
        return max(prediction, 0.0)  # consumption cannot be negative

    def predict_next_hour(self, history: pd.DataFrame, target_timestamp: pd.Timestamp | None = None) -> HourPrediction:
        """Predict one hour. ``history`` needs columns timestamp, kwh (complete hours only)."""
        if history.empty:
            raise InsufficientHistoryError("No history available for this meter.")
        target = (
            pd.Timestamp(target_timestamp).floor("h")
            if target_timestamp is not None
            else history["timestamp"].max() + pd.Timedelta(hours=1)
        )
        return HourPrediction(timestamp=target, predicted_kwh=self._predict_at(history, target), step=1)

    def forecast(
        self, history: pd.DataFrame, hours: int = 24, start: pd.Timestamp | None = None
    ) -> list[HourPrediction]:
        """Recursive multi-step forecast for ``hours`` consecutive hours (1..24)."""
        if not 1 <= hours <= MAX_FORECAST_HORIZON_HOURS:
            raise ValueError(f"hours must be between 1 and {MAX_FORECAST_HORIZON_HOURS}")
        first = self.predict_next_hour(history, start)
        results = [first]
        working = pd.concat(
            [
                history[["timestamp", "kwh"]],
                pd.DataFrame({"timestamp": [first.timestamp], "kwh": [first.predicted_kwh]}),
            ],
            ignore_index=True,
        )
        for step in range(2, hours + 1):
            target = first.timestamp + pd.Timedelta(hours=step - 1)
            value = self._predict_at(working, target)
            results.append(HourPrediction(timestamp=target, predicted_kwh=value, step=step))
            working = pd.concat([working, pd.DataFrame({"timestamp": [target], "kwh": [value]})], ignore_index=True)
        return results
