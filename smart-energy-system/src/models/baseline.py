"""Phase 8 — Baseline models.

A baseline is the simplest reasonable forecast. ML models are only worth their complexity if
they beat it on the same test period.

* PersistenceBaseline:    prediction(T) = consumption(T - 1h)    ("tomorrow is like today")
* SeasonalNaiveBaseline:  prediction(T) = consumption(T - 24h)   (same hour yesterday)

They follow the scikit-learn ``fit``/``predict`` interface so the training/evaluation code can
treat every model the same way.
"""

import numpy as np
import pandas as pd


class _LagBaseline:
    lag_column: str = ""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_LagBaseline":  # noqa: N803
        return self  # nothing to learn

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return X[self.lag_column].to_numpy(dtype=float)

    def get_params(self, deep: bool = True) -> dict:
        return {"lag_column": self.lag_column}


class PersistenceBaseline(_LagBaseline):
    lag_column = "lag_1h"


class SeasonalNaiveBaseline(_LagBaseline):
    lag_column = "lag_24h"
