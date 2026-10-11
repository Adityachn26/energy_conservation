"""Phase 11 — Evaluation metrics.

* MAE  — average absolute error in kWh. Easy to explain; treats all errors equally.
* RMSE — square-root of mean squared error, in kWh. Punishes large errors more, which matters
         for peak hours. Sensitive to outliers.
* R²   — share of the variance explained (1 = perfect, 0 = no better than predicting the mean,
         can be negative). Unit-free, but can look good even when absolute errors are large.
* NMAE — MAE divided by mean consumption (as %). A scale-free alternative to MAPE.

MAPE is deliberately NOT used: many hours have consumption at or near 0 kWh, and MAPE divides by
the actual value, so it explodes (or is undefined) exactly on those hours.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mean_actual = float(np.mean(y_true))
    return {
        "mae": float(mae),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nmae_pct": float(100 * mae / mean_actual) if mean_actual > 0 else float("nan"),
    }


def per_meter_metrics(df: pd.DataFrame, actual_col: str, pred_col: str) -> pd.DataFrame:
    """MAE/RMSE/R² per meter — shows whether the model works for every meter, not just on average."""
    rows = []
    for meter, group in df.groupby("meter", observed=True):
        rows.append({"meter": meter, "n": len(group), **regression_metrics(group[actual_col], group[pred_col])})
    return pd.DataFrame(rows).sort_values("mae")
