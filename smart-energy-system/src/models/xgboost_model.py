"""Phase 10 — XGBoost (gradient-boosted trees) regressor.

Boosting builds trees one after another; each new tree corrects the errors of the previous ones.
It usually beats Random Forest on tabular data, but can overfit — so we use **early stopping**:
training stops when the error on the VALIDATION set has not improved for 50 rounds.
The test set is never used for any training decision.
"""

from xgboost import XGBRegressor

from src.config import RANDOM_STATE

XGBOOST_CANDIDATES = [
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 10},
]


def build_xgboost(**params) -> XGBRegressor:
    return XGBRegressor(
        **params,
        n_estimators=2000,  # upper bound; early stopping picks the real number
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="rmse",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
