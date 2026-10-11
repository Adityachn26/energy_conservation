"""Phase 9 — Random Forest regressor.

A Random Forest averages many decision trees, each trained on a random sample of rows and
features. Averaging reduces variance, so it is robust and needs little tuning.

The candidate settings are deliberately limited (depth, leaf size, row subsampling) so that a
forest trained on ~230k rows stays small enough to save, load and serve quickly.
"""

from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE

# Small candidate grid — each is scored on the VALIDATION set and the best one is kept.
RANDOM_FOREST_CANDIDATES = [
    {"n_estimators": 150, "max_depth": 14, "min_samples_leaf": 20, "max_features": 0.6},
    {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 10, "max_features": 0.6},
]


def build_random_forest(**params) -> RandomForestRegressor:
    return RandomForestRegressor(
        **params,
        max_samples=0.5,  # each tree sees 50% of rows: faster, smaller, still accurate
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
