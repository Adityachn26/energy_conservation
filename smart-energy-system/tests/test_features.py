"""Phase 6: feature engineering — correctness and data-leakage guards."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    FEATURE_COLUMNS,
    add_time_features,
    check_regular_hourly_grid,
    compute_features,
    select_model_rows,
)
from tests.conftest import hourly_frame


def test_time_features():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-04 18:00", "2020-01-06 07:00"])})  # Sat, Mon
    out = add_time_features(df)
    assert out["hour"].tolist() == [18, 7]
    assert out["day_of_week"].tolist() == [5, 0]
    assert out["is_weekend"].tolist() == [1, 0]
    assert out["month"].tolist() == [1, 1]
    assert out["day_of_month"].tolist() == [4, 6]


def test_lags_are_hours_not_rows_across_gaps():
    # Hour index 3 is missing. lag_1h at hour 4 must be NaN, NOT the value from hour 2.
    df = compute_features(hourly_frame({"A": [1.0, 2.0, 3.0, None, 5.0, 6.0]}))
    assert np.isnan(df.loc[4, "lag_1h"])
    assert df.loc[4, "lag_2h"] == 3.0
    assert df.loc[5, "lag_1h"] == 5.0


def test_rolling_mean_excludes_current_hour():
    df = compute_features(hourly_frame({"A": [1.0, 2.0, 3.0, 100.0]}))
    # Window for hour 3 = hours 0..2 -> (1+2+3)/3 = 2; the 100 at hour 3 must not be included.
    assert df.loc[3, "rolling_mean_3h"] == pytest.approx(2.0)


def test_features_do_not_change_when_current_value_changes():
    """Leakage guard: the target at time T must not influence any feature at time T."""
    values = list(np.random.default_rng(0).random(200))
    base = compute_features(hourly_frame({"A": values}))
    values[150] = 999.0
    changed = compute_features(hourly_frame({"A": values}))
    pd.testing.assert_series_equal(base.loc[150, FEATURE_COLUMNS], changed.loc[150, FEATURE_COLUMNS])


def test_lags_never_cross_meters():
    df = compute_features(hourly_frame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]}))
    first_b = df.loc[df["meter"] == "B"].iloc[0]
    assert np.isnan(first_b["lag_1h"]), "meter B's first lag must not come from meter A"


def test_filtered_data_is_rejected():
    grid = hourly_frame({"A": [1.0, 2.0, None, 4.0]})
    with pytest.raises(ValueError, match="regular hourly grid"):
        check_regular_hourly_grid(grid.dropna(subset=["kwh"]))


def test_select_model_rows_accounts_for_every_row():
    values = [float(i % 24) + 1 for i in range(400)]
    values[300] = None
    featured = compute_features(hourly_frame({"A": values}))
    rows, log = select_model_rows(featured)
    assert (
        log["grid_rows"] == log["model_rows"] + log["dropped_incomplete_target_hour"] + log["dropped_missing_history"]
    )
    assert log["dropped_incomplete_target_hour"] == 1
    assert rows[FEATURE_COLUMNS].notna().all().all()
