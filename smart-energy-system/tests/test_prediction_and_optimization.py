"""Prediction engine, peak detection, cost estimation and recommendations."""

import pandas as pd
import pytest

from src.features.feature_engineering import FEATURE_COLUMNS
from src.models.baseline import PersistenceBaseline
from src.optimization.cost_estimation import (
    cost_breakdown,
    estimate_contracted_load_kw,
    estimate_cost,
    uppcl_energy_charge,
    uppcl_monthly_bill,
)
from src.optimization.peak_detection import (
    classify_severity,
    compute_thresholds,
    detect_peaks,
    peak_summary,
)
from src.optimization.recommendations import _format_windows, generate_recommendations
from src.prediction.predictor import EnergyPredictor, InsufficientHistoryError


def history(hours: int = 400, start: str = "2020-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=hours, freq="h")
    kwh = [0.2 + (1.0 if t.hour in (19, 20) else 0.0) for t in ts]
    return pd.DataFrame({"timestamp": ts, "kwh": kwh})


@pytest.fixture
def predictor() -> EnergyPredictor:
    return EnergyPredictor(
        PersistenceBaseline(), {"model_name": "persistence", "version": "test", "feature_columns": FEATURE_COLUMNS}
    )


# ---------------------------------------------------------------- predictor
def test_next_hour_prediction(predictor):
    h = history()
    result = predictor.predict_next_hour(h)
    assert result.timestamp == h["timestamp"].max() + pd.Timedelta(hours=1)
    assert result.predicted_kwh == pytest.approx(h["kwh"].iloc[-1])  # persistence = last value


def test_value_at_target_hour_is_never_used(predictor):
    h = history()
    target = h["timestamp"].iloc[-1]
    leaky = h.copy()
    leaky.loc[leaky.index[-1], "kwh"] = 999.0
    assert predictor.predict_next_hour(leaky, target).predicted_kwh == pytest.approx(h["kwh"].iloc[-2])


def test_insufficient_history_raises(predictor):
    h = history()
    with pytest.raises(InsufficientHistoryError):
        predictor.predict_next_hour(h.iloc[:-1], h["timestamp"].max() + pd.Timedelta(hours=1))  # last hour missing
    with pytest.raises(InsufficientHistoryError):
        predictor.predict_next_hour(history(hours=10))


def test_recursive_forecast(predictor):
    forecast = predictor.forecast(history(), hours=5)
    assert [p.step for p in forecast] == [1, 2, 3, 4, 5]
    assert all(p.predicted_kwh == pytest.approx(forecast[0].predicted_kwh) for p in forecast)
    with pytest.raises(ValueError):
        predictor.forecast(history(), hours=25)


# ---------------------------------------------------------------- peaks
def test_peak_thresholds_are_per_meter():
    df = pd.DataFrame({"meter": ["A"] * 10 + ["B"] * 10, "kwh": list(range(10)) + [x * 10 for x in range(10)]})
    thresholds = compute_thresholds(df, 90)
    assert thresholds["B"] == pytest.approx(10 * thresholds["A"])


def test_detect_peaks_and_severity():
    df = pd.DataFrame(
        {"meter": ["A"] * 4, "timestamp": pd.date_range("2020-01-01", periods=4, freq="h"), "kwh": [1.0, 1.1, 1.4, 2.0]}
    )
    out = detect_peaks(df, pd.Series({"A": 1.0}))
    assert out["is_peak"].tolist() == [False, True, True, True]
    assert pd.isna(out["severity"].iloc[0])
    assert out["severity"].iloc[1:].tolist() == ["moderate", "high", "critical"]
    assert classify_severity(0.5, 1.0) is None
    summary = peak_summary(out)
    assert summary["peak_hours"] == 3 and summary["peak_frequency_pct"] == pytest.approx(75.0)


def test_zero_threshold_meter_has_no_peaks():
    # Real data: a mostly idle meter has a 90th percentile of 0 kWh. Every hour with any use
    # would be "above threshold" but have no severity, which the database rejects.
    df = pd.DataFrame(
        {"meter": ["A"] * 3, "timestamp": pd.date_range("2020-01-01", periods=3, freq="h"), "kwh": [0.0, 0.0, 0.06]}
    )
    out = detect_peaks(df, pd.Series({"A": 0.0}))
    assert not out["is_peak"].any()
    assert out["severity"].isna().all()


def test_invalid_percentile_rejected():
    with pytest.raises(ValueError):
        compute_thresholds(pd.DataFrame({"meter": ["A"], "kwh": [1.0]}), 100)


# ---------------------------------------------------------------- cost
def test_estimate_cost():
    assert estimate_cost(10, 7.5) == pytest.approx(75.0)
    with pytest.raises(ValueError):
        estimate_cost(10, -1)
    with pytest.raises(ValueError):
        estimate_cost(-1, 7)


def test_cost_breakdown_daily():
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=48, freq="h"), "kwh": [0.5] * 48})
    out = cost_breakdown(df, tariff_per_kwh=8.0, freq="D")
    assert out["kwh"].tolist() == [12.0, 12.0]
    assert out["estimated_cost"].tolist() == [96.0, 96.0]


def test_uppcl_slab_energy_charge_is_telescopic():
    # Each slab's rate applies only to the units inside that slab (worked by hand).
    assert uppcl_energy_charge(0) == 0.0
    assert uppcl_energy_charge(100) == pytest.approx(550.0)  # 100 × 5.50
    assert uppcl_energy_charge(200) == pytest.approx(1125.0)  # 150 × 5.50 + 50 × 6.00
    assert uppcl_energy_charge(600) == pytest.approx(3725.0)  # 825 + 900 + 200 × 6.50 + 100 × 7.00
    with pytest.raises(ValueError):
        uppcl_energy_charge(-1)


def test_uppcl_monthly_bill_adds_fixed_charge():
    bill = uppcl_monthly_bill(200, contracted_load_kw=2)
    assert bill == {"fixed_charge": 220.0, "energy_charge": 1125.0, "total": 1345.0}


def test_contracted_load_estimate_rounds_up_with_1kw_minimum():
    assert estimate_contracted_load_kw(0.0) == 1
    assert estimate_contracted_load_kw(0.3) == 1
    assert estimate_contracted_load_kw(2.4) == 3


# ---------------------------------------------------------------- recommendations
def test_format_windows():
    assert _format_windows([20, 18, 19, 23]) == ["18:00–21:00", "23:00–00:00"]


def test_predicted_peak_recommendation():
    h = history()
    ts = h["timestamp"].max() + pd.Timedelta(hours=1)
    recs = generate_recommendations(h, peak_threshold=1.0, predictions=[(ts, 2.0)])
    types = {r.recommendation_type for r in recs}
    assert "predicted_peak" in types
    peak_rec = next(r for r in recs if r.recommendation_type == "predicted_peak")
    assert peak_rec.potential_saving_kwh == pytest.approx(1.0)
    assert peak_rec.severity == "critical"


def test_recurring_window_is_detected():
    recs = generate_recommendations(history(), peak_threshold=1.0)
    window = next(r for r in recs if r.recommendation_type == "recurring_peak_window")
    assert "19:00–21:00" in window.message


def test_normal_consumption_gives_reassurance():
    recs = generate_recommendations(history(), peak_threshold=5.0)  # nothing near the threshold
    assert [r.recommendation_type for r in recs] == ["normal"]


def test_always_on_overnight_load_is_flagged():
    flat = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=72, freq="h"), "kwh": [0.3] * 72})
    recs = generate_recommendations(flat, peak_threshold=5.0)
    assert "high_overnight_baseload" in {r.recommendation_type for r in recs}
