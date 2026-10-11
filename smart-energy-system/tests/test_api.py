"""API endpoint tests against an in-memory SQLite database (no PostgreSQL needed)."""

from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database import crud
from database.database import Base, get_session
from database.models import EnergyConsumption, Meter, ModelMetadata, Prediction
from src.features.feature_engineering import FEATURE_COLUMNS
from src.models.baseline import PersistenceBaseline
from src.prediction.predictor import EnergyPredictor

HOURS = 400
START = pd.Timestamp("2020-01-01")


@pytest.fixture(scope="module")
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)  # noqa: N806

    with SessionLocal() as session:
        session.add_all([Meter(meter_code="M1"), Meter(meter_code="M2")])
        session.commit()
        timestamps = pd.date_range(START, periods=HOURS, freq="h")
        rows = []
        for meter_id, scale in ((1, 1.0), (2, 2.0)):
            for ts in timestamps:
                kwh = scale * (0.2 + (1.0 if ts.hour in (19, 20) else 0.0))
                rows.append(
                    {
                        "meter_id": meter_id,
                        "timestamp": ts.to_pydatetime(),
                        "consumption_kwh": kwh,
                        "avg_voltage": 230.0,
                        "avg_current": 1.0,
                        "frequency": 50.0,
                        "readings_count": 20,
                    }
                )
        crud.bulk_insert(session, EnergyConsumption, rows)
        model = ModelMetadata(
            model_name="persistence",
            version="test",
            mae=0.1,
            rmse=0.2,
            r2=0.9,
            nmae_pct=10.0,
            training_date=datetime(2020, 1, 1),
            is_active=True,
            features=FEATURE_COLUMNS,
        )
        session.add(model)
        session.flush()
        session.add(
            Prediction(
                meter_id=1,
                timestamp=timestamps[-1].to_pydatetime(),
                predicted_kwh=0.25,
                model_id=model.id,
                prediction_type="backtest",
                horizon_step=1,
            )
        )
        session.commit()
        crud.get_settings(session)  # defaults: tariff 7.0, P90
        crud.recompute_peak_events(session, 90)

    def override_session():
        with SessionLocal() as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    with TestClient(app) as test_client:
        app.state.predictor = EnergyPredictor(
            PersistenceBaseline(),
            {
                "model_name": "persistence",
                "version": "test",
                "feature_columns": FEATURE_COLUMNS,
                "trained_at": "2020-01-01T00:00:00",
                "is_synthetic": True,
            },
        )
        yield test_client
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] == "ok" and body["model_loaded"]


def test_meters(client):
    meters = client.get("/meters").json()
    assert [m["meter_code"] for m in meters] == ["M1", "M2"]
    assert meters[0]["hourly_records"] == HOURS


def test_consumption_granularity(client):
    hourly = client.get("/consumption", params={"meter_code": "M1"}).json()
    assert len(hourly["points"]) == HOURS
    daily = client.get("/consumption", params={"meter_code": "M1", "granularity": "daily"}).json()
    assert daily["points"][0]["consumption_kwh"] == pytest.approx(24 * 0.2 + 2 * 1.0)
    assert daily["points"][0]["meter_hours"] == 24


def test_overview_cost_is_labelled_estimate(client):
    body = client.get("/overview", params={"meter_code": "M1"}).json()
    assert body["estimated_cost"] == pytest.approx(body["total_kwh"] * 7.0, rel=1e-6)
    assert body["cost_label"] == "Estimated Cost"
    assert body["peak_event_count"] > 0


def test_unknown_meter_is_404(client):
    assert client.get("/consumption", params={"meter_code": "NOPE"}).status_code == 404
    assert client.post("/predict", json={"meter_code": "NOPE"}).status_code == 404


def test_predict_next_hour_and_store(client):
    response = client.post("/predict", json={"meter_code": "M1", "horizon_hours": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["forecast"]) == 3
    assert body["forecast"][0]["timestamp"] == (START + pd.Timedelta(hours=HOURS)).isoformat()
    assert any("recursive" in note for note in body["notes"])
    assert any("SYNTHETIC" in note for note in body["notes"])
    live = client.get("/predictions", params={"meter_code": "M1", "prediction_type": "live"}).json()
    assert len(live) == 3


def test_predict_validation(client):
    assert client.post("/predict", json={"meter_code": "M1", "horizon_hours": 0}).status_code == 422
    assert client.post("/predict", json={"meter_code": "M1", "horizon_hours": 25}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_backtest_predictions_join_actuals(client):
    rows = client.get("/predictions", params={"meter_code": "M1", "prediction_type": "backtest"}).json()
    assert len(rows) == 1
    assert rows[0]["error_kwh"] == pytest.approx(rows[0]["predicted_kwh"] - rows[0]["actual_kwh"])


def test_peaks_and_summary(client):
    peaks = client.get("/peaks", params={"meter_code": "M2", "order_by": "kwh", "limit": 5}).json()
    assert len(peaks) == 5 and all(p["actual_kwh"] > p["threshold"] for p in peaks)
    summary = client.get("/peaks/summary", params={"meter_code": "M1"}).json()
    assert set(summary["most_common_peak_hours"]) <= {19, 20}


def test_recommendations_endpoint(client):
    recs = client.get("/recommendations", params={"meter_code": "M1"}).json()
    assert recs, "the earlier /predict call stored recommendations"
    keys = [(r["recommendation_type"], r["message"]) for r in recs]
    assert len(keys) == len(set(keys)), "duplicates are collapsed"


def test_model_performance(client):
    body = client.get("/model-performance").json()
    assert body["models"][0]["model_name"] == "persistence"


def test_settings_update_and_validation(client):
    assert client.get("/settings").json()["tariff_per_kwh"] == 7.0
    updated = client.put("/settings", json={"tariff_per_kwh": 9.5, "peak_percentile": 95}).json()
    assert updated["tariff_per_kwh"] == 9.5 and updated["peak_percentile"] == 95
    fewer_peaks = client.get("/peaks", params={"limit": 50000}).json()
    assert all(p["percentile"] == 95 for p in fewer_peaks)

    for bad in ({"tariff_per_kwh": -1}, {"peak_percentile": 100}, {"timezone": "Mars/Base"}, {"currency": "RUPEES"}):
        assert client.put("/settings", json=bad).status_code == 422, bad
    client.put("/settings", json={"tariff_per_kwh": 7.0, "peak_percentile": 90})
