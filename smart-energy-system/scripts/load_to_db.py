"""Load processed data and model results into PostgreSQL (Phase 15).

Prerequisites:
    1. PostgreSQL running and .env configured     (docker compose up -d db)
    2. Schema created:                            alembic upgrade head
    3. Pipeline + training done:                  python -m scripts.run_pipeline / scripts.train

Usage:
    python -m scripts.load_to_db

The load is a full refresh: existing data rows are replaced (system settings are kept).
"""

import json
import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import delete, inspect
from sqlalchemy.orm import Session

from database import crud
from database.database import get_engine, get_session_factory
from database.models import (
    EnergyConsumption,
    Meter,
    ModelMetadata,
    PeakEvent,
    Prediction,
    Recommendation,
)
from src.config import (
    HOURLY_CLEAN_PARQUET,
    MODEL_METADATA_PATH,
    TEST_PREDICTIONS_PATH,
    setup_logging,
)
from src.data.loader import load_parquet
from src.optimization.recommendations import generate_recommendations

logger = logging.getLogger("load_to_db")


def reset_data(session: Session) -> None:
    for table in (Recommendation, PeakEvent, Prediction, EnergyConsumption, ModelMetadata, Meter):
        session.execute(delete(table))
    session.commit()
    logger.info("Existing data rows deleted (settings kept)")


def load_meters_and_consumption(session: Session, hourly: pd.DataFrame) -> dict[str, int]:
    meter_codes = sorted(hourly["meter"].unique())
    crud.bulk_insert(session, Meter, [{"meter_code": code} for code in meter_codes])
    session.commit()
    meter_ids = {m.meter_code: m.id for m in session.query(Meter).all()}

    rows = [
        {
            "meter_id": meter_ids[r.meter],
            "timestamp": r.timestamp.to_pydatetime(),
            "consumption_kwh": float(r.kwh),
            "avg_voltage": float(r.voltage),
            "avg_current": float(r.current),
            "frequency": float(r.frequency),
            "readings_count": int(r.readings_per_hour),
        }
        for r in hourly.itertuples()
    ]
    crud.bulk_insert(session, EnergyConsumption, rows)
    session.commit()
    logger.info("Inserted %s meters and %s hourly rows", len(meter_ids), f"{len(rows):,}")
    return meter_ids


def load_models_and_backtest(session: Session, meter_ids: dict[str, int]) -> None:
    metadata = json.loads(MODEL_METADATA_PATH.read_text())
    trained_at = datetime.fromisoformat(metadata["trained_at"])
    active_id = None
    for entry in metadata["all_models"]:
        is_best = entry["model_name"] == metadata["model_name"]
        record = ModelMetadata(
            model_name=entry["model_name"],
            version=metadata["version"],
            mae=entry["test"]["mae"],
            rmse=entry["test"]["rmse"],
            r2=entry["test"]["r2"],
            nmae_pct=entry["test"]["nmae_pct"],
            training_date=trained_at,
            model_path=metadata["model_path"] if is_best else None,
            features=metadata["feature_columns"],
            hyperparameters=entry["hyperparameters"],
            is_active=is_best,
            data_source=metadata.get("source_file"),
            is_synthetic=bool(metadata.get("is_synthetic")),
        )
        session.add(record)
        session.flush()
        if is_best:
            active_id = record.id
    session.commit()

    preds = pd.read_parquet(TEST_PREDICTIONS_PATH)
    column = f"pred_{metadata['model_name']}"
    rows = [
        {
            "meter_id": meter_ids[r.meter],
            "timestamp": r.timestamp.to_pydatetime(),
            "predicted_kwh": max(float(getattr(r, column)), 0.0),
            "model_id": active_id,
            "prediction_type": "backtest",
            "horizon_step": 1,
        }
        for r in preds.itertuples()
        if r.meter in meter_ids
    ]
    crud.bulk_insert(session, Prediction, rows)
    session.commit()
    logger.info("Inserted %s model records and %s backtest predictions", len(metadata["all_models"]), f"{len(rows):,}")


def load_recommendations(session: Session, meter_ids: dict[str, int], percentile: float) -> None:
    all_history = crud.consumption_frame(session)
    rows = []
    for meter_id in meter_ids.values():
        history = all_history.loc[all_history["meter_id"] == meter_id]
        if history.empty:
            continue
        threshold = float(history["kwh"].quantile(percentile / 100))
        for rec in generate_recommendations(history, threshold):
            rows.append(
                {
                    "meter_id": meter_id,
                    "timestamp": rec.timestamp.to_pydatetime() if rec.timestamp is not None else None,
                    "recommendation_type": rec.recommendation_type,
                    "severity": rec.severity,
                    "message": rec.message,
                    "potential_saving_kwh": rec.potential_saving_kwh,
                }
            )
    crud.bulk_insert(session, Recommendation, rows)
    session.commit()
    logger.info("Inserted %s historical recommendations", len(rows))


def main() -> None:
    setup_logging()
    if "energy_consumption" not in inspect(get_engine()).get_table_names():
        raise SystemExit("Tables not found. Run `alembic upgrade head` first.")

    hourly = load_parquet(HOURLY_CLEAN_PARQUET)
    with get_session_factory()() as session:
        reset_data(session)
        settings = crud.get_settings(session)
        meter_ids = load_meters_and_consumption(session, hourly)
        load_models_and_backtest(session, meter_ids)
        crud.recompute_peak_events(session, float(settings.peak_percentile))
        load_recommendations(session, meter_ids, float(settings.peak_percentile))
    logger.info("Database load complete.")


if __name__ == "__main__":
    main()
