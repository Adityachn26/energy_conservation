from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, select

from api import services
from api.dependencies import DbSession, get_predictor, meter_or_404
from api.schemas.prediction import PredictionRecord, PredictRequest, PredictResponse
from database.models import EnergyConsumption, ModelMetadata, Prediction
from src.prediction.predictor import EnergyPredictor

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictResponse, summary="Forecast the next 1–24 hours for a meter")
def predict(
    body: PredictRequest,
    session: DbSession,
    predictor: Annotated[EnergyPredictor, Depends(get_predictor)],
):
    meter = meter_or_404(session, body.meter_code)
    return services.run_prediction(session, predictor, meter, body.horizon_hours, body.store)


@router.get("/predictions", response_model=list[PredictionRecord], summary="Stored predictions vs actuals")
def list_predictions(
    session: DbSession,
    meter_code: str,
    start: datetime | None = None,
    end: datetime | None = None,
    prediction_type: Literal["backtest", "live"] | None = None,
    limit: int = Query(5000, ge=1, le=50000),
):
    meter = meter_or_404(session, meter_code)
    stmt = (
        select(
            Prediction.timestamp,
            Prediction.predicted_kwh,
            EnergyConsumption.consumption_kwh.label("actual_kwh"),
            ModelMetadata.model_name,
            ModelMetadata.version.label("model_version"),
            Prediction.prediction_type,
            Prediction.horizon_step,
        )
        .join(ModelMetadata, ModelMetadata.id == Prediction.model_id)
        .outerjoin(
            EnergyConsumption,
            and_(
                EnergyConsumption.meter_id == Prediction.meter_id,
                EnergyConsumption.timestamp == Prediction.timestamp,
            ),
        )
        .where(Prediction.meter_id == meter.id)
        .order_by(Prediction.timestamp, Prediction.id)
        .limit(limit)
    )
    if start:
        stmt = stmt.where(Prediction.timestamp >= start)
    if end:
        stmt = stmt.where(Prediction.timestamp <= end)
    if prediction_type:
        stmt = stmt.where(Prediction.prediction_type == prediction_type)

    records = []
    for row in session.execute(stmt):
        data = dict(row._mapping)
        actual = data["actual_kwh"]
        data["error_kwh"] = None if actual is None else data["predicted_kwh"] - actual
        records.append(data)
    return records
