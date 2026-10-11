"""Shared FastAPI dependencies (database session, loaded model, meter lookup)."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from database import crud
from database.database import get_session
from database.models import Meter
from src.prediction.predictor import EnergyPredictor

DbSession = Annotated[Session, Depends(get_session)]


def get_predictor(request: Request) -> EnergyPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained model loaded. Run `python -m scripts.train` and restart the API.",
        )
    return predictor


def meter_or_404(session: Session, meter_code: str) -> Meter:
    meter = crud.get_meter_by_code(session, meter_code)
    if meter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown meter '{meter_code}'")
    return meter
