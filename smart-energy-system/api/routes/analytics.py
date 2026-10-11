import csv
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Query, Request
from sqlalchemy import select

from api import services
from api.dependencies import DbSession, meter_or_404
from api.schemas.analytics import (
    ModelOut,
    ModelPerformanceResponse,
    PeakEventOut,
    PeakSummary,
    SettingsOut,
    SettingsUpdate,
)
from api.schemas.prediction import RecommendationOut
from database import crud
from database.models import Meter, ModelMetadata, PeakEvent, Recommendation
from src.config import EXPERIMENTS_DIR

router = APIRouter()


# ---------------------------------------------------------------- peaks
@router.get("/peaks", response_model=list[PeakEventOut], tags=["peaks"], summary="Detected peak hours")
def list_peaks(
    session: DbSession,
    meter_code: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    order_by: Literal["timestamp", "kwh"] = "timestamp",
    limit: int = Query(1000, ge=1, le=50000),
):
    stmt = select(
        Meter.meter_code,
        PeakEvent.timestamp,
        PeakEvent.actual_kwh,
        PeakEvent.threshold,
        PeakEvent.percentile,
        PeakEvent.severity,
    ).join(Meter, Meter.id == PeakEvent.meter_id)
    if meter_code:
        stmt = stmt.where(PeakEvent.meter_id == meter_or_404(session, meter_code).id)
    if start:
        stmt = stmt.where(PeakEvent.timestamp >= start)
    if end:
        stmt = stmt.where(PeakEvent.timestamp <= end)
    stmt = stmt.order_by(PeakEvent.actual_kwh.desc() if order_by == "kwh" else PeakEvent.timestamp).limit(limit)
    return [dict(row._mapping) for row in session.execute(stmt)]


@router.get("/peaks/summary", response_model=PeakSummary, tags=["peaks"], summary="When and how often peaks occur")
def peaks_summary(
    session: DbSession, meter_code: str | None = None, start: datetime | None = None, end: datetime | None = None
):
    meter = meter_or_404(session, meter_code) if meter_code else None
    return services.peaks_summary(session, meter, start, end)


# ---------------------------------------------------------------- recommendations
@router.get(
    "/recommendations",
    response_model=list[RecommendationOut],
    tags=["recommendations"],
    summary="Latest energy-saving recommendations",
)
def list_recommendations(session: DbSession, meter_code: str | None = None, limit: int = Query(50, ge=1, le=1000)):
    stmt = select(Recommendation).order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
    if meter_code:
        stmt = stmt.where(Recommendation.meter_id == meter_or_404(session, meter_code).id)

    # Each saved forecast re-generates history-based advice; show each distinct message once
    # (its most recent occurrence) instead of repeating it.
    seen, results = set(), []
    for r in session.scalars(stmt.limit(limit * 10)):
        key = (r.meter_id, r.recommendation_type, r.message)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            RecommendationOut(
                recommendation_type=r.recommendation_type,
                severity=r.severity,
                message=r.message,
                timestamp=r.timestamp,
                potential_saving_kwh=r.potential_saving_kwh,
                created_at=r.created_at,
            )
        )
        if len(results) >= limit:
            break
    return results


# ---------------------------------------------------------------- models
@router.get("/model-performance", response_model=ModelPerformanceResponse, tags=["models"])
def model_performance(request: Request, session: DbSession):
    models = session.scalars(select(ModelMetadata).order_by(ModelMetadata.id.desc())).all()
    latest_version = models[0].version if models else None
    current = [m for m in models if m.version == latest_version]

    predictor = getattr(request.app.state, "predictor", None)
    metadata = predictor.metadata if predictor else {}

    importance_path = EXPERIMENTS_DIR / "feature_importance.csv"
    importance = None
    if importance_path.exists():
        with importance_path.open() as fh:
            importance = [{"feature": r["feature"], "importance": float(r["importance"])} for r in csv.DictReader(fh)]

    return ModelPerformanceResponse(
        models=[ModelOut.model_validate(m, from_attributes=True) for m in sorted(current, key=lambda m: m.rmse or 0)],
        active_model=metadata.get("model_name"),
        selection_metric=metadata.get("selection_metric"),
        split=metadata.get("split"),
        feature_columns=metadata.get("feature_columns"),
        feature_importance=importance,
        data_source=metadata.get("source_file") or (current[0].data_source if current else None),
        is_synthetic=metadata.get("is_synthetic", current[0].is_synthetic if current else None),
    )


# ---------------------------------------------------------------- settings
def _settings_out(settings) -> SettingsOut:
    return SettingsOut(
        tariff_per_kwh=float(settings.tariff_per_kwh),
        peak_percentile=float(settings.peak_percentile),
        timezone=settings.timezone,
        currency=settings.currency,
        updated_at=settings.updated_at,
    )


@router.get("/settings", response_model=SettingsOut, tags=["settings"])
def get_settings(session: DbSession):
    return _settings_out(crud.get_settings(session))


@router.put("/settings", response_model=SettingsOut, tags=["settings"], summary="Update tariff / peak percentile")
def update_settings(body: SettingsUpdate, session: DbSession):
    settings, percentile_changed = crud.update_settings(session, **body.model_dump())
    if percentile_changed:
        crud.recompute_peak_events(session, float(settings.peak_percentile))
    return _settings_out(settings)
