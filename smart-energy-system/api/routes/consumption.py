from datetime import datetime

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from api import services
from api.dependencies import DbSession, meter_or_404
from api.schemas.consumption import (
    ConsumptionResponse,
    Granularity,
    MeterOut,
    MeterTotal,
    OverviewResponse,
)
from database import crud
from database.models import EnergyConsumption, Meter

router = APIRouter(tags=["consumption"])


@router.get("/meters", response_model=list[MeterOut], summary="All meters with data coverage")
def list_meters(session: DbSession):
    return crud.list_meters_with_stats(session)


@router.get("/consumption", response_model=ConsumptionResponse, summary="Historical consumption")
def get_consumption(
    session: DbSession,
    meter_code: str | None = Query(None, description="Omit for all meters combined"),
    start: datetime | None = None,
    end: datetime | None = None,
    granularity: Granularity = Granularity.hourly,
):
    meter = meter_or_404(session, meter_code) if meter_code else None
    df = crud.consumption_frame(session, [meter.id] if meter else None, start, end)
    return ConsumptionResponse(
        meter_code=meter_code, granularity=granularity, points=services.consumption_series(df, granularity)
    )


@router.get("/consumption/by-meter", response_model=list[MeterTotal], summary="Total consumption per meter")
def consumption_by_meter(session: DbSession, start: datetime | None = None, end: datetime | None = None):
    stmt = (
        select(
            Meter.meter_code,
            func.sum(EnergyConsumption.consumption_kwh).label("total_kwh"),
            func.avg(EnergyConsumption.consumption_kwh).label("average_hourly_kwh"),
            func.count(EnergyConsumption.id).label("hours"),
        )
        .join(Meter, Meter.id == EnergyConsumption.meter_id)
        .group_by(Meter.meter_code)
        .order_by(func.sum(EnergyConsumption.consumption_kwh).desc())
    )
    if start:
        stmt = stmt.where(EnergyConsumption.timestamp >= start)
    if end:
        stmt = stmt.where(EnergyConsumption.timestamp <= end)
    return [dict(row._mapping) for row in session.execute(stmt)]


@router.get("/overview", response_model=OverviewResponse, summary="Dashboard KPIs for a meter/period")
def get_overview(
    session: DbSession,
    meter_code: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
):
    meter = meter_or_404(session, meter_code) if meter_code else None
    return services.overview(session, meter, start, end)
