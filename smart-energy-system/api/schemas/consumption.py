"""Schemas for meters, consumption history and overview KPIs."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Granularity(StrEnum):
    hourly = "hourly"
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class MeterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    meter_code: str
    location: str | None
    hourly_records: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None


class ConsumptionPoint(BaseModel):
    timestamp: datetime
    consumption_kwh: float
    avg_voltage: float | None
    avg_current: float | None
    avg_frequency: float | None
    meter_hours: int  # how many complete meter-hours were aggregated into this point


class ConsumptionResponse(BaseModel):
    meter_code: str | None  # None = all meters combined
    granularity: Granularity
    points: list[ConsumptionPoint]


class MeterTotal(BaseModel):
    meter_code: str
    total_kwh: float
    average_hourly_kwh: float
    hours: int


class OverviewResponse(BaseModel):
    meter_code: str | None
    start: datetime | None
    end: datetime | None
    hours_covered: int
    total_kwh: float
    average_hourly_kwh: float
    peak_hourly_kwh: float | None
    peak_timestamp: datetime | None
    peak_event_count: int
    estimated_cost: float
    tariff_per_kwh: float
    currency: str
    cost_label: str
    cost_disclaimer: str
    latest_timestamp: datetime | None
