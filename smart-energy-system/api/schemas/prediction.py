"""Schemas for predictions (stored back-tests and live forecasts)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.config import MAX_FORECAST_HORIZON_HOURS


class PredictRequest(BaseModel):
    meter_code: str = Field(min_length=1, max_length=64, examples=["SYN_01"])
    horizon_hours: int = Field(
        default=1,
        ge=1,
        le=MAX_FORECAST_HORIZON_HOURS,
        description="1 = next hour (evaluated use-case). 2–24 = recursive forecast; accuracy degrades with horizon.",
    )
    store: bool = Field(default=True, description="Save predictions and recommendations to the database")


class HourForecast(BaseModel):
    timestamp: datetime
    predicted_kwh: float
    step: int
    is_peak: bool
    estimated_cost: float


class RecommendationOut(BaseModel):
    recommendation_type: str
    severity: Literal["info", "warning", "critical"]
    message: str
    timestamp: datetime | None = None
    potential_saving_kwh: float | None = None
    created_at: datetime | None = None


class PredictResponse(BaseModel):
    meter_code: str
    model_name: str
    model_version: str
    based_on_data_until: datetime
    forecast: list[HourForecast]
    total_predicted_kwh: float
    total_estimated_cost: float
    currency: str
    cost_label: str
    peak_threshold_kwh: float
    peak_percentile: float
    recommendations: list[RecommendationOut]
    notes: list[str]


class PredictionRecord(BaseModel):
    timestamp: datetime
    predicted_kwh: float
    actual_kwh: float | None
    error_kwh: float | None  # predicted - actual
    model_name: str
    model_version: str
    prediction_type: str
    horizon_step: int
