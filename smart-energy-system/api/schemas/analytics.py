"""Schemas for peaks, model performance and settings."""

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

from src.optimization.cost_estimation import MAX_TARIFF_PER_KWH


class PeakEventOut(BaseModel):
    meter_code: str
    timestamp: datetime
    actual_kwh: float
    threshold: float
    percentile: float
    severity: str


class PeakDate(BaseModel):
    date: date
    peak_hours: int
    max_kwh: float


class PeakSummary(BaseModel):
    meter_code: str | None
    percentile: float
    thresholds: dict[str, float]  # meter_code -> threshold kWh
    total_hours: int
    peak_hours: int
    peak_frequency_pct: float
    max_kwh: float | None
    by_hour_of_day: dict[int, int]
    by_day_of_week: dict[int, int]  # 0 = Monday
    most_common_peak_hours: list[int]
    top_dates: list[PeakDate]


class ModelOut(BaseModel):
    model_name: str
    version: str
    mae: float | None
    rmse: float | None
    r2: float | None
    nmae_pct: float | None
    is_active: bool
    training_date: datetime
    hyperparameters: dict | None


class ModelPerformanceResponse(BaseModel):
    models: list[ModelOut]
    active_model: str | None
    selection_metric: str | None
    evaluation_set: str = "test (chronologically last ~15%, never used for training or selection)"
    split: dict | None
    feature_columns: list[str] | None
    feature_importance: list[dict] | None
    data_source: str | None
    is_synthetic: bool | None


class SettingsOut(BaseModel):
    tariff_per_kwh: float
    peak_percentile: float
    timezone: str
    currency: str
    updated_at: datetime | None


class SettingsUpdate(BaseModel):
    tariff_per_kwh: float | None = Field(default=None, ge=0, le=MAX_TARIFF_PER_KWH)
    peak_percentile: float | None = Field(default=None, ge=50, lt=100)
    timezone: str | None = Field(default=None, max_length=64)
    currency: str | None = Field(default=None, min_length=3, max_length=3)

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str | None) -> str | None:
        if value is None:
            return value
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown timezone: {value}") from exc
        return value

    @field_validator("currency")
    @classmethod
    def _upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value
