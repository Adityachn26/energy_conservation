"""ORM table definitions (Phase 15). See docs/DATABASE.md for the ER diagram and rationale.

Conventions:
* ``bigint`` identity primary keys (``BigId``; plain INTEGER on SQLite so tests autoincrement).
* Every foreign-key column is indexed (PostgreSQL does not do this automatically).
* ``timestamp`` columns hold the meter's LOCAL time (no UTC offset in the source data); the
  timezone is recorded once in ``system_settings.timezone``.
* Money (tariff) uses NUMERIC for exact arithmetic; measurements use double precision.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.database import Base

BigId = BigInteger().with_variant(Integer, "sqlite")


class Meter(Base):
    __tablename__ = "meters"

    id: Mapped[int] = mapped_column(BigId, Identity(), primary_key=True)
    meter_code: Mapped[str] = mapped_column(Text, unique=True)
    location: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    readings: Mapped[list["EnergyConsumption"]] = relationship(back_populates="meter")


class EnergyConsumption(Base):
    """Clean hourly consumption (complete hours only — missing hours are simply absent)."""

    __tablename__ = "energy_consumption"
    __table_args__ = (
        # Also serves as the (meter_id, timestamp) composite index and the meter_id FK index.
        UniqueConstraint("meter_id", "timestamp", name="uq_energy_consumption_meter_ts"),
        CheckConstraint("consumption_kwh >= 0", name="ck_energy_consumption_kwh_nonneg"),
        Index("ix_energy_consumption_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigId, Identity(), primary_key=True)
    meter_id: Mapped[int] = mapped_column(ForeignKey("meters.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    consumption_kwh: Mapped[float] = mapped_column(Float)
    avg_voltage: Mapped[float | None] = mapped_column(Float)
    avg_current: Mapped[float | None] = mapped_column(Float)
    frequency: Mapped[float | None] = mapped_column(Float)
    readings_count: Mapped[int] = mapped_column(SmallInteger)

    meter: Mapped[Meter] = relationship(back_populates="readings")


class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    __table_args__ = (UniqueConstraint("model_name", "version", name="uq_model_metadata_name_version"),)

    id: Mapped[int] = mapped_column(BigId, Identity(), primary_key=True)
    model_name: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(Text)
    mae: Mapped[float | None] = mapped_column(Float)
    rmse: Mapped[float | None] = mapped_column(Float)
    r2: Mapped[float | None] = mapped_column(Float)
    nmae_pct: Mapped[float | None] = mapped_column(Float)
    training_date: Mapped[datetime] = mapped_column(DateTime)
    model_path: Mapped[str | None] = mapped_column(Text)  # NULL for models not saved (non-selected)
    features: Mapped[list | None] = mapped_column(JSON)
    hyperparameters: Mapped[dict | None] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)  # the model the API serves
    data_source: Mapped[str | None] = mapped_column(Text)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_meter_ts", "meter_id", "timestamp"),
        Index("ix_predictions_model_id", "model_id"),
        CheckConstraint("prediction_type IN ('backtest', 'live')", name="ck_predictions_type"),
        CheckConstraint("predicted_kwh >= 0", name="ck_predictions_kwh_nonneg"),
    )

    id: Mapped[int] = mapped_column(BigId, Identity(), primary_key=True)
    meter_id: Mapped[int] = mapped_column(ForeignKey("meters.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)  # the hour being predicted
    predicted_kwh: Mapped[float] = mapped_column(Float)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_metadata.id", ondelete="CASCADE"))
    # backtest = test-set prediction from training; live = requested through POST /predict
    prediction_type: Mapped[str] = mapped_column(Text)
    horizon_step: Mapped[int] = mapped_column(SmallInteger, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PeakEvent(Base):
    __tablename__ = "peak_events"
    __table_args__ = (
        UniqueConstraint("meter_id", "timestamp", name="uq_peak_events_meter_ts"),
        CheckConstraint("severity IN ('moderate', 'high', 'critical')", name="ck_peak_events_severity"),
    )

    id: Mapped[int] = mapped_column(BigId, Identity(), primary_key=True)
    meter_id: Mapped[int] = mapped_column(ForeignKey("meters.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    actual_kwh: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    percentile: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(Text)


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (
        Index("ix_recommendations_meter_created", "meter_id", "created_at"),
        CheckConstraint("severity IN ('info', 'warning', 'critical')", name="ck_recommendations_severity"),
    )

    id: Mapped[int] = mapped_column(BigId, Identity(), primary_key=True)
    meter_id: Mapped[int] = mapped_column(ForeignKey("meters.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime | None] = mapped_column(DateTime)  # hour the advice refers to
    recommendation_type: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(Text)
    message: Mapped[str] = mapped_column(Text)
    potential_saving_kwh: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SystemSettings(Base):
    """Single-row table (id is always 1) holding admin-configurable settings."""

    __tablename__ = "system_settings"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_system_settings_single_row"),
        CheckConstraint("tariff_per_kwh >= 0", name="ck_system_settings_tariff_nonneg"),
        CheckConstraint("peak_percentile >= 50 AND peak_percentile < 100", name="ck_system_settings_percentile"),
    )

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=1)
    tariff_per_kwh: Mapped[Decimal] = mapped_column(Numeric(10, 4))
    peak_percentile: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    timezone: Mapped[str] = mapped_column(Text)
    currency: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
