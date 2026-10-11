"""Reusable database operations used by both the API and the data-loading script.

Keeping queries here (instead of inside API route functions) means they can be tested and reused,
and the API routes stay short and readable.
"""

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from database.models import EnergyConsumption, Meter, ModelMetadata, PeakEvent, SystemSettings
from src.config import APP_TIMEZONE, CURRENCY, DEFAULT_PEAK_PERCENTILE, DEFAULT_TARIFF_PER_KWH
from src.optimization.peak_detection import compute_thresholds, detect_peaks

logger = logging.getLogger(__name__)

INSERT_CHUNK_SIZE = 10_000


def bulk_insert(session: Session, model, rows: Sequence[dict]) -> int:
    """Insert many rows in chunks (one round-trip per chunk instead of one per row)."""
    for start in range(0, len(rows), INSERT_CHUNK_SIZE):
        session.execute(insert(model), list(rows[start : start + INSERT_CHUNK_SIZE]))
    return len(rows)


# ---------------------------------------------------------------- settings
def get_settings(session: Session) -> SystemSettings:
    """Return the single settings row, creating it with defaults on first use."""
    settings = session.get(SystemSettings, 1)
    if settings is None:
        settings = SystemSettings(
            id=1,
            tariff_per_kwh=Decimal(str(DEFAULT_TARIFF_PER_KWH)),
            peak_percentile=Decimal(str(DEFAULT_PEAK_PERCENTILE)),
            timezone=APP_TIMEZONE,
            currency=CURRENCY,
        )
        session.add(settings)
        session.commit()
        logger.info("Created default system settings")
    return settings


def update_settings(session: Session, **changes) -> tuple[SystemSettings, bool]:
    """Apply changes; returns (settings, percentile_changed)."""
    settings = get_settings(session)
    percentile_changed = False
    if changes.get("tariff_per_kwh") is not None:
        settings.tariff_per_kwh = Decimal(str(changes["tariff_per_kwh"]))
    if changes.get("peak_percentile") is not None:
        new_value = Decimal(str(changes["peak_percentile"]))
        percentile_changed = new_value != settings.peak_percentile
        settings.peak_percentile = new_value
    if changes.get("timezone"):
        settings.timezone = changes["timezone"]
    if changes.get("currency"):
        settings.currency = changes["currency"]
    session.commit()
    session.refresh(settings)
    logger.info("Settings updated: %s", {k: v for k, v in changes.items() if v is not None})
    return settings, percentile_changed


# ---------------------------------------------------------------- meters
def get_meter_by_code(session: Session, meter_code: str) -> Meter | None:
    return session.scalar(select(Meter).where(Meter.meter_code == meter_code))


def list_meters_with_stats(session: Session) -> list[dict]:
    stmt = (
        select(
            Meter.id,
            Meter.meter_code,
            Meter.location,
            func.count(EnergyConsumption.id).label("hourly_records"),
            func.min(EnergyConsumption.timestamp).label("first_timestamp"),
            func.max(EnergyConsumption.timestamp).label("last_timestamp"),
        )
        .outerjoin(EnergyConsumption, EnergyConsumption.meter_id == Meter.id)
        .group_by(Meter.id)
        .order_by(Meter.meter_code)
    )
    return [dict(row._mapping) for row in session.execute(stmt)]


# ---------------------------------------------------------------- consumption
def consumption_frame(
    session: Session,
    meter_ids: Iterable[int] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Hourly consumption as a DataFrame: meter_id, meter, timestamp, kwh, voltage, current, frequency."""
    stmt = select(
        EnergyConsumption.meter_id,
        Meter.meter_code.label("meter"),
        EnergyConsumption.timestamp,
        EnergyConsumption.consumption_kwh.label("kwh"),
        EnergyConsumption.avg_voltage.label("voltage"),
        EnergyConsumption.avg_current.label("current"),
        EnergyConsumption.frequency,
    ).join(Meter, Meter.id == EnergyConsumption.meter_id)
    if meter_ids is not None:
        stmt = stmt.where(EnergyConsumption.meter_id.in_(list(meter_ids)))
    if start is not None:
        stmt = stmt.where(EnergyConsumption.timestamp >= start)
    if end is not None:
        stmt = stmt.where(EnergyConsumption.timestamp <= end)
    stmt = stmt.order_by(EnergyConsumption.meter_id, EnergyConsumption.timestamp)

    columns = ["meter_id", "meter", "timestamp", "kwh", "voltage", "current", "frequency"]
    df = pd.DataFrame(session.execute(stmt).all(), columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def meter_peak_threshold(session: Session, meter_id: int, percentile: float) -> float | None:
    history = consumption_frame(session, meter_ids=[meter_id])
    if history.empty:
        return None
    return float(compute_thresholds(history, percentile).iloc[0])


# ---------------------------------------------------------------- peaks
def recompute_peak_events(session: Session, percentile: float) -> int:
    """Rebuild the peak_events table for all meters using the given percentile."""
    df = consumption_frame(session)
    session.execute(delete(PeakEvent))
    if df.empty:
        session.commit()
        return 0
    thresholds = compute_thresholds(df, percentile)
    peaks = detect_peaks(df, thresholds)
    events = peaks.loc[peaks["is_peak"]]
    rows = [
        {
            "meter_id": int(r.meter_id),
            "timestamp": r.timestamp.to_pydatetime(),
            "actual_kwh": float(r.kwh),
            "threshold": float(r.peak_threshold),
            "percentile": float(percentile),
            "severity": r.severity,
        }
        for r in events.itertuples()
    ]
    bulk_insert(session, PeakEvent, rows)
    session.commit()
    logger.info("Peak events recomputed at P%s: %s events", percentile, f"{len(rows):,}")
    return len(rows)


# ---------------------------------------------------------------- models
def get_active_model(session: Session) -> ModelMetadata | None:
    return session.scalar(
        select(ModelMetadata).where(ModelMetadata.is_active.is_(True)).order_by(ModelMetadata.id.desc())
    )
