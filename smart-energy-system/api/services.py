"""Application services: the logic behind the API endpoints that combines several modules.

Prediction workflow (POST /predict):
    meter history (DB) -> features -> saved model -> forecast -> peak check ->
    estimated cost -> recommendations -> store in DB -> response
"""

import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.schemas.analytics import PeakDate, PeakSummary
from api.schemas.consumption import ConsumptionPoint, Granularity, OverviewResponse
from api.schemas.prediction import HourForecast, PredictResponse, RecommendationOut
from database import crud
from database.models import Meter, ModelMetadata, PeakEvent, Prediction, Recommendation
from src.features.feature_engineering import MAX_LOOKBACK_HOURS
from src.optimization.cost_estimation import COST_DISCLAIMER, ESTIMATED_COST_LABEL, estimate_cost
from src.optimization.peak_detection import compute_thresholds, detect_peaks, peak_summary
from src.optimization.recommendations import generate_recommendations
from src.prediction.predictor import EnergyPredictor

logger = logging.getLogger(__name__)

RESAMPLE_RULE = {Granularity.daily: "D", Granularity.weekly: "W-MON", Granularity.monthly: "MS"}


# ---------------------------------------------------------------- consumption
def consumption_series(df: pd.DataFrame, granularity: Granularity) -> list[ConsumptionPoint]:
    """Sum kWh across meters per hour, then resample to the requested granularity."""
    if df.empty:
        return []
    hourly = df.groupby("timestamp").agg(
        consumption_kwh=("kwh", "sum"),
        avg_voltage=("voltage", "mean"),
        avg_current=("current", "mean"),
        avg_frequency=("frequency", "mean"),
        meter_hours=("kwh", "size"),
    )
    if granularity != Granularity.hourly:
        hourly = hourly.resample(RESAMPLE_RULE[granularity], label="left", closed="left").agg(
            {
                "consumption_kwh": "sum",
                "avg_voltage": "mean",
                "avg_current": "mean",
                "avg_frequency": "mean",
                "meter_hours": "sum",
            }
        )
        hourly = hourly.loc[hourly["meter_hours"] > 0]

    def optional(value) -> float | None:
        return None if pd.isna(value) else float(value)

    return [
        ConsumptionPoint(
            timestamp=ts,
            consumption_kwh=float(row.consumption_kwh),
            avg_voltage=optional(row.avg_voltage),
            avg_current=optional(row.avg_current),
            avg_frequency=optional(row.avg_frequency),
            meter_hours=int(row.meter_hours),
        )
        for ts, row in zip(hourly.index, hourly.itertuples(index=False), strict=True)
    ]


def overview(session: Session, meter: Meter | None, start, end) -> OverviewResponse:
    settings = crud.get_settings(session)
    tariff = float(settings.tariff_per_kwh)
    df = crud.consumption_frame(session, [meter.id] if meter else None, start, end)

    peak_stmt = select(func.count(PeakEvent.id))
    if meter:
        peak_stmt = peak_stmt.where(PeakEvent.meter_id == meter.id)
    if start:
        peak_stmt = peak_stmt.where(PeakEvent.timestamp >= start)
    if end:
        peak_stmt = peak_stmt.where(PeakEvent.timestamp <= end)

    if df.empty:
        total, average, peak_value, peak_ts, latest = 0.0, 0.0, None, None, None
    else:
        per_hour = df.groupby("timestamp")["kwh"].sum()  # combined demand of selected meters
        total = float(per_hour.sum())
        average = float(per_hour.mean())
        peak_value, peak_ts = float(per_hour.max()), per_hour.idxmax()
        latest = df["timestamp"].max()

    return OverviewResponse(
        meter_code=meter.meter_code if meter else None,
        start=start,
        end=end,
        hours_covered=int(df["timestamp"].nunique()) if not df.empty else 0,
        total_kwh=round(total, 4),
        average_hourly_kwh=round(average, 4),
        peak_hourly_kwh=peak_value,
        peak_timestamp=peak_ts,
        peak_event_count=int(session.scalar(peak_stmt) or 0),
        estimated_cost=estimate_cost(total, tariff),
        tariff_per_kwh=tariff,
        currency=settings.currency,
        cost_label=ESTIMATED_COST_LABEL,
        cost_disclaimer=COST_DISCLAIMER,
        latest_timestamp=latest,
    )


# ---------------------------------------------------------------- peaks
def peaks_summary(session: Session, meter: Meter | None, start, end) -> PeakSummary:
    percentile = float(crud.get_settings(session).peak_percentile)
    meter_ids = [meter.id] if meter else None
    full_history = crud.consumption_frame(session, meter_ids)
    thresholds = compute_thresholds(full_history, percentile) if not full_history.empty else pd.Series(dtype=float)

    window = full_history
    if start is not None:
        window = window.loc[window["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        window = window.loc[window["timestamp"] <= pd.Timestamp(end)]

    if window.empty:
        stats = {
            "total_hours": 0,
            "peak_hours": 0,
            "peak_frequency_pct": 0.0,
            "max_kwh": None,
            "by_hour_of_day": {h: 0 for h in range(24)},
            "by_day_of_week": {d: 0 for d in range(7)},
            "most_common_peak_hours": [],
        }
        top_dates = []
    else:
        detected = detect_peaks(window, thresholds)
        stats = peak_summary(detected)
        events = detected.loc[detected["is_peak"]]
        by_date = (
            events.groupby(events["timestamp"].dt.date)["kwh"]
            .agg(["size", "max"])
            .sort_values(["size", "max"], ascending=False)
            .head(10)
        )
        top_dates = [
            PeakDate(date=d, peak_hours=int(r["size"]), max_kwh=float(r["max"])) for d, r in by_date.iterrows()
        ]

    return PeakSummary(
        meter_code=meter.meter_code if meter else None,
        percentile=percentile,
        thresholds={str(k): float(v) for k, v in thresholds.items()},
        top_dates=top_dates,
        **stats,
    )


# ---------------------------------------------------------------- prediction
def _active_model_id(session: Session, predictor: EnergyPredictor) -> int:
    """Find (or register) the model_metadata row of the model currently loaded in memory."""
    record = session.scalar(
        select(ModelMetadata).where(
            ModelMetadata.model_name == predictor.model_name, ModelMetadata.version == predictor.version
        )
    )
    if record is None:
        metrics = predictor.metadata.get("test_metrics", {})
        record = ModelMetadata(
            model_name=predictor.model_name,
            version=predictor.version,
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            r2=metrics.get("r2"),
            nmae_pct=metrics.get("nmae_pct"),
            training_date=datetime.fromisoformat(predictor.metadata["trained_at"]),
            model_path=predictor.metadata.get("model_path"),
            features=predictor.feature_columns,
            hyperparameters=predictor.metadata.get("hyperparameters"),
            is_active=True,
            data_source=predictor.metadata.get("source_file"),
            is_synthetic=bool(predictor.metadata.get("is_synthetic")),
        )
        session.add(record)
        session.flush()
    return record.id


def run_prediction(
    session: Session, predictor: EnergyPredictor, meter: Meter, horizon_hours: int, store: bool
) -> PredictResponse:
    settings = crud.get_settings(session)
    tariff, percentile = float(settings.tariff_per_kwh), float(settings.peak_percentile)

    history = crud.consumption_frame(session, [meter.id])
    latest = history["timestamp"].max() if not history.empty else None
    recent = (
        history.loc[history["timestamp"] > latest - pd.Timedelta(hours=MAX_LOOKBACK_HOURS + 2)]
        if latest is not None
        else history
    )

    # InsufficientHistoryError propagates to a 422 handler in api/main.py.
    forecast = (
        [predictor.predict_next_hour(recent)] if horizon_hours == 1 else predictor.forecast(recent, hours=horizon_hours)
    )
    threshold = float(compute_thresholds(history, percentile).iloc[0])
    logger.info(
        "Prediction meter=%s horizon=%s model=%s first=%.4f kWh",
        meter.meter_code,
        horizon_hours,
        predictor.version,
        forecast[0].predicted_kwh,
    )

    recommendations = generate_recommendations(
        history[["timestamp", "kwh"]], threshold, [(p.timestamp, p.predicted_kwh) for p in forecast]
    )

    if store:
        model_id = _active_model_id(session, predictor)
        session.add_all(
            Prediction(
                meter_id=meter.id,
                timestamp=p.timestamp.to_pydatetime(),
                predicted_kwh=p.predicted_kwh,
                model_id=model_id,
                prediction_type="live",
                horizon_step=p.step,
            )
            for p in forecast
        )
        session.add_all(
            Recommendation(
                meter_id=meter.id,
                timestamp=r.timestamp.to_pydatetime() if r.timestamp is not None else None,
                recommendation_type=r.recommendation_type,
                severity=r.severity,
                message=r.message,
                potential_saving_kwh=r.potential_saving_kwh,
            )
            for r in recommendations
        )
        session.commit()

    notes = [
        "Forecast starts at the hour after the latest stored reading for this meter "
        "(the dataset is historical, so this is not the current wall-clock hour).",
        f"Costs are estimates at a flat tariff of {tariff} {settings.currency}/kWh. {COST_DISCLAIMER}",
    ]
    if horizon_hours > 1:
        notes.append(
            "Hours 2+ use a recursive strategy (each prediction feeds the next). Only next-hour "
            "accuracy was measured on the test set; errors grow with the horizon."
        )
    if predictor.metadata.get("is_synthetic"):
        notes.append("Model was trained on SYNTHETIC data — predictions are for demonstration only.")

    total = sum(p.predicted_kwh for p in forecast)
    return PredictResponse(
        meter_code=meter.meter_code,
        model_name=predictor.model_name,
        model_version=predictor.version,
        based_on_data_until=latest,
        forecast=[
            HourForecast(
                timestamp=p.timestamp,
                predicted_kwh=round(p.predicted_kwh, 4),
                step=p.step,
                is_peak=p.predicted_kwh > threshold,
                estimated_cost=estimate_cost(p.predicted_kwh, tariff),
            )
            for p in forecast
        ],
        total_predicted_kwh=round(total, 4),
        total_estimated_cost=estimate_cost(total, tariff),
        currency=settings.currency,
        cost_label=ESTIMATED_COST_LABEL,
        peak_threshold_kwh=round(threshold, 4),
        peak_percentile=percentile,
        recommendations=[RecommendationOut(**r.to_dict()) for r in recommendations],
        notes=notes,
    )
