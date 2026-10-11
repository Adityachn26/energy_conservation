"""FastAPI application entry point.

Run locally:   uvicorn api.main:app --reload
Docs:          http://localhost:8000/docs
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from api.routes import analytics, consumption, health, predictions
from src.config import setup_logging
from src.prediction.predictor import EnergyPredictor, InsufficientHistoryError

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    # Load the trained model ONCE at start-up — never retrain inside the app.
    try:
        app.state.predictor = EnergyPredictor.load()
    except FileNotFoundError as exc:
        app.state.predictor = None
        logger.warning("Starting without a model: %s", exc)
    yield


app = FastAPI(
    title="Smart Energy Consumption API",
    description=(
        "Hourly electricity consumption analytics, next-hour forecasting, peak detection, "
        "estimated cost and rule-based recommendations (CEEW smart-meter data)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        1000 * (time.perf_counter() - started),
    )
    return response


@app.exception_handler(InsufficientHistoryError)
async def insufficient_history_handler(request: Request, exc: InsufficientHistoryError):
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    # Log full details server-side; never leak internals (e.g. SQL, credentials) to the client.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health.router)
app.include_router(consumption.router)
app.include_router(predictions.router)
app.include_router(analytics.router)
