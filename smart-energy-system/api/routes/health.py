from fastapi import APIRouter, Request
from sqlalchemy import text

from api.dependencies import DbSession

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service, database and model status")
def health(request: Request, session: DbSession) -> dict:
    try:
        session.execute(text("SELECT 1"))
        database = "ok"
    except Exception:  # noqa: BLE001 — report, don't crash, the health endpoint
        database = "unavailable"
    predictor = getattr(request.app.state, "predictor", None)
    metadata = predictor.metadata if predictor else {}
    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "model_loaded": predictor is not None,
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("version"),
        "data_source": metadata.get("source_file"),
        "is_synthetic": metadata.get("is_synthetic"),
    }
