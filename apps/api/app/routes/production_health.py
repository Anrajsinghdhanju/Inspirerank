from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Response, status
from redis import Redis
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine


router = APIRouter()
ROOT = Path(__file__).resolve().parents[4]

REQUIRED_ARTIFACTS = [
    ROOT / "artifacts/catalog_siglip/item_ids.json",
    ROOT / "artifacts/catalog_siglip/text_embeddings.npy",
    ROOT / "artifacts/content_two_tower_v1/model.pt",
    ROOT / "data/recsys/arts_crafts_5core/train.csv",
    ROOT / "data/recsys/arts_crafts_5core/items.csv",
]


@router.get("/health/live")
def live():
    return {"status": "alive", "environment": settings.app_env}


@router.get("/health/ready")
def ready(response: Response):
    checks = {
        "postgres": False,
        "redis": False,
        "artifacts": False,
    }

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["postgres"] = True
    except Exception:
        pass

    try:
        redis = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        checks["redis"] = bool(redis.ping())
    except Exception:
        pass

    checks["artifacts"] = all(path.exists() for path in REQUIRED_ARTIFACTS)
    is_ready = all(checks.values())

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
    }
