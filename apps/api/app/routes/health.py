from fastapi import APIRouter, HTTPException
from redis import Redis
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dependency check failed: {exc}") from exc

    return {
        "status": "ok",
        "postgres": "ok",
        "redis": "ok",
    }
