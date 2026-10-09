from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.observability import MODEL_READY, QUERY_ENCODER_READY
from app.middleware.observability import observability_middleware
from app.routes.diagnostics import router as diagnostics_router
from app.routes.discovery import router as discovery_router
from app.routes.feed import router as feed_router
from app.routes.health import router as health_router
from app.routes.interactions import router as interactions_router
from app.routes.metrics import router as metrics_router
from app.routes.production_health import router as production_health_router
from app.routes.recommendations import router as recommendations_router
from app.routes.search import router as search_router
from app.routes.stats import router as stats_router

configure_logging(settings.log_level)
logger = logging.getLogger("inspirerank.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_READY.set(0)
    QUERY_ENCODER_READY.set(0)

    if settings.warm_recommender:
        logger.info("warming_recommender")
        from app.services.hybrid_recommender import get_recommender

        get_recommender()
        MODEL_READY.set(1)
        logger.info("recommender_ready")

    if settings.warm_query_encoder:
        logger.info("warming_query_encoder")
        from app.services.semantic_query_encoder import get_query_encoder

        get_query_encoder().encode("inspirerank warmup")
        QUERY_ENCODER_READY.set(1)
        logger.info("query_encoder_ready")

    yield


app = FastAPI(
    title="InspireRank API",
    version="0.15.0",
    description="Multimodal personalized recommendation and discovery platform.",
    lifespan=lifespan,
)

app.middleware("http")(observability_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(production_health_router)
app.include_router(metrics_router)
app.include_router(health_router)
app.include_router(stats_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(feed_router, prefix="/api/v1")
app.include_router(interactions_router, prefix="/api/v1")
app.include_router(discovery_router, prefix="/api/v1")
app.include_router(diagnostics_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "InspireRank API",
        "status": "running",
        "docs": "/docs",
    }
