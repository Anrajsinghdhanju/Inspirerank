from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.search import router as search_router
from app.routes.stats import router as stats_router

app = FastAPI(
    title="InspireRank API",
    version="0.2.0",
    description="Backend API for a multimodal recommendation and discovery system.",
)

app.include_router(health_router)
app.include_router(stats_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "InspireRank API",
        "status": "running",
        "docs": "/docs",
    }
