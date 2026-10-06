from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.stats import router as stats_router

app = FastAPI(
    title="InspireRank API",
    version="0.1.0",
    description="Backend API for a multimodal recommendation and discovery system.",
)

app.include_router(health_router)
app.include_router(stats_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "InspireRank API",
        "status": "running",
        "docs": "/docs",
    }
