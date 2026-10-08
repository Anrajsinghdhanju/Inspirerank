from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.discovery import router as discovery_router
from app.routes.feed import router as feed_router
from app.routes.health import router as health_router
from app.routes.interactions import router as interactions_router
from app.routes.recommendations import router as recommendations_router
from app.routes.search import router as search_router
from app.routes.stats import router as stats_router


app = FastAPI(
    title="InspireRank API",
    version="0.11.0",
    description=(
        "Multimodal personalized recommendation "
        "and discovery platform."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(stats_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(recommendations_router, prefix="/api/v1")
app.include_router(feed_router, prefix="/api/v1")
app.include_router(interactions_router, prefix="/api/v1")
app.include_router(discovery_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "InspireRank API",
        "status": "running",
        "docs": "/docs",
    }
