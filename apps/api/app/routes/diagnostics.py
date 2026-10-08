from fastapi import APIRouter

from app.services.hybrid_recommender import get_recommender


router = APIRouter()


@router.get("/diagnostics/catalog-quality")
def catalog_quality():
    return get_recommender().quality_summary()
