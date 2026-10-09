from fastapi import APIRouter

from app.services.hybrid_recommender import get_recommender
from app.services.semantic_query_encoder import get_query_encoder


router = APIRouter()


@router.get("/diagnostics/catalog-quality")
def catalog_quality():
    return get_recommender().quality_summary()


@router.get("/diagnostics/query-cache")
def query_cache():
    return get_query_encoder().cache_info()
