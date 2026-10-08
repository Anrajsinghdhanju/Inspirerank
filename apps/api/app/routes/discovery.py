from fastapi import APIRouter, HTTPException, Query

from app.services.hybrid_recommender import get_recommender


router = APIRouter()


@router.get("/discovery/search")
def personalized_semantic_search(
    user_id: str,
    q: str = Query(min_length=2, max_length=300),
    limit: int = Query(default=24, ge=1, le=50),
):
    recommender = get_recommender()

    if not recommender.has_user(user_id):
        raise HTTPException(
            status_code=404,
            detail="User not found in recommendation cohort.",
        )

    return recommender.search(
        user_id=user_id,
        query=q,
        limit=limit,
    )
