from fastapi import APIRouter, HTTPException, Query

from app.services.hybrid_recommender import get_recommender

router = APIRouter()


@router.get("/demo-users")
def demo_users(
    limit: int = Query(default=20, ge=1, le=100),
):
    recommender = get_recommender()
    return {
        "users": recommender.demo_users(limit=limit),
    }


@router.get("/feed/{user_id}")
def personalized_feed(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=50),
):
    recommender = get_recommender()

    try:
        return recommender.recommend(user_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="User not found in the recommendation cohort.",
        ) from exc
