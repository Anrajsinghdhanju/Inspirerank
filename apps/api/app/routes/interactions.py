from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.hybrid_recommender import get_recommender
from app.services.realtime_feedback import get_feedback_store

router = APIRouter()


class InteractionRequest(BaseModel):
    user_id: str
    item_id: str
    event_type: Literal["like", "save", "not_interested"]


@router.post("/interactions")
def create_interaction(payload: InteractionRequest):
    recommender = get_recommender()

    if not recommender.has_user(payload.user_id):
        raise HTTPException(
            status_code=404,
            detail="User not found in recommendation cohort.",
        )

    if not recommender.has_item(payload.item_id):
        raise HTTPException(
            status_code=404,
            detail="Item not found in recommendation catalog.",
        )

    store = get_feedback_store()

    store.persist_event(
        payload.user_id,
        payload.item_id,
        payload.event_type,
    )

    realtime_updated = store.push_recent_event(
        payload.user_id,
        payload.item_id,
        payload.event_type,
    )

    return {
        "status": "recorded",
        "user_id": payload.user_id,
        "item_id": payload.item_id,
        "event_type": payload.event_type,
        "persisted": True,
        "realtime_updated": realtime_updated,
        "feed_should_refresh": realtime_updated,
    }


@router.get("/interactions/{user_id}/recent")
def recent_interactions(user_id: str):
    recommender = get_recommender()

    if not recommender.has_user(user_id):
        raise HTTPException(
            status_code=404,
            detail="User not found in recommendation cohort.",
        )

    events = get_feedback_store().recent_events(user_id)

    return {
        "user_id": user_id,
        "count": len(events),
        "events": events,
    }


@router.delete("/interactions/{user_id}/recent")
def clear_recent_interactions(user_id: str):
    recommender = get_recommender()

    if not recommender.has_user(user_id):
        raise HTTPException(
            status_code=404,
            detail="User not found in recommendation cohort.",
        )

    cleared = get_feedback_store().clear_recent_events(user_id)

    return {
        "user_id": user_id,
        "cleared": cleared,
    }
