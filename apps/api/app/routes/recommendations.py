from __future__ import annotations

from math import exp
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from app.db.models import Interaction, Item, ItemEmbedding
from app.db.session import SessionLocal

router = APIRouter()


def _rating_weight(rating: float | None) -> float:
    """
    Convert Amazon star ratings into a positive preference weight.

    We intentionally use ratings >= 4 as positive preference signals for this
    baseline. Lower ratings are treated as non-positive and are not used to
    construct the profile.
    """
    if rating is None:
        return 0.0
    if rating >= 5.0:
        return 1.0
    if rating >= 4.0:
        return 0.7
    return 0.0


def _recency_weight(timestamp_ms: int, newest_timestamp_ms: int) -> float:
    """
    Mild exponential decay so more recent interests matter slightly more.

    Half-life ~= 365 days. The Amazon data is historical, so this is primarily
    an engineering baseline rather than a claim about optimal user behavior.
    """
    age_days = max(0.0, (newest_timestamp_ms - timestamp_ms) / 86_400_000)
    return exp(-0.69314718056 * age_days / 365.0)


def _build_profile(rows: list[tuple[Interaction, ItemEmbedding]]) -> np.ndarray:
    newest = max(interaction.event_timestamp_ms for interaction, _ in rows)

    vectors: list[np.ndarray] = []
    weights: list[float] = []

    for interaction, embedding in rows:
        rating_weight = _rating_weight(interaction.rating)
        if rating_weight <= 0:
            continue

        recency = _recency_weight(interaction.event_timestamp_ms, newest)
        weight = rating_weight * recency

        vectors.append(np.asarray(embedding.multimodal_embedding, dtype=np.float32))
        weights.append(weight)

    if not vectors:
        raise ValueError("No positive interactions available for profile construction.")

    matrix = np.stack(vectors)
    weight_array = np.asarray(weights, dtype=np.float32)
    profile = np.average(matrix, axis=0, weights=weight_array)

    norm = np.linalg.norm(profile)
    if norm == 0:
        raise ValueError("User profile has zero magnitude.")

    return profile / norm


@router.get("/users/eligible")
def eligible_users(
    min_interactions: int = Query(default=3, ge=2, le=100),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Find users who have enough interactions WITH embedded catalog items to make
    a meaningful personalized recommendation.
    """
    with SessionLocal() as session:
        statement = (
            select(
                Interaction.user_id,
                func.count(Interaction.id).label("interaction_count"),
            )
            .join(ItemEmbedding, Interaction.item_id == ItemEmbedding.item_id)
            .where(Interaction.rating >= 4.0)
            .group_by(Interaction.user_id)
            .having(func.count(Interaction.id) >= min_interactions)
            .order_by(func.count(Interaction.id).desc())
            .limit(limit)
        )

        rows = session.execute(statement).all()

    return {
        "min_interactions": min_interactions,
        "count": len(rows),
        "users": [
            {
                "user_id": user_id,
                "positive_embedded_interactions": int(interaction_count),
            }
            for user_id, interaction_count in rows
        ],
    }


@router.get("/recommendations/{user_id}")
def recommend_for_user(
    user_id: str,
    limit: int = Query(default=12, ge=1, le=50),
) -> dict[str, Any]:
    """
    Content-based personalization baseline.

    1. Fetch the user's positively rated, embedded items.
    2. Build a weighted multimodal user vector.
    3. Retrieve unseen items nearest to that profile with pgvector.
    """
    with SessionLocal() as session:
        history_statement = (
            select(Interaction, ItemEmbedding)
            .join(ItemEmbedding, Interaction.item_id == ItemEmbedding.item_id)
            .where(Interaction.user_id == user_id)
            .order_by(Interaction.event_timestamp_ms.desc())
        )
        history_rows = list(session.execute(history_statement).all())

        if not history_rows:
            raise HTTPException(
                status_code=404,
                detail="User has no interactions with embedded catalog items.",
            )

        positive_rows = [
            (interaction, embedding)
            for interaction, embedding in history_rows
            if _rating_weight(interaction.rating) > 0
        ]

        if len(positive_rows) < 2:
            raise HTTPException(
                status_code=422,
                detail=(
                    "User needs at least 2 positive interactions with embedded "
                    "items for this personalization baseline."
                ),
            )

        try:
            profile = _build_profile(positive_rows)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        seen_item_ids = {interaction.item_id for interaction, _ in history_rows}

        distance = ItemEmbedding.multimodal_embedding.cosine_distance(
            profile.tolist()
        ).label("distance")

        recommendation_statement = (
            select(Item, distance)
            .join(ItemEmbedding, Item.item_id == ItemEmbedding.item_id)
            .where(Item.item_id.not_in(seen_item_ids))
            .order_by(distance)
            .limit(limit)
        )

        recommendation_rows = session.execute(recommendation_statement).all()

        history_item_ids = [
            interaction.item_id for interaction, _ in positive_rows[:5]
        ]
        history_items = list(
            session.scalars(
                select(Item).where(Item.item_id.in_(history_item_ids))
            ).all()
        )
        item_by_id = {item.item_id: item for item in history_items}

    return {
        "user_id": user_id,
        "strategy": "multimodal_weighted_profile_v1",
        "profile_interactions": len(positive_rows),
        "history_examples": [
            {
                "item_id": interaction.item_id,
                "title": item_by_id.get(interaction.item_id).title
                if item_by_id.get(interaction.item_id)
                else None,
                "rating": interaction.rating,
            }
            for interaction, _ in positive_rows[:5]
        ],
        "recommendations": [
            {
                "item_id": item.item_id,
                "title": item.title,
                "category": item.main_category,
                "image_url": item.image_url,
                "price": item.price,
                "average_rating": item.average_rating,
                "similarity": round(1.0 - float(row_distance), 4),
            }
            for item, row_distance in recommendation_rows
        ],
    }
