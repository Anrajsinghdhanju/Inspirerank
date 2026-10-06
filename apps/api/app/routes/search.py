from typing import Literal

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.db.models import Item, ItemEmbedding
from app.db.session import SessionLocal
from app.ml.encoder import get_encoder

router = APIRouter()


@router.get("/search")
def semantic_search(
    q: str = Query(min_length=2, max_length=200),
    limit: int = Query(default=12, ge=1, le=50),
    mode: Literal["multimodal", "image", "text"] = "multimodal",
) -> dict:
    query_vector = get_encoder().encode_query(q)

    embedding_column = {
        "multimodal": ItemEmbedding.multimodal_embedding,
        "image": ItemEmbedding.image_embedding,
        "text": ItemEmbedding.text_embedding,
    }[mode]

    distance = embedding_column.cosine_distance(query_vector).label("distance")

    statement = (
        select(Item, distance)
        .join(ItemEmbedding, Item.item_id == ItemEmbedding.item_id)
        .order_by(distance)
        .limit(limit)
    )

    with SessionLocal() as session:
        rows = session.execute(statement).all()

    return {
        "query": q,
        "mode": mode,
        "count": len(rows),
        "results": [
            {
                "item_id": item.item_id,
                "title": item.title,
                "category": item.main_category,
                "image_url": item.image_url,
                "price": item.price,
                "average_rating": item.average_rating,
                "similarity": round(1.0 - float(row_distance), 4),
            }
            for item, row_distance in rows
        ],
    }
