from fastapi import APIRouter
from sqlalchemy import text

from app.db.session import engine

router = APIRouter()


@router.get("/stats")
def stats() -> dict[str, int]:
    with engine.connect() as connection:
        users = connection.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
        items = connection.execute(text("SELECT COUNT(*) FROM items")).scalar_one()
        interactions = connection.execute(
            text("SELECT COUNT(*) FROM interactions")
        ).scalar_one()
        embeddings = connection.execute(
            text("SELECT COUNT(*) FROM item_embeddings")
        ).scalar_one()

    return {
        "users": users,
        "items": items,
        "interactions": interactions,
        "embeddings": embeddings,
    }
