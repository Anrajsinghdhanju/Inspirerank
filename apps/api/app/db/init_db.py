from sqlalchemy import text

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.session import engine


def main() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_item_embeddings_multimodal_hnsw
                ON item_embeddings
                USING hnsw (multimodal_embedding vector_cosine_ops)
                """
            )
        )

    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
