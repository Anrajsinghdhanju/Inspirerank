from sqlalchemy import text

from app.db.base import Base
from app.db.session import engine

# Import models so SQLAlchemy registers their metadata.
from app.db import models  # noqa: F401


def main() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
