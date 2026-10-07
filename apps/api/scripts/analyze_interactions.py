from __future__ import annotations

from sqlalchemy import text

from app.db.session import engine


QUERIES = {
    "users": "SELECT COUNT(*) FROM users",
    "items": "SELECT COUNT(*) FROM items",
    "interactions": "SELECT COUNT(*) FROM interactions",
    "embeddings": "SELECT COUNT(*) FROM item_embeddings",
    "interactions_on_embedded_items": """
        SELECT COUNT(*)
        FROM interactions i
        JOIN item_embeddings e ON e.item_id = i.item_id
    """,
    "users_with_2_positive_embedded": """
        SELECT COUNT(*)
        FROM (
            SELECT i.user_id
            FROM interactions i
            JOIN item_embeddings e ON e.item_id = i.item_id
            WHERE i.rating >= 4
            GROUP BY i.user_id
            HAVING COUNT(*) >= 2
        ) x
    """,
    "users_with_3_positive_embedded": """
        SELECT COUNT(*)
        FROM (
            SELECT i.user_id
            FROM interactions i
            JOIN item_embeddings e ON e.item_id = i.item_id
            WHERE i.rating >= 4
            GROUP BY i.user_id
            HAVING COUNT(*) >= 3
        ) x
    """,
    "users_with_5_positive_embedded": """
        SELECT COUNT(*)
        FROM (
            SELECT i.user_id
            FROM interactions i
            JOIN item_embeddings e ON e.item_id = i.item_id
            WHERE i.rating >= 4
            GROUP BY i.user_id
            HAVING COUNT(*) >= 5
        ) x
    """,
}


def main() -> None:
    with engine.connect() as connection:
        print("\nInspireRank data coverage")
        print("=" * 45)
        for name, sql in QUERIES.items():
            value = connection.execute(text(sql)).scalar_one()
            print(f"{name:34} {value:,}")

        print("\nTop users by positive embedded interactions")
        print("=" * 45)
        rows = connection.execute(
            text(
                """
                SELECT i.user_id, COUNT(*) AS n
                FROM interactions i
                JOIN item_embeddings e ON e.item_id = i.item_id
                WHERE i.rating >= 4
                GROUP BY i.user_id
                ORDER BY n DESC
                LIMIT 10
                """
            )
        ).all()

        if not rows:
            print("No eligible users yet.")
        else:
            for user_id, count in rows:
                print(f"{user_id:24} {count}")


if __name__ == "__main__":
    main()
