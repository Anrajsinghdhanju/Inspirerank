"""
Stream a manageable Amazon Reviews 2023 sample into PostgreSQL.

Dataset:
  McAuley-Lab/Amazon-Reviews-2023
Category:
  Handmade_Products

This script uses the raw JSONL files through Hugging Face's resolve endpoint and
`datasets` streaming mode, so a smoke test does not require downloading the full
dataset first.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

from datasets import load_dataset
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Interaction, Item, User
from app.db.session import SessionLocal

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
META_URL = f"{BASE}/meta_categories/meta_Handmade_Products.jsonl"
REVIEWS_URL = f"{BASE}/review_categories/Handmade_Products.jsonl"


def chunks(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def first_image_url(images) -> str | None:
    if not images:
        return None

    # Raw JSONL form: list[dict]
    if isinstance(images, list):
        preferred_keys = ("hi_res", "large", "medium", "small", "thumb")
        for image in images:
            if not isinstance(image, dict):
                continue
            for key in preferred_keys:
                value = image.get(key)
                if value:
                    return value

    # Some converted dataset variants expose dict[list].
    if isinstance(images, dict):
        for key in ("hi_res", "large", "medium", "small", "thumb"):
            values = images.get(key)
            if isinstance(values, list):
                for value in values:
                    if value:
                        return value
            elif values:
                return values

    return None


def compact_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if x)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def ingest_items(max_items: int, batch_size: int) -> set[str]:
    dataset = load_dataset(
        "json",
        data_files=META_URL,
        split="train",
        streaming=True,
    )

    item_ids: set[str] = set()
    batch: list[dict] = []

    with SessionLocal() as session:
        for row in dataset:
            item_id = row.get("parent_asin")
            image_url = first_image_url(row.get("images"))

            # For our visual recommender MVP, keep only items with an image.
            if not item_id or not image_url:
                continue

            item_ids.add(item_id)
            batch.append(
                {
                    "item_id": item_id,
                    "title": row.get("title"),
                    "main_category": row.get("main_category"),
                    "description": compact_text(row.get("description")),
                    "features": compact_text(row.get("features")),
                    "image_url": image_url,
                    "price": None if row.get("price") is None else str(row.get("price")),
                    "average_rating": row.get("average_rating"),
                    "rating_number": row.get("rating_number"),
                }
            )

            if len(batch) >= batch_size:
                statement = insert(Item).values(batch)
                statement = statement.on_conflict_do_nothing(index_elements=["item_id"])
                session.execute(statement)
                session.commit()
                batch.clear()

            if len(item_ids) >= max_items:
                break

        if batch:
            statement = insert(Item).values(batch)
            statement = statement.on_conflict_do_nothing(index_elements=["item_id"])
            session.execute(statement)
            session.commit()

    print(f"Items selected: {len(item_ids):,}")
    return item_ids


def ingest_reviews(item_ids: set[str], max_reviews: int, batch_size: int) -> None:
    dataset = load_dataset(
        "json",
        data_files=REVIEWS_URL,
        split="train",
        streaming=True,
    )

    interaction_batch: list[dict] = []
    user_batch: dict[str, dict] = {}
    inserted_candidate_count = 0

    with SessionLocal() as session:
        for row in dataset:
            item_id = row.get("parent_asin")
            user_id = row.get("user_id")
            timestamp = row.get("timestamp")

            if item_id not in item_ids or not user_id or timestamp is None:
                continue

            user_batch[user_id] = {"user_id": user_id}
            interaction_batch.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": row.get("rating"),
                    "event_timestamp_ms": int(timestamp),
                    "verified_purchase": row.get("verified_purchase"),
                    "review_title": row.get("title"),
                    "review_text": row.get("text"),
                }
            )
            inserted_candidate_count += 1

            if len(interaction_batch) >= batch_size:
                user_stmt = insert(User).values(list(user_batch.values()))
                user_stmt = user_stmt.on_conflict_do_nothing(index_elements=["user_id"])
                session.execute(user_stmt)

                event_stmt = insert(Interaction).values(interaction_batch)
                event_stmt = event_stmt.on_conflict_do_nothing(
                    constraint="uq_interaction_user_item_time"
                )
                session.execute(event_stmt)
                session.commit()

                user_batch.clear()
                interaction_batch.clear()

            if inserted_candidate_count >= max_reviews:
                break

        if interaction_batch:
            user_stmt = insert(User).values(list(user_batch.values()))
            user_stmt = user_stmt.on_conflict_do_nothing(index_elements=["user_id"])
            session.execute(user_stmt)

            event_stmt = insert(Interaction).values(interaction_batch)
            event_stmt = event_stmt.on_conflict_do_nothing(
                constraint="uq_interaction_user_item_time"
            )
            session.execute(event_stmt)
            session.commit()

    print(f"Matching interactions processed: {inserted_candidate_count:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=2_000)
    parser.add_argument("--max-reviews", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    item_ids = ingest_items(args.max_items, args.batch_size)
    ingest_reviews(item_ids, args.max_reviews, args.batch_size)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
