from __future__ import annotations

import argparse
from io import BytesIO

import httpx
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Item, ItemEmbedding
from app.db.session import SessionLocal
from app.ml.encoder import MODEL_NAME, get_encoder


def build_item_text(item: Item) -> str:
    parts = [
        item.title or "",
        item.main_category or "",
        item.description or "",
        item.features or "",
    ]
    text = ". ".join(part.strip() for part in parts if part and part.strip())
    return text[:2_000] or "product"


def download_image(client: httpx.Client, url: str) -> Image.Image:
    response = client.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")


def load_items(limit: int, overwrite: bool) -> list[Item]:
    with SessionLocal() as session:
        statement = select(Item).where(Item.image_url.is_not(None)).order_by(Item.item_id)

        if not overwrite:
            statement = (
                statement.outerjoin(ItemEmbedding, Item.item_id == ItemEmbedding.item_id)
                .where(ItemEmbedding.item_id.is_(None))
            )

        return list(session.scalars(statement.limit(limit)).all())


def save_batch(
    item_ids: list[str],
    image_embeddings: list[list[float]],
    text_embeddings: list[list[float]],
    multimodal_embeddings: list[list[float]],
) -> None:
    rows = [
        {
            "item_id": item_id,
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "multimodal_embedding": multimodal_embedding,
            "model_name": MODEL_NAME,
        }
        for item_id, image_embedding, text_embedding, multimodal_embedding in zip(
            item_ids,
            image_embeddings,
            text_embeddings,
            multimodal_embeddings,
            strict=True,
        )
    ]

    with SessionLocal() as session:
        statement = insert(ItemEmbedding).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=["item_id"],
            set_={
                "image_embedding": statement.excluded.image_embedding,
                "text_embedding": statement.excluded.text_embedding,
                "multimodal_embedding": statement.excluded.multimodal_embedding,
                "model_name": statement.excluded.model_name,
            },
        )
        session.execute(statement)
        session.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    items = load_items(args.limit, args.overwrite)
    if not items:
        print("No items need embeddings.")
        return

    encoder = get_encoder()
    print(f"Device: {encoder.device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Items queued: {len(items):,}")

    processed = 0
    failed = 0

    with httpx.Client(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "InspireRank/0.2 educational-project"},
    ) as client:
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]

            valid_ids: list[str] = []
            valid_images: list[Image.Image] = []
            valid_texts: list[str] = []

            for item in batch_items:
                try:
                    image = download_image(client, item.image_url)
                except (
                    httpx.HTTPError,
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                ) as exc:
                    failed += 1
                    print(f"Skipping {item.item_id}: {type(exc).__name__}")
                    continue

                valid_ids.append(item.item_id)
                valid_images.append(image)
                valid_texts.append(build_item_text(item))

            if not valid_ids:
                continue

            image_vectors, text_vectors, multimodal_vectors = encoder.encode_items(
                valid_images,
                valid_texts,
            )
            save_batch(
                valid_ids,
                image_vectors,
                text_vectors,
                multimodal_vectors,
            )

            processed += len(valid_ids)
            print(
                f"Embedded {processed:,}/{len(items):,} "
                f"(failed image downloads: {failed:,})"
            )

    print(f"Done. Embedded: {processed:,}; failed: {failed:,}")


if __name__ == "__main__":
    main()
