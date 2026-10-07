from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATA_DIR = Path("data/recsys/arts_crafts_5core")

PARQUET_URLS = [
    (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
        "resolve/main/raw_meta_Arts_Crafts_and_Sewing/full-00000-of-00004.parquet"
    ),
    (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
        "resolve/main/raw_meta_Arts_Crafts_and_Sewing/full-00001-of-00004.parquet"
    ),
    (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
        "resolve/main/raw_meta_Arts_Crafts_and_Sewing/full-00002-of-00004.parquet"
    ),
    (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
        "resolve/main/raw_meta_Arts_Crafts_and_Sewing/full-00003-of-00004.parquet"
    ),
]


def flatten_text(value) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(
            flatten_text(item)
            for item in value
            if item is not None and flatten_text(item)
        ).strip()

    if isinstance(value, dict):
        pieces = []
        for key, item in value.items():
            text = flatten_text(item)
            if text:
                pieces.append(f"{key}: {text}")
        return " ".join(pieces).strip()

    return str(value).strip()


def _first_nonempty(value):
    if isinstance(value, list):
        for item in value:
            if item:
                return item
        return None
    return value or None


def choose_image_url(images) -> str:
    """
    Amazon metadata can expose images either as a dict of parallel arrays or,
    in some raw forms, as a list of dictionaries.

    Prefer the MAIN image and the highest practical resolution.
    """
    if not images:
        return ""

    # Converted Parquet/Hugging Face form: dict[list]
    if isinstance(images, dict):
        variants = images.get("variant") or []
        hi_res = images.get("hi_res") or []
        large = images.get("large") or []
        thumb = images.get("thumb") or []

        # Try the MAIN image first.
        for idx, variant in enumerate(variants):
            if variant == "MAIN":
                for collection in (hi_res, large, thumb):
                    if idx < len(collection) and collection[idx]:
                        return str(collection[idx])

        # Fall back to the first valid image.
        for collection in (hi_res, large, thumb):
            value = _first_nonempty(collection)
            if value:
                return str(value)

    # Raw JSON style: list[dict]
    if isinstance(images, list):
        main_candidates = [
            image for image in images
            if isinstance(image, dict) and image.get("variant") == "MAIN"
        ]
        candidates = main_candidates + [
            image for image in images
            if isinstance(image, dict) and image not in main_candidates
        ]

        for image in candidates:
            for key in ("hi_res", "large", "medium", "small", "thumb"):
                if image.get(key):
                    return str(image[key])

    return ""


def build_catalog_ids() -> set[str]:
    ids: set[str] = set()

    for split in ("train", "valid", "test"):
        frame = pd.read_csv(
            DATA_DIR / f"{split}.csv",
            usecols=["parent_asin"],
        )
        ids.update(frame["parent_asin"].astype(str))

    return ids


def build_search_text(row: dict) -> str:
    parts = [
        flatten_text(row.get("title")),
        flatten_text(row.get("main_category")),
        flatten_text(row.get("categories")),
        flatten_text(row.get("features")),
        flatten_text(row.get("description")),
        flatten_text(row.get("details")),
    ]

    return " ".join(part for part in parts if part).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "items.csv",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=DATA_DIR / "missing_metadata_items.txt",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit on target item IDs.",
    )
    args = parser.parse_args()

    catalog_ids = build_catalog_ids()

    if args.limit is not None:
        catalog_ids = set(sorted(catalog_ids)[: args.limit])

    print(f"Target catalog IDs: {len(catalog_ids):,}")
    print("Streaming official Arts, Crafts & Sewing metadata...")

    dataset = load_dataset(
        "parquet",
        data_files=PARQUET_URLS,
        split="train",
        streaming=True,
    )

    rows: list[dict] = []
    found: set[str] = set()
    scanned = 0

    for row in dataset:
        scanned += 1

        parent_asin = str(row.get("parent_asin") or "")
        if parent_asin not in catalog_ids:
            if scanned % 100_000 == 0:
                print(
                    f"scanned={scanned:,} "
                    f"matched={len(found):,}/{len(catalog_ids):,}"
                )
            continue

        if parent_asin in found:
            continue

        title = flatten_text(row.get("title"))
        description = flatten_text(row.get("description"))
        features = flatten_text(row.get("features"))
        categories = flatten_text(row.get("categories"))
        details = flatten_text(row.get("details"))
        image_url = choose_image_url(row.get("images"))

        rows.append(
            {
                "parent_asin": parent_asin,
                "title": title,
                "main_category": flatten_text(row.get("main_category")),
                "description": description,
                "features": features,
                "categories": categories,
                "details": details,
                "image_url": image_url,
                "average_rating": row.get("average_rating"),
                "rating_number": row.get("rating_number"),
                "price": row.get("price"),
                "search_text": build_search_text(row),
                "has_text": bool(title or description or features or categories),
                "has_image": bool(image_url),
            }
        )
        found.add(parent_asin)

        if len(found) % 2_500 == 0:
            print(
                f"scanned={scanned:,} "
                f"matched={len(found):,}/{len(catalog_ids):,}"
            )

        if len(found) == len(catalog_ids):
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)

    catalog = pd.DataFrame(rows)
    if not catalog.empty:
        catalog = catalog.sort_values("parent_asin").reset_index(drop=True)
    catalog.to_csv(args.output, index=False)

    missing = sorted(catalog_ids - found)
    args.missing_output.write_text(
        "\n".join(missing),
        encoding="utf-8",
    )

    total = len(catalog_ids)
    matched = len(catalog)
    text_count = int(catalog["has_text"].sum()) if matched else 0
    image_count = int(catalog["has_image"].sum()) if matched else 0
    both_count = (
        int((catalog["has_text"] & catalog["has_image"]).sum())
        if matched
        else 0
    )

    print("\nCatalog metadata coverage")
    print("=" * 55)
    print(f"target items:                 {total:,}")
    print(f"metadata matched:             {matched:,} ({matched / total:.2%})")
    print(f"items with text:              {text_count:,} ({text_count / total:.2%})")
    print(f"items with image URL:         {image_count:,} ({image_count / total:.2%})")
    print(f"items with text + image:      {both_count:,} ({both_count / total:.2%})")
    print(f"missing metadata:             {len(missing):,}")
    print(f"\nSaved catalog -> {args.output}")
    print(f"Saved missing IDs -> {args.missing_output}")


if __name__ == "__main__":
    main()
