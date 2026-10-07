from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/recsys/arts_crafts_5core")


def main() -> None:
    path = DATA_DIR / "items.csv"
    items = pd.read_csv(path)

    print("\nInspireRank multimodal catalog")
    print("=" * 55)
    print(f"metadata rows:                {len(items):,}")
    print(f"unique items:                 {items['parent_asin'].nunique():,}")
    print(f"items with text:              {items['has_text'].sum():,}")
    print(f"items with image:             {items['has_image'].sum():,}")
    print(
        f"items with both:              "
        f"{(items['has_text'] & items['has_image']).sum():,}"
    )

    title_lengths = items["title"].fillna("").str.len()
    text_lengths = items["search_text"].fillna("").str.len()

    print(f"median title chars:           {title_lengths.median():.0f}")
    print(f"median combined text chars:   {text_lengths.median():.0f}")

    print("\nSample catalog items")
    print("=" * 55)

    sample_columns = [
        "parent_asin",
        "title",
        "has_image",
    ]

    for row in items[sample_columns].head(10).itertuples(index=False):
        title = str(row.title)
        if len(title) > 90:
            title = title[:87] + "..."
        print(f"{row.parent_asin} | image={row.has_image} | {title}")


if __name__ == "__main__":
    main()
