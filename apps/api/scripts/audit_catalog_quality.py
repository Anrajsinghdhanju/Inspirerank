from __future__ import annotations

from collections import Counter

from app.services.hybrid_recommender import get_recommender


def main() -> None:
    recommender = get_recommender()
    summary = recommender.quality_summary()

    print("\nInspireRank catalog quality audit")
    print("=" * 64)
    print(f"catalog items:        {summary['catalog_items']:,}")
    print(f"allowed items:        {summary['allowed_items']:,}")
    print(f"filtered items:       {summary['filtered_items']:,}")
    print(f"allowed rate:         {summary['allowed_rate']:.2%}")

    print("\nFilter reasons")
    print("=" * 64)

    for reason, count in sorted(
        summary["reason_counts"].items(),
        key=lambda pair: pair[1],
        reverse=True,
    ):
        print(f"{reason:32} {count:,}")

    print("\nSample filtered items")
    print("=" * 64)

    shown = 0

    for item_id in recommender.item_ids:
        idx = recommender.item_to_idx[item_id]

        if recommender.quality_allowed[idx].item():
            continue

        metadata = recommender.metadata.get(item_id, {})
        title = str(metadata.get("title") or "")
        reasons = recommender.quality_reasons.get(item_id, ())

        print(
            f"{item_id} | {', '.join(reasons)} | "
            f"{title[:110]}"
        )

        shown += 1
        if shown >= 30:
            break


if __name__ == "__main__":
    main()
