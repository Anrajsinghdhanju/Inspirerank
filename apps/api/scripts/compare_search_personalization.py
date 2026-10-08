from __future__ import annotations

import argparse
from itertools import combinations

from app.services.hybrid_recommender import get_recommender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    recommender = get_recommender()
    demo_users = recommender.demo_users(
        limit=args.users
    )

    results = {}

    print(
        f'\nQuery: "{args.query}"'
    )
    print("=" * 76)

    for user in demo_users:
        user_id = user["user_id"]

        response = recommender.search(
            user_id=user_id,
            query=args.query,
            limit=args.limit,
        )

        item_ids = [
            item["item_id"]
            for item in response["results"]
        ]
        results[user_id] = item_ids

        print(
            f"\nUser {user_id[:14]}... | "
            f"history={user['history_count']} | "
            f"specificity="
            f"{response['query_specificity']:.2f} | "
            f"personalization="
            f"{response['personalization_strength']:.2%}"
        )

        print("  Query-conditioned history:")
        for history_item in response[
            "query_conditioned_history"
        ][:3]:
            print(
                "   - "
                f"{history_item['title'][:78]} "
                f"(attention={history_item['attention']:.2f})"
            )

        print("  Top results:")
        for rank, item in enumerate(
            response["results"][:5],
            start=1,
        ):
            title = str(
                item.get("title")
                or "Untitled"
            )
            print(
                f"   {rank:>2}. "
                f"{title[:90]}"
            )

    overlaps = []

    print(
        "\nPairwise top-k overlap"
    )
    print("=" * 76)

    for user_a, user_b in combinations(
        results,
        2,
    ):
        set_a = set(
            results[user_a]
        )
        set_b = set(
            results[user_b]
        )

        overlap_count = len(
            set_a & set_b
        )
        overlap_rate = (
            overlap_count
            / args.limit
        )
        overlaps.append(
            overlap_rate
        )

        print(
            f"{user_a[:10]}... vs "
            f"{user_b[:10]}... "
            f"{overlap_count}/{args.limit} "
            f"({overlap_rate:.1%})"
        )

    if overlaps:
        print(
            f"\nMean pairwise overlap@{args.limit}: "
            f"{sum(overlaps) / len(overlaps):.1%}"
        )


if __name__ == "__main__":
    main()
