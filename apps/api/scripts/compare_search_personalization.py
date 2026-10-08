from __future__ import annotations

import argparse
from itertools import combinations

from app.services.hybrid_recommender import get_recommender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        required=True,
        help='Example: "craft supplies"',
    )
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    recommender = get_recommender()
    demo_users = recommender.demo_users(limit=args.users)

    results = {}

    print(f'\nQuery: "{args.query}"')
    print("=" * 72)

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
            f"specificity={response['query_specificity']:.2f} | "
            f"personalization={response['personalization_strength']:.2%}"
        )

        for rank, item in enumerate(
            response["results"][:5],
            start=1,
        ):
            title = str(item.get("title") or "Untitled")
            print(f"  {rank:>2}. {title[:90]}")

    overlaps = []

    print("\nPairwise top-k overlap")
    print("=" * 72)

    for user_a, user_b in combinations(results, 2):
        a = set(results[user_a])
        b = set(results[user_b])

        overlap_count = len(a & b)
        overlap_rate = overlap_count / args.limit
        overlaps.append(overlap_rate)

        print(
            f"{user_a[:10]}... vs {user_b[:10]}... "
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
