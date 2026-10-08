from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.services.hybrid_recommender import get_recommender
from app.services.semantic_query_encoder import get_query_encoder


DEFAULT_QUERIES = [
    "craft supplies",
    "creative gift",
    "DIY project",
    "beginner art supplies",
    "watercolor supplies for beginners",
    "origami paper with Japanese patterns",
]


def mean_pairwise_overlap(
    result_lists: list[list[str]],
    k: int,
) -> float:
    rates = []

    for a, b in combinations(result_lists, 2):
        set_a = set(a[:k])
        set_b = set(b[:k])
        rates.append(
            len(set_a & set_b) / k
        )

    return float(np.mean(rates)) if rates else 0.0


def mean_rank_displacement(
    result_lists: list[list[str]],
    k: int,
) -> float:
    """
    Average normalized rank movement for items shared by a pair of users.
    0 = same ordering for shared items.
    1 = maximally different rank locations.

    Pairwise overlap is reported separately, so this metric focuses only on
    ordering of shared items.
    """
    pair_values = []

    for a, b in combinations(result_lists, 2):
        rank_a = {
            item: rank
            for rank, item in enumerate(a[:k])
        }
        rank_b = {
            item: rank
            for rank, item in enumerate(b[:k])
        }

        shared = set(rank_a) & set(rank_b)

        if not shared:
            pair_values.append(1.0)
            continue

        displacement = np.mean(
            [
                abs(rank_a[item] - rank_b[item])
                / max(k - 1, 1)
                for item in shared
            ]
        )

        pair_values.append(float(displacement))

    return float(np.mean(pair_values)) if pair_values else 0.0


def semantic_metrics(
    recommender,
    query_vector: torch.Tensor,
    item_ids: list[str],
) -> tuple[float, float]:
    indices = torch.tensor(
        [
            recommender.item_to_idx[item_id]
            for item_id in item_ids
        ],
        dtype=torch.long,
        device=recommender.device,
    )

    vectors = recommender.text_matrix[indices]

    relevance = (
        vectors @ query_vector
    ).mean().item()

    if len(vectors) < 2:
        return float(relevance), 0.0

    similarity = vectors @ vectors.T
    upper = torch.triu(
        torch.ones_like(
            similarity,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    pairwise_similarity = similarity[
        upper
    ].mean().item()

    diversity = 1.0 - pairwise_similarity

    return (
        float(relevance),
        float(diversity),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Can be supplied multiple times. Defaults to a mixed query suite.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/search_system_metrics.json"
        ),
    )
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES

    recommender = get_recommender()
    users = recommender.demo_users(
        limit=args.users
    )

    # Warm up lazy SigLIP query encoder so model-loading time does not pollute
    # request latency measurements.
    get_query_encoder().encode(
        "warmup query"
    )

    report = {
        "users": args.users,
        "limit": args.limit,
        "quality": recommender.quality_summary(),
        "queries": [],
    }

    total_latencies = []
    stage_latencies = {
        "query_encode": [],
        "user_profiles": [],
        "candidate_retrieval": [],
        "personalized_rerank": [],
        "mmr": [],
    }

    print("\nInspireRank search-system evaluation")
    print("=" * 78)

    for query in queries:
        result_lists = []
        relevance_values = []
        diversity_values = []
        query_latencies = []

        query_vector = (
            get_query_encoder()
            .encode(query)
            .to(recommender.device)
        )

        per_user = []

        for user in users:
            response = recommender.search(
                user_id=user["user_id"],
                query=query,
                limit=args.limit,
            )

            item_ids = [
                item["item_id"]
                for item in response["results"]
            ]

            result_lists.append(item_ids)

            relevance, diversity = semantic_metrics(
                recommender,
                query_vector,
                item_ids,
            )

            relevance_values.append(relevance)
            diversity_values.append(diversity)

            timings = response["timings_ms"]
            query_latencies.append(
                timings["total"]
            )
            total_latencies.append(
                timings["total"]
            )

            for stage in stage_latencies:
                stage_latencies[stage].append(
                    timings[stage]
                )

            per_user.append(
                {
                    "user_id": user["user_id"],
                    "specificity": response["query_specificity"],
                    "personalization_strength": response[
                        "personalization_strength"
                    ],
                    "semantic_relevance": relevance,
                    "diversity": diversity,
                    "total_latency_ms": timings["total"],
                }
            )

        query_report = {
            "query": query,
            "overlap_at_5": mean_pairwise_overlap(
                result_lists,
                min(5, args.limit),
            ),
            "overlap_at_10": mean_pairwise_overlap(
                result_lists,
                min(10, args.limit),
            ),
            "overlap_at_20": mean_pairwise_overlap(
                result_lists,
                min(20, args.limit),
            ),
            "rank_displacement_at_10": mean_rank_displacement(
                result_lists,
                min(10, args.limit),
            ),
            "semantic_relevance_mean": float(
                np.mean(relevance_values)
            ),
            "diversity_mean": float(
                np.mean(diversity_values)
            ),
            "latency_ms_mean": float(
                np.mean(query_latencies)
            ),
            "per_user": per_user,
        }

        report["queries"].append(
            query_report
        )

        print(f'\n"{query}"')
        print(
            f"  overlap@5/10/20: "
            f"{query_report['overlap_at_5']:.1%} / "
            f"{query_report['overlap_at_10']:.1%} / "
            f"{query_report['overlap_at_20']:.1%}"
        )
        print(
            f"  rank displacement@10: "
            f"{query_report['rank_displacement_at_10']:.3f}"
        )
        print(
            f"  semantic relevance: "
            f"{query_report['semantic_relevance_mean']:.4f}"
        )
        print(
            f"  diversity: "
            f"{query_report['diversity_mean']:.4f}"
        )
        print(
            f"  mean latency: "
            f"{query_report['latency_ms_mean']:.2f} ms"
        )

    latency_summary = {
        "total_p50_ms": percentile(
            total_latencies,
            50,
        ),
        "total_p95_ms": percentile(
            total_latencies,
            95,
        ),
        "total_mean_ms": float(
            np.mean(total_latencies)
        )
        if total_latencies
        else 0.0,
        "stages_mean_ms": {
            stage: float(np.mean(values))
            if values
            else 0.0
            for stage, values
            in stage_latencies.items()
        },
    }

    report["latency"] = latency_summary

    print("\nLatency summary")
    print("=" * 78)
    print(
        f"total p50: {latency_summary['total_p50_ms']:.2f} ms"
    )
    print(
        f"total p95: {latency_summary['total_p95_ms']:.2f} ms"
    )

    for stage, value in latency_summary[
        "stages_mean_ms"
    ].items():
        print(
            f"{stage:24} {value:8.2f} ms"
        )

    print("\nCatalog quality")
    print("=" * 78)
    quality = report["quality"]
    print(
        f"allowed: {quality['allowed_items']:,}/"
        f"{quality['catalog_items']:,} "
        f"({quality['allowed_rate']:.2%})"
    )
    print(
        f"filtered: {quality['filtered_items']:,}"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved -> {args.output}"
    )


if __name__ == "__main__":
    main()
