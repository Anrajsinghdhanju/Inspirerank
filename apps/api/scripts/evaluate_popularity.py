from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/recsys/arts_crafts_5core")
ARTIFACT_DIR = Path("artifacts/popularity_baseline")


def evaluate_split(
    train: pd.DataFrame,
    split: pd.DataFrame,
    ks: tuple[int, ...] = (10, 20),
) -> dict:
    popularity = Counter(train["parent_asin"].astype(str))
    ranked_items = [item for item, _ in popularity.most_common()]
    rank_lookup = {item: rank for rank, item in enumerate(ranked_items, start=1)}

    seen = defaultdict(set)
    for row in train.itertuples(index=False):
        seen[str(row.user_id)].add(str(row.parent_asin))

    totals = defaultdict(float)
    targets = 0

    for row in split.itertuples(index=False):
        user = str(row.user_id)
        target = str(row.parent_asin)
        targets += 1

        # Compute target rank after filtering the user's seen items.
        target_global_rank = rank_lookup.get(target)
        if target_global_rank is None:
            rank = None
        else:
            preceding_seen = sum(
                1
                for item in seen[user]
                if item != target
                and rank_lookup.get(item, 10**18) < target_global_rank
            )
            rank = target_global_rank - preceding_seen

        for k in ks:
            hit = rank is not None and rank <= k
            totals[f"recall@{k}"] += 1.0 if hit else 0.0
            totals[f"ndcg@{k}"] += (
                1.0 / math.log2(rank + 1)
                if hit and rank is not None
                else 0.0
            )
        totals["mrr"] += 1.0 / rank if rank is not None else 0.0

    return {
        key: value / targets if targets else 0.0
        for key, value in totals.items()
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "valid.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    result = {
        "validation": evaluate_split(train, valid),
        "test": evaluate_split(train, test),
    }

    print("\nPopularity baseline")
    print("=" * 40)
    for split_name, metrics in result.items():
        print(f"\n{split_name.upper()}")
        for key, value in metrics.items():
            print(f"{key:12} {value:.4f}")

    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
