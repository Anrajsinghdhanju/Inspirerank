from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from ml_recsys.two_tower import TwoTowerModel


DATA_DIR = Path("data/recsys/arts_crafts_5core")
ARTIFACT_DIR = Path("artifacts/two_tower_v1")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_seen_items(
    train: pd.DataFrame,
    item_to_idx: dict[str, int],
) -> dict[str, set[int]]:
    seen: dict[str, set[int]] = defaultdict(set)

    for row in train.itertuples(index=False):
        item = str(row.parent_asin)
        if item in item_to_idx:
            seen[str(row.user_id)].add(item_to_idx[item])

    return seen


def rank_metrics(rank: int | None, ks: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    for k in ks:
        hit = rank is not None and rank <= k
        metrics[f"recall@{k}"] = 1.0 if hit else 0.0
        metrics[f"ndcg@{k}"] = (
            1.0 / math.log2(rank + 1)
            if hit and rank is not None
            else 0.0
        )

    metrics["mrr"] = 1.0 / rank if rank is not None else 0.0
    return metrics


@torch.inference_mode()
def evaluate(
    model: TwoTowerModel,
    split: pd.DataFrame,
    train: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    device: torch.device,
    batch_size: int,
    ks: tuple[int, ...] = (10, 20),
) -> dict:
    model.eval()

    all_item_ids = torch.arange(len(item_to_idx), device=device)
    item_matrix = model.encode_items(all_item_ids)

    seen_by_user = build_seen_items(train, item_to_idx)

    totals = defaultdict(float)
    warm_totals = defaultdict(float)

    total_targets = 0
    warm_targets = 0
    cold_targets = 0

    records = split[["user_id", "parent_asin"]].copy()
    records["user_id"] = records["user_id"].astype(str)
    records["parent_asin"] = records["parent_asin"].astype(str)

    for start in range(0, len(records), batch_size):
        batch = records.iloc[start : start + batch_size]

        usable = [
            (row.user_id, row.parent_asin)
            for row in batch.itertuples(index=False)
            if row.user_id in user_to_idx
        ]

        if not usable:
            continue

        user_indices = torch.tensor(
            [user_to_idx[user] for user, _ in usable],
            dtype=torch.long,
            device=device,
        )

        user_matrix = model.encode_users(user_indices)
        scores = user_matrix @ item_matrix.T

        for row_index, (user_id, target_item) in enumerate(usable):
            total_targets += 1

            target_idx = item_to_idx.get(target_item)
            if target_idx is None:
                cold_targets += 1

                cold_metrics = rank_metrics(None, ks)
                for key, value in cold_metrics.items():
                    totals[key] += value
                continue

            warm_targets += 1

            row_scores = scores[row_index]

            # Mask historical items so we recommend unseen content, but never
            # mask the held-out target if the product happened to repeat.
            for seen_idx in seen_by_user.get(user_id, set()):
                if seen_idx != target_idx:
                    row_scores[seen_idx] = -torch.inf

            target_score = row_scores[target_idx]
            rank = int((row_scores > target_score).sum().item()) + 1

            metrics = rank_metrics(rank, ks)
            for key, value in metrics.items():
                totals[key] += value
                warm_totals[key] += value

    result = {
        "targets": total_targets,
        "warm_targets": warm_targets,
        "cold_targets": cold_targets,
        "warm_coverage": warm_targets / total_targets if total_targets else 0.0,
        "cold_item_rate": cold_targets / total_targets if total_targets else 0.0,
        "max_possible_overall_recall_for_id_only": (
            warm_targets / total_targets if total_targets else 0.0
        ),
        "overall": {
            key: value / total_targets if total_targets else 0.0
            for key, value in totals.items()
        },
        "warm_only": {
            key: value / warm_targets if warm_targets else 0.0
            for key, value in warm_totals.items()
        },
    }

    return result


def print_metrics(name: str, result: dict) -> None:
    print(f"\n{name}")
    print("=" * 60)
    print(f"targets:                    {result['targets']:,}")
    print(f"warm targets:               {result['warm_targets']:,}")
    print(f"cold targets:               {result['cold_targets']:,}")
    print(f"warm coverage:               {result['warm_coverage']:.2%}")
    print(
        "ID-only max overall recall: "
        f"{result['max_possible_overall_recall_for_id_only']:.2%}"
    )

    print("\nOverall metrics (cold items count as misses)")
    for key, value in result["overall"].items():
        print(f"{key:12} {value:.4f}")

    print("\nWarm-item metrics")
    for key, value in result["warm_only"].items():
        print(f"{key:12} {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "valid.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    for frame in (train, valid, test):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["parent_asin"] = frame["parent_asin"].astype(str)

    user_to_idx = {
        key: int(value)
        for key, value in load_json(ARTIFACT_DIR / "user_to_idx.json").items()
    }
    item_to_idx = {
        key: int(value)
        for key, value in load_json(ARTIFACT_DIR / "item_to_idx.json").items()
    }

    checkpoint = torch.load(
        ARTIFACT_DIR / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Device: {device}")

    valid_result = evaluate(
        model,
        valid,
        train,
        user_to_idx,
        item_to_idx,
        device,
        args.batch_size,
    )
    test_result = evaluate(
        model,
        test,
        train,
        user_to_idx,
        item_to_idx,
        device,
        args.batch_size,
    )

    print_metrics("VALIDATION", valid_result)
    print_metrics("TEST", test_result)

    output = {
        "validation": valid_result,
        "test": test_result,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved metrics -> {ARTIFACT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
