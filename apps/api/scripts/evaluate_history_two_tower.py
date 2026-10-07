from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from ml_recsys.history_two_tower import HistoryTwoTower


DATA_DIR = Path("data/recsys/arts_crafts_5core")
ARTIFACT_DIR = Path("artifacts/history_two_tower_v2")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_train_histories(
    train: pd.DataFrame,
    item_to_idx: dict[str, int],
) -> dict[str, list[int]]:
    histories: dict[str, list[int]] = {}

    ordered = train.sort_values(["user_id", "timestamp"])
    for user_id, group in ordered.groupby("user_id", sort=False):
        histories[str(user_id)] = [
            item_to_idx[item]
            for item in group["parent_asin"].astype(str)
            if item in item_to_idx
        ]

    return histories


def pad_histories(
    histories: list[list[int]],
    pad_idx: int,
    max_history: int,
    device: torch.device,
):
    trimmed = [history[-max_history:] for history in histories]
    max_len = max(len(history) for history in trimmed)

    ids = []
    masks = []

    for history in trimmed:
        pad_count = max_len - len(history)
        ids.append(history + [pad_idx] * pad_count)
        masks.append([1] * len(history) + [0] * pad_count)

    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def metrics_from_rank(rank: int | None, ks=(10, 20)):
    result = {}
    for k in ks:
        hit = rank is not None and rank <= k
        result[f"recall@{k}"] = 1.0 if hit else 0.0
        result[f"ndcg@{k}"] = (
            1.0 / math.log2(rank + 1)
            if hit and rank is not None
            else 0.0
        )
    result["mrr"] = 1.0 / rank if rank is not None else 0.0
    return result


@torch.inference_mode()
def evaluate_split(
    model: HistoryTwoTower,
    split: pd.DataFrame,
    histories: dict[str, list[int]],
    item_to_idx: dict[str, int],
    device: torch.device,
    batch_size: int,
    max_history: int,
    update_history_after: bool,
) -> dict:
    model.eval()

    all_item_ids = torch.arange(len(item_to_idx), device=device)
    item_matrix = model.encode_items(all_item_ids)

    totals = defaultdict(float)
    warm_totals = defaultdict(float)
    total = 0
    warm = 0
    cold = 0

    working_histories = {user: list(items) for user, items in histories.items()}

    records = split.sort_values(["user_id", "timestamp"]).copy()
    records["user_id"] = records["user_id"].astype(str)
    records["parent_asin"] = records["parent_asin"].astype(str)

    for start in range(0, len(records), batch_size):
        batch = records.iloc[start : start + batch_size]

        usable_rows = []
        usable_histories = []

        for row in batch.itertuples(index=False):
            user_id = str(row.user_id)
            if user_id not in working_histories or not working_histories[user_id]:
                continue

            usable_rows.append((user_id, str(row.parent_asin)))
            usable_histories.append(working_histories[user_id])

        if not usable_rows:
            continue

        history_ids, history_mask = pad_histories(
            usable_histories,
            model.pad_idx,
            max_history,
            device,
        )
        user_vectors = model.encode_users(history_ids, history_mask)
        scores = user_vectors @ item_matrix.T

        for row_idx, (user_id, target_item) in enumerate(usable_rows):
            total += 1
            target_idx = item_to_idx.get(target_item)

            if target_idx is None:
                cold += 1
                metrics = metrics_from_rank(None)
                for key, value in metrics.items():
                    totals[key] += value
            else:
                warm += 1
                row_scores = scores[row_idx]

                for seen_idx in set(working_histories[user_id]):
                    if seen_idx != target_idx:
                        row_scores[seen_idx] = -torch.inf

                target_score = row_scores[target_idx]
                rank = int((row_scores > target_score).sum().item()) + 1

                metrics = metrics_from_rank(rank)
                for key, value in metrics.items():
                    totals[key] += value
                    warm_totals[key] += value

            if update_history_after and target_idx is not None:
                working_histories[user_id].append(target_idx)

    return {
        "targets": total,
        "warm_targets": warm,
        "cold_targets": cold,
        "warm_coverage": warm / total if total else 0.0,
        "overall": {
            key: value / total if total else 0.0
            for key, value in totals.items()
        },
        "warm_only": {
            key: value / warm if warm else 0.0
            for key, value in warm_totals.items()
        },
    }


def print_result(name: str, result: dict) -> None:
    print(f"\n{name}")
    print("=" * 60)
    print(f"targets:       {result['targets']:,}")
    print(f"warm targets:  {result['warm_targets']:,}")
    print(f"cold targets:  {result['cold_targets']:,}")
    print(f"warm coverage:  {result['warm_coverage']:.2%}")

    print("\nOverall")
    for key, value in result["overall"].items():
        print(f"{key:12} {value:.4f}")

    print("\nWarm only")
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

    model = HistoryTwoTower(
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    histories = build_train_histories(train, item_to_idx)

    print(f"Device: {device}")

    valid_result = evaluate_split(
        model=model,
        split=valid,
        histories=histories,
        item_to_idx=item_to_idx,
        device=device,
        batch_size=args.batch_size,
        max_history=checkpoint["max_history"],
        update_history_after=False,
    )

    # For test-time evaluation, the validation interaction is part of the past.
    test_histories = {user: list(items) for user, items in histories.items()}
    for row in valid.itertuples(index=False):
        user_id = str(row.user_id)
        target_idx = item_to_idx.get(str(row.parent_asin))
        if user_id in test_histories and target_idx is not None:
            test_histories[user_id].append(target_idx)

    test_result = evaluate_split(
        model=model,
        split=test,
        histories=test_histories,
        item_to_idx=item_to_idx,
        device=device,
        batch_size=args.batch_size,
        max_history=checkpoint["max_history"],
        update_history_after=False,
    )

    print_result("VALIDATION", valid_result)
    print_result("TEST", test_result)

    output = {
        "validation": valid_result,
        "test": test_result,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved -> {ARTIFACT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
