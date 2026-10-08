from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DATA_DIR = Path("data/recsys/arts_crafts_5core")
EMBED_DIR = Path("artifacts/catalog_siglip")
OUTPUT_DIR = Path("artifacts/content_baseline")


def load_item_map() -> tuple[list[str], dict[str, int]]:
    item_ids = json.loads(
        (EMBED_DIR / "item_ids.json").read_text(encoding="utf-8")
    )
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    return item_ids, item_to_idx


def build_histories(
    train: pd.DataFrame,
    item_to_idx: dict[str, int],
) -> tuple[dict[str, list[int]], dict[str, set[int]]]:
    histories: dict[str, list[int]] = defaultdict(list)
    seen: dict[str, set[int]] = defaultdict(set)

    ordered = train.sort_values(["user_id", "timestamp"])

    for row in ordered.itertuples(index=False):
        user_id = str(row.user_id)
        item_idx = item_to_idx.get(str(row.parent_asin))
        if item_idx is None:
            continue

        histories[user_id].append(item_idx)
        seen[user_id].add(item_idx)

    return dict(histories), dict(seen)


def metrics_from_rank(rank: int | None, ks=(10, 20)) -> dict[str, float]:
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
    split: pd.DataFrame,
    histories: dict[str, list[int]],
    seen_by_user: dict[str, set[int]],
    item_to_idx: dict[str, int],
    item_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict:
    total = 0
    totals = defaultdict(float)

    # Because content vectors exist for all catalog items, there is no
    # train-item cold-start barrier here.
    records = split[["user_id", "parent_asin"]].copy()
    records["user_id"] = records["user_id"].astype(str)
    records["parent_asin"] = records["parent_asin"].astype(str)

    usable = [
        (row.user_id, row.parent_asin)
        for row in records.itertuples(index=False)
        if row.user_id in histories and row.parent_asin in item_to_idx
    ]

    for start in range(0, len(usable), batch_size):
        batch = usable[start : start + batch_size]

        profile_vectors = []

        for user_id, _ in batch:
            history_indices = histories[user_id]
            history_tensor = torch.tensor(
                history_indices,
                dtype=torch.long,
                device=device,
            )
            profile = item_matrix[history_tensor].mean(dim=0)
            profile = F.normalize(profile, p=2, dim=0)
            profile_vectors.append(profile)

        user_matrix = torch.stack(profile_vectors, dim=0)
        scores = user_matrix @ item_matrix.T

        for row_idx, (user_id, target_item) in enumerate(batch):
            total += 1
            target_idx = item_to_idx[target_item]
            row_scores = scores[row_idx]

            for seen_idx in seen_by_user.get(user_id, set()):
                if seen_idx != target_idx:
                    row_scores[seen_idx] = -torch.inf

            target_score = row_scores[target_idx]
            rank = int((row_scores > target_score).sum().item()) + 1

            for key, value in metrics_from_rank(rank).items():
                totals[key] += value

    return {
        "targets": total,
        "coverage": total / len(split) if len(split) else 0.0,
        "metrics": {
            key: value / total if total else 0.0
            for key, value in totals.items()
        },
    }


def print_result(name: str, result: dict) -> None:
    print(f"\n{name}")
    print("=" * 55)
    print(f"targets evaluated: {result['targets']:,}")
    print(f"catalog coverage:  {result['coverage']:.2%}")

    for key, value in result["metrics"].items():
        print(f"{key:12} {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["text", "multimodal"],
        default="multimodal",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "valid.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    for frame in (train, valid, test):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["parent_asin"] = frame["parent_asin"].astype(str)

    _, item_to_idx = load_item_map()
    histories, seen_by_user = build_histories(train, item_to_idx)

    embedding_file = {
        "text": "text_embeddings.npy",
        "multimodal": "multimodal_embeddings.npy",
    }[args.mode]

    vectors = np.load(
        EMBED_DIR / embedding_file,
        mmap_mode="r",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_matrix = torch.from_numpy(np.asarray(vectors)).to(
        device=device,
        dtype=torch.float32,
    )
    item_matrix = F.normalize(item_matrix, p=2, dim=-1)

    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Catalog candidates: {item_matrix.shape[0]:,}")

    valid_result = evaluate_split(
        valid,
        histories,
        seen_by_user,
        item_to_idx,
        item_matrix,
        device,
        args.batch_size,
    )

    # At test time, validation interaction is now part of user history.
    test_histories = {
        user: list(history)
        for user, history in histories.items()
    }
    test_seen = {
        user: set(items)
        for user, items in seen_by_user.items()
    }

    for row in valid.itertuples(index=False):
        user_id = str(row.user_id)
        item_idx = item_to_idx.get(str(row.parent_asin))
        if item_idx is None or user_id not in test_histories:
            continue
        test_histories[user_id].append(item_idx)
        test_seen[user_id].add(item_idx)

    test_result = evaluate_split(
        test,
        test_histories,
        test_seen,
        item_to_idx,
        item_matrix,
        device,
        args.batch_size,
    )

    print_result("VALIDATION", valid_result)
    print_result("TEST", test_result)

    output = {
        "mode": args.mode,
        "validation": valid_result,
        "test": test_result,
    }

    output_path = OUTPUT_DIR / f"{args.mode}_metrics.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
