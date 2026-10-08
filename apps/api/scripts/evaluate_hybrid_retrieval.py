from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml_recsys.content_two_tower import LearnedMultimodalTwoTower


DATA_DIR = Path("data/recsys/arts_crafts_5core")
EMBED_DIR = Path("artifacts/catalog_siglip")
LEARNED_DIR = Path("artifacts/content_two_tower_v1")
OUTPUT_DIR = Path("artifacts/hybrid_retrieval")


def metric_row(rank: int, ks=(10, 20)) -> dict[str, float]:
    out = {}
    for k in ks:
        hit = rank <= k
        out[f"recall@{k}"] = 1.0 if hit else 0.0
        out[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
    out["mrr"] = 1.0 / rank
    return out


def zscore_rows(scores: torch.Tensor) -> torch.Tensor:
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (scores - mean) / std


def load_float(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.array(
        np.load(path, mmap_mode="r"),
        dtype=np.float32,
        copy=True,
    )
    return torch.from_numpy(arr).to(device)


def load_catalog(device: torch.device, base_mode: str):
    item_ids = json.loads(
        (EMBED_DIR / "item_ids.json").read_text(encoding="utf-8")
    )
    item_to_idx = {item: idx for idx, item in enumerate(item_ids)}

    base_filename = {
        "text": "text_embeddings.npy",
        "multimodal": "multimodal_embeddings.npy",
    }[base_mode]

    base_matrix = F.normalize(
        load_float(EMBED_DIR / base_filename, device),
        p=2,
        dim=-1,
    )
    text_features = load_float(EMBED_DIR / "text_embeddings.npy", device)
    image_features = load_float(EMBED_DIR / "image_embeddings.npy", device)

    image_mask = torch.from_numpy(
        np.array(
            np.load(EMBED_DIR / "image_mask.npy", mmap_mode="r"),
            dtype=bool,
            copy=True,
        )
    ).to(device)

    return (
        item_ids,
        item_to_idx,
        base_matrix,
        text_features,
        image_features,
        image_mask,
    )


def load_model(device: torch.device):
    checkpoint = torch.load(
        LEARNED_DIR / "model.pt",
        map_location=device,
        weights_only=True,
    )

    model = LearnedMultimodalTwoTower(
        input_dim=checkpoint["input_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def build_histories(train: pd.DataFrame, item_to_idx: dict[str, int]):
    histories = defaultdict(list)
    seen = defaultdict(set)

    ordered = train.sort_values(["user_id", "timestamp"])

    for row in ordered.itertuples(index=False):
        user = str(row.user_id)
        idx = item_to_idx.get(str(row.parent_asin))
        if idx is None:
            continue
        histories[user].append(idx)
        seen[user].add(idx)

    return dict(histories), dict(seen)


def pad_histories(histories, max_history, device):
    trimmed = [h[-max_history:] for h in histories]
    max_len = max(len(h) for h in trimmed)

    ids, masks = [], []
    for h in trimmed:
        pad = max_len - len(h)
        ids.append(h + [0] * pad)
        masks.append([1] * len(h) + [0] * pad)

    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


@torch.inference_mode()
def encode_learned_catalog(
    model,
    text_features,
    image_features,
    image_mask,
    batch_size,
):
    chunks = []
    for start in range(0, len(text_features), batch_size):
        end = min(start + batch_size, len(text_features))
        chunks.append(
            model.encode_items(
                text_features[start:end],
                image_features[start:end],
                image_mask[start:end],
            )
        )
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def score_batch(
    batch,
    histories,
    seen,
    base_matrix,
    learned_catalog,
    model,
    text_features,
    image_features,
    image_mask,
    max_history,
    device,
):
    batch_histories = [histories[user] for user, _ in batch]

    base_profiles = []
    for h in batch_histories:
        idx = torch.tensor(h, dtype=torch.long, device=device)
        profile = F.normalize(base_matrix[idx].mean(dim=0), p=2, dim=0)
        base_profiles.append(profile)

    base_users = torch.stack(base_profiles, dim=0)
    base_scores = zscore_rows(base_users @ base_matrix.T)

    history_ids, history_mask = pad_histories(
        batch_histories,
        max_history,
        device,
    )
    b, l = history_ids.shape
    flat = history_ids.reshape(-1)

    history_vectors = model.encode_items(
        text_features[flat],
        image_features[flat],
        image_mask[flat],
    ).reshape(b, l, -1)

    learned_users = model.encode_users(
        history_vectors,
        history_mask,
    )
    learned_scores = zscore_rows(learned_users @ learned_catalog.T)

    for row_idx, (user, target_idx) in enumerate(batch):
        for seen_idx in seen.get(user, set()):
            if seen_idx != target_idx:
                base_scores[row_idx, seen_idx] = -torch.inf
                learned_scores[row_idx, seen_idx] = -torch.inf

    return base_scores, learned_scores


def evaluate_config(
    split,
    histories,
    seen,
    item_to_idx,
    train_items,
    train_counts_tensor,
    base_matrix,
    learned_catalog,
    model,
    text_features,
    image_features,
    image_mask,
    max_history,
    device,
    batch_size,
    alpha,
    min_interactions,
):
    totals = {
        "overall": defaultdict(float),
        "warm": defaultdict(float),
        "cold": defaultdict(float),
    }
    counts = {"overall": 0, "warm": 0, "cold": 0}

    raw_rows = [
        (str(row.user_id), str(row.parent_asin))
        for row in split.itertuples(index=False)
        if str(row.user_id) in histories
        and str(row.parent_asin) in item_to_idx
    ]

    trust_mask = (
        train_counts_tensor >= min_interactions
    ).to(base_matrix.dtype)

    for start in range(0, len(raw_rows), batch_size):
        raw_batch = raw_rows[start:start + batch_size]
        batch = [
            (user, item_to_idx[target])
            for user, target in raw_batch
        ]

        base_scores, learned_scores = score_batch(
            batch,
            histories,
            seen,
            base_matrix,
            learned_catalog,
            model,
            text_features,
            image_features,
            image_mask,
            max_history,
            device,
        )

        # Learned component only contributes to sufficiently supported items.
        learned_bonus = learned_scores * trust_mask.unsqueeze(0)
        final_scores = base_scores + alpha * learned_bonus

        for row_idx, ((user, target_item), (_, target_idx)) in enumerate(
            zip(raw_batch, batch, strict=True)
        ):
            target_score = final_scores[row_idx, target_idx]
            rank = int(
                (final_scores[row_idx] > target_score).sum().item()
            ) + 1

            bucket = "warm" if target_item in train_items else "cold"
            vals = metric_row(rank)

            counts["overall"] += 1
            counts[bucket] += 1

            for key, value in vals.items():
                totals["overall"][key] += value
                totals[bucket][key] += value

    result = {}

    for bucket in ("overall", "warm", "cold"):
        n = counts[bucket]
        result[bucket] = {"targets": n}
        result[bucket].update(
            {
                key: value / n if n else 0.0
                for key, value in totals[bucket].items()
            }
        )

    return result


def print_result(name, result):
    print(f"\n{name}")
    print("=" * 60)

    for bucket in ("overall", "warm", "cold"):
        vals = result[bucket]
        print(f"\n{bucket.upper()} ({vals['targets']:,} targets)")
        for key, value in vals.items():
            if key != "targets":
                print(f"{key:12} {value:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-mode",
        choices=["text", "multimodal"],
        default="text",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--catalog-batch-size", type=int, default=512)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "valid.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    for frame in (train, valid, test):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["parent_asin"] = frame["parent_asin"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        item_ids,
        item_to_idx,
        base_matrix,
        text_features,
        image_features,
        image_mask,
    ) = load_catalog(device, args.base_mode)

    model, checkpoint = load_model(device)

    print(f"Device: {device}")
    print(f"Base mode: {args.base_mode}")
    print("Encoding learned catalog vectors...")

    learned_catalog = encode_learned_catalog(
        model,
        text_features,
        image_features,
        image_mask,
        args.catalog_batch_size,
    )

    histories, seen = build_histories(train, item_to_idx)
    train_items = set(train["parent_asin"].astype(str))

    counts = Counter(train["parent_asin"].astype(str))
    train_counts_tensor = torch.tensor(
        [counts.get(item_id, 0) for item_id in item_ids],
        dtype=torch.float32,
        device=device,
    )

    print("\nValidation grid")
    print("=" * 60)

    best = None
    grid = []

    for alpha in args.alphas:
        for min_count in args.min_interactions:
            result = evaluate_config(
                valid,
                histories,
                seen,
                item_to_idx,
                train_items,
                train_counts_tensor,
                base_matrix,
                learned_catalog,
                model,
                text_features,
                image_features,
                image_mask,
                checkpoint["max_history"],
                device,
                args.batch_size,
                alpha,
                min_count,
            )

            score = result["overall"]["ndcg@20"]

            row = {
                "alpha": alpha,
                "min_interactions": min_count,
                "validation": result,
                "objective_ndcg20": score,
            }
            grid.append(row)

            print(
                f"alpha={alpha:<4} "
                f"min_count={min_count:<2} "
                f"R@20={result['overall']['recall@20']:.4f} "
                f"NDCG@20={result['overall']['ndcg@20']:.4f}"
            )

            if best is None or score > best["objective_ndcg20"]:
                best = row

    print(
        "\nBest validation setting: "
        f"alpha={best['alpha']}, "
        f"min_interactions={best['min_interactions']}"
    )
    print_result("BEST VALIDATION", best["validation"])

    test_histories = {u: list(h) for u, h in histories.items()}
    test_seen = {u: set(s) for u, s in seen.items()}

    for row in valid.itertuples(index=False):
        user = str(row.user_id)
        idx = item_to_idx.get(str(row.parent_asin))
        if user in test_histories and idx is not None:
            test_histories[user].append(idx)
            test_seen[user].add(idx)

    test_result = evaluate_config(
        test,
        test_histories,
        test_seen,
        item_to_idx,
        train_items,
        train_counts_tensor,
        base_matrix,
        learned_catalog,
        model,
        text_features,
        image_features,
        image_mask,
        checkpoint["max_history"],
        device,
        args.batch_size,
        best["alpha"],
        best["min_interactions"],
    )

    print_result("TEST", test_result)

    output = {
        "base_mode": args.base_mode,
        "best_alpha": best["alpha"],
        "best_min_interactions": best["min_interactions"],
        "validation": best["validation"],
        "test": test_result,
        "validation_grid": grid,
    }

    output_path = OUTPUT_DIR / f"{args.base_mode}_metrics.json"
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
