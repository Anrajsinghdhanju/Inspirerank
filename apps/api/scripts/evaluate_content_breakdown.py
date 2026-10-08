from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DATA_DIR = Path("data/recsys/arts_crafts_5core")
EMBED_DIR = Path("artifacts/catalog_siglip")


def metric_row(rank, ks=(10, 20)):
    out = {}
    for k in ks:
        hit = rank <= k
        out[f"recall@{k}"] = 1.0 if hit else 0.0
        out[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
    out["mrr"] = 1.0 / rank
    return out


def load_matrix(mode):
    item_ids = json.loads((EMBED_DIR / "item_ids.json").read_text())
    item_to_idx = {x: i for i, x in enumerate(item_ids)}
    filename = "text_embeddings.npy" if mode == "text" else "multimodal_embeddings.npy"
    arr = np.array(np.load(EMBED_DIR / filename, mmap_mode="r"), dtype=np.float32, copy=True)
    return item_to_idx, F.normalize(torch.from_numpy(arr), p=2, dim=-1)


def histories_from_train(train, item_to_idx):
    histories, seen = defaultdict(list), defaultdict(set)
    for row in train.sort_values(["user_id", "timestamp"]).itertuples(index=False):
        idx = item_to_idx.get(str(row.parent_asin))
        if idx is not None:
            u = str(row.user_id)
            histories[u].append(idx)
            seen[u].add(idx)
    return dict(histories), dict(seen)


@torch.inference_mode()
def evaluate(split, histories, seen, item_to_idx, item_matrix, train_items):
    sums = {b: defaultdict(float) for b in ("overall", "warm", "cold")}
    counts = {b: 0 for b in ("overall", "warm", "cold")}

    for row in split.itertuples(index=False):
        user, target = str(row.user_id), str(row.parent_asin)
        if user not in histories or target not in item_to_idx:
            continue
        target_idx = item_to_idx[target]
        profile = F.normalize(item_matrix[torch.tensor(histories[user])].mean(dim=0), p=2, dim=0)
        scores = item_matrix @ profile
        for seen_idx in seen.get(user, set()):
            if seen_idx != target_idx:
                scores[seen_idx] = -torch.inf
        rank = int((scores > scores[target_idx]).sum().item()) + 1
        bucket = "warm" if target in train_items else "cold"
        vals = metric_row(rank)
        counts["overall"] += 1
        counts[bucket] += 1
        for k, v in vals.items():
            sums["overall"][k] += v
            sums[bucket][k] += v

    result = {}
    for bucket in ("overall", "warm", "cold"):
        n = counts[bucket]
        result[bucket] = {"targets": n}
        result[bucket].update({k: v / n if n else 0.0 for k, v in sums[bucket].items()})
    return result


def print_result(name, result):
    print(f"\n{name}\n" + "=" * 60)
    for bucket in ("overall", "warm", "cold"):
        vals = result[bucket]
        print(f"\n{bucket.upper()} ({vals['targets']:,} targets)")
        for k, v in vals.items():
            if k != "targets":
                print(f"{k:12} {v:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["text", "multimodal"], required=True)
    args = p.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv")
    valid = pd.read_csv(DATA_DIR / "valid.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    for frame in (train, valid, test):
        frame["user_id"] = frame["user_id"].astype(str)
        frame["parent_asin"] = frame["parent_asin"].astype(str)

    item_to_idx, item_matrix = load_matrix(args.mode)
    histories, seen = histories_from_train(train, item_to_idx)
    train_items = set(train["parent_asin"])

    print(f"Mode: {args.mode}")
    vr = evaluate(valid, histories, seen, item_to_idx, item_matrix, train_items)

    th = {u: list(x) for u, x in histories.items()}
    ts = {u: set(x) for u, x in seen.items()}
    for row in valid.itertuples(index=False):
        u, item = str(row.user_id), str(row.parent_asin)
        idx = item_to_idx.get(item)
        if u in th and idx is not None:
            th[u].append(idx)
            ts[u].add(idx)

    tr = evaluate(test, th, ts, item_to_idx, item_matrix, train_items)
    print_result("VALIDATION", vr)
    print_result("TEST", tr)


if __name__ == "__main__":
    main()
