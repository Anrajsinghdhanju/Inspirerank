from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ml_recsys.content_two_tower import LearnedMultimodalTwoTower

DATA_DIR = Path("data/recsys/arts_crafts_5core")
EMBED_DIR = Path("artifacts/catalog_siglip")
OUT = Path("artifacts/content_two_tower_v1")


def load_features(device):
    item_ids = json.loads((EMBED_DIR / "item_ids.json").read_text())
    item_to_idx = {x: i for i, x in enumerate(item_ids)}
    def load(name, dtype=np.float32):
        return torch.from_numpy(np.array(np.load(EMBED_DIR / name, mmap_mode="r"), dtype=dtype, copy=True)).to(device)
    return item_to_idx, load("text_embeddings.npy"), load("image_embeddings.npy"), load("image_mask.npy", bool)


def histories(train, item_to_idx):
    h, s = defaultdict(list), defaultdict(set)
    for row in train.sort_values(["user_id", "timestamp"]).itertuples(index=False):
        idx = item_to_idx.get(str(row.parent_asin))
        if idx is not None:
            u = str(row.user_id)
            h[u].append(idx); s[u].add(idx)
    return dict(h), dict(s)


def metric_row(rank):
    out = {}
    for k in (10, 20):
        hit = rank <= k
        out[f"recall@{k}"] = 1.0 if hit else 0.0
        out[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
    out["mrr"] = 1.0 / rank
    return out


@torch.inference_mode()
def encode_catalog(model, text, image, mask, batch_size):
    chunks = []
    for start in range(0, len(text), batch_size):
        end = min(start + batch_size, len(text))
        chunks.append(model.encode_items(text[start:end], image[start:end], mask[start:end]))
    return torch.cat(chunks)


@torch.inference_mode()
def evaluate(model, split, hist, seen, item_to_idx, catalog, text, image, mask,
             train_items, device, max_history, batch_size):
    sums = {b: defaultdict(float) for b in ("overall", "warm", "cold")}
    counts = {b: 0 for b in ("overall", "warm", "cold")}
    rows = [(str(r.user_id), str(r.parent_asin)) for r in split.itertuples(index=False)
            if str(r.user_id) in hist and str(r.parent_asin) in item_to_idx]

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        hs = [hist[u][-max_history:] for u, _ in batch]
        max_len = max(len(x) for x in hs)
        ids, masks = [], []
        for x in hs:
            pad = max_len - len(x)
            ids.append(x + [0] * pad)
            masks.append([1] * len(x) + [0] * pad)
        ids = torch.tensor(ids, dtype=torch.long, device=device)
        masks = torch.tensor(masks, dtype=torch.bool, device=device)

        b, l = ids.shape
        hv = model.encode_items(text[ids.reshape(-1)], image[ids.reshape(-1)], mask[ids.reshape(-1)]).reshape(b, l, -1)
        uv = model.encode_users(hv, masks)
        scores = uv @ catalog.T

        for i, (u, target) in enumerate(batch):
            tidx = item_to_idx[target]
            for sidx in seen.get(u, set()):
                if sidx != tidx:
                    scores[i, sidx] = -torch.inf
            rank = int((scores[i] > scores[i, tidx]).sum().item()) + 1
            bucket = "warm" if target in train_items else "cold"
            vals = metric_row(rank)
            counts["overall"] += 1; counts[bucket] += 1
            for k, v in vals.items():
                sums["overall"][k] += v; sums[bucket][k] += v

    result = {}
    for bucket in ("overall", "warm", "cold"):
        n = counts[bucket]
        result[bucket] = {"targets": n}
        result[bucket].update({k: v/n if n else 0.0 for k, v in sums[bucket].items()})
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
    p.add_argument("--catalog-batch-size", type=int, default=512)
    p.add_argument("--user-batch-size", type=int, default=64)
    args = p.parse_args()

    train, valid, test = [pd.read_csv(DATA_DIR / f"{s}.csv") for s in ("train", "valid", "test")]
    for f in (train, valid, test):
        f["user_id"] = f["user_id"].astype(str)
        f["parent_asin"] = f["parent_asin"].astype(str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_to_idx, text, image, mask = load_features(device)
    ckpt = torch.load(OUT / "model.pt", map_location=device, weights_only=True)
    model = LearnedMultimodalTwoTower(ckpt["input_dim"], ckpt["embedding_dim"], ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Device: {device}")
    print("Encoding learned catalog vectors...")
    catalog = encode_catalog(model, text, image, mask, args.catalog_batch_size)

    h, s = histories(train, item_to_idx)
    train_items = set(train["parent_asin"])
    vr = evaluate(model, valid, h, s, item_to_idx, catalog, text, image, mask,
                  train_items, device, ckpt["max_history"], args.user_batch_size)

    th, ts = {u:list(x) for u,x in h.items()}, {u:set(x) for u,x in s.items()}
    for row in valid.itertuples(index=False):
        u, target = str(row.user_id), str(row.parent_asin)
        idx = item_to_idx.get(target)
        if u in th and idx is not None:
            th[u].append(idx); ts[u].add(idx)

    tr = evaluate(model, test, th, ts, item_to_idx, catalog, text, image, mask,
                  train_items, device, ckpt["max_history"], args.user_batch_size)

    print_result("VALIDATION", vr)
    print_result("TEST", tr)
    (OUT / "metrics.json").write_text(json.dumps({"validation":vr, "test":tr}, indent=2))


if __name__ == "__main__":
    main()
