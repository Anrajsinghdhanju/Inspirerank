from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ml_recsys.content_two_tower import LearnedMultimodalTwoTower

DATA_DIR = Path("data/recsys/arts_crafts_5core")
EMBED_DIR = Path("artifacts/catalog_siglip")
OUT = Path("artifacts/content_two_tower_v1")


def load_features():
    item_ids = json.loads((EMBED_DIR / "item_ids.json").read_text())
    item_to_idx = {x: i for i, x in enumerate(item_ids)}
    def load(name, dtype=np.float32):
        return torch.from_numpy(np.array(np.load(EMBED_DIR / name, mmap_mode="r"), dtype=dtype, copy=True))
    return item_to_idx, load("text_embeddings.npy"), load("image_embeddings.npy"), load("image_mask.npy", bool)


def build_sequences(train, item_to_idx):
    seqs, seen = {}, {}
    for u, g in train.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False):
        ids = [item_to_idx[x] for x in g["parent_asin"].astype(str) if x in item_to_idx]
        if len(ids) >= 2:
            seqs[str(u)] = ids
            seen[str(u)] = set(ids)
    return seqs, seen


class SeqDataset(Dataset):
    def __init__(self, seqs, seen, num_items, max_history, negatives, seed):
        self.examples, self.seen = [], seen
        self.num_items, self.max_history, self.negatives = num_items, max_history, negatives
        self.rng = random.Random(seed)
        for u, seq in seqs.items():
            for pos in range(1, len(seq)):
                self.examples.append((u, seq[max(0, pos-max_history):pos], seq[pos]))
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        u, hist, target = self.examples[idx]
        negs = []
        while len(negs) < self.negatives:
            c = self.rng.randrange(self.num_items)
            if c != target and c not in self.seen[u]:
                negs.append(c)
        return hist, target, negs


def collate(batch):
    max_len = max(len(h) for h, _, _ in batch)
    histories, masks, targets, negs = [], [], [], []
    for h, t, n in batch:
        pad = max_len - len(h)
        histories.append(h + [0] * pad)
        masks.append([1] * len(h) + [0] * pad)
        targets.append(t)
        negs.append(n)
    return (torch.tensor(histories), torch.tensor(masks, dtype=torch.bool),
            torch.tensor(targets), torch.tensor(negs))


def loss_fn(user_vec, pos_vec, neg_vec, temperature):
    pos = (user_vec * pos_vec).sum(-1, keepdim=True)
    neg = torch.einsum("bd,bkd->bk", user_vec, neg_vec)
    logits = torch.cat([pos, neg], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--negatives", type=int, default=32)
    p.add_argument("--max-history", type=int, default=20)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    item_to_idx, text, image, mask = load_features()
    train = pd.read_csv(DATA_DIR / "train.csv")
    train["user_id"] = train["user_id"].astype(str)
    train["parent_asin"] = train["parent_asin"].astype(str)
    seqs, seen = build_sequences(train, item_to_idx)
    ds = SeqDataset(seqs, seen, len(item_to_idx), args.max_history, args.negatives, 42)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text, image, mask = text.to(device), image.to(device), mask.to(device)
    model = LearnedMultimodalTwoTower(768, args.embedding_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    print(f"Device: {device}")
    print(f"Catalog items: {len(item_to_idx):,}")
    print(f"Sequential pairs: {len(ds):,}")
    print(f"Negatives/example: {args.negatives}\n")

    history_log = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for hist_ids, hist_mask, targets, negatives in dl:
            hist_ids, hist_mask = hist_ids.to(device), hist_mask.to(device)
            targets, negatives = targets.to(device), negatives.to(device)

            b, l = hist_ids.shape
            hist_vec = model.encode_items(
                text[hist_ids.reshape(-1)],
                image[hist_ids.reshape(-1)],
                mask[hist_ids.reshape(-1)],
            ).reshape(b, l, -1)
            user_vec = model.encode_users(hist_vec, hist_mask)

            pos_vec = model.encode_items(text[targets], image[targets], mask[targets])
            flat_neg = negatives.reshape(-1)
            neg_vec = model.encode_items(text[flat_neg], image[flat_neg], mask[flat_neg])
            neg_vec = neg_vec.reshape(negatives.shape[0], negatives.shape[1], -1)

            loss = loss_fn(user_vec, pos_vec, neg_vec, args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item() * b
            n += b

        avg = total / n
        history_log.append({"epoch": epoch, "loss": avg})
        print(f"Epoch {epoch:02d}/{args.epochs} loss={avg:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": 768,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "max_history": args.max_history,
    }, OUT / "model.pt")
    (OUT / "train_history.json").write_text(json.dumps(history_log, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
