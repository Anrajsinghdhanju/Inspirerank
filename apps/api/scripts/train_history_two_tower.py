from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml_recsys.history_two_tower import HistoryTwoTower


DATA_DIR = Path("data/recsys/arts_crafts_5core")
ARTIFACT_DIR = Path("artifacts/history_two_tower_v2")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_item_vocab(train: pd.DataFrame) -> dict[str, int]:
    items = sorted(train["parent_asin"].astype(str).unique())
    return {item: idx for idx, item in enumerate(items)}


def build_sequences(
    train: pd.DataFrame,
    item_to_idx: dict[str, int],
) -> tuple[dict[str, list[int]], dict[str, set[int]]]:
    sequences: dict[str, list[int]] = {}
    seen: dict[str, set[int]] = {}

    ordered = train.sort_values(["user_id", "timestamp"])

    for user_id, group in ordered.groupby("user_id", sort=False):
        ids = [
            item_to_idx[item]
            for item in group["parent_asin"].astype(str)
            if item in item_to_idx
        ]
        if len(ids) >= 2:
            user = str(user_id)
            sequences[user] = ids
            seen[user] = set(ids)

    return sequences, seen


class SequentialPairs(Dataset):
    def __init__(
        self,
        sequences: dict[str, list[int]],
        seen_by_user: dict[str, set[int]],
        num_items: int,
        max_history: int,
        negatives: int,
        seed: int,
    ) -> None:
        self.examples: list[tuple[str, list[int], int]] = []
        self.seen_by_user = seen_by_user
        self.num_items = num_items
        self.max_history = max_history
        self.negatives = negatives
        self.rng = random.Random(seed)

        for user_id, sequence in sequences.items():
            for target_position in range(1, len(sequence)):
                start = max(0, target_position - max_history)
                history = sequence[start:target_position]
                target = sequence[target_position]
                self.examples.append((user_id, history, target))

    def __len__(self) -> int:
        return len(self.examples)

    def _sample_negatives(self, user_id: str, target: int) -> list[int]:
        blocked = self.seen_by_user[user_id]
        result: list[int] = []

        while len(result) < self.negatives:
            candidate = self.rng.randrange(self.num_items)
            if candidate == target or candidate in blocked:
                continue
            result.append(candidate)

        return result

    def __getitem__(self, index: int):
        user_id, history, target = self.examples[index]
        negatives = self._sample_negatives(user_id, target)
        return history, target, negatives


def collate_batch(batch, pad_idx: int):
    max_len = max(len(history) for history, _, _ in batch)

    histories = []
    masks = []
    targets = []
    negatives = []

    for history, target, negative_ids in batch:
        pad_count = max_len - len(history)
        histories.append(history + [pad_idx] * pad_count)
        masks.append([1] * len(history) + [0] * pad_count)
        targets.append(target)
        negatives.append(negative_ids)

    return (
        torch.tensor(histories, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )


def sampled_softmax_loss(
    user_vectors: torch.Tensor,
    positive_vectors: torch.Tensor,
    negative_vectors: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    positive_scores = (user_vectors * positive_vectors).sum(dim=-1, keepdim=True)
    negative_scores = torch.einsum("bd,bkd->bk", user_vectors, negative_vectors)

    logits = torch.cat([positive_scores, negative_scores], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return torch.nn.functional.cross_entropy(logits, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=64)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    train["user_id"] = train["user_id"].astype(str)
    train["parent_asin"] = train["parent_asin"].astype(str)

    item_to_idx = build_item_vocab(train)
    sequences, seen_by_user = build_sequences(train, item_to_idx)

    dataset = SequentialPairs(
        sequences=sequences,
        seen_by_user=seen_by_user,
        num_items=len(item_to_idx),
        max_history=args.max_history,
        negatives=args.negatives,
        seed=args.seed,
    )

    model = HistoryTwoTower(
        num_items=len(item_to_idx),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda batch: collate_batch(batch, model.pad_idx),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warm items: {len(item_to_idx):,}")
    print(f"Users with sequences: {len(sequences):,}")
    print(f"Sequential training pairs: {len(dataset):,}")
    print(f"Negatives/example: {args.negatives}")
    print()

    history_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        examples_seen = 0

        for history_ids, history_mask, target_ids, negative_ids in loader:
            history_ids = history_ids.to(device, non_blocking=True)
            history_mask = history_mask.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            negative_ids = negative_ids.to(device, non_blocking=True)

            user_vectors = model.encode_users(history_ids, history_mask)
            positive_vectors = model.encode_items(target_ids)

            flat_negatives = negative_ids.reshape(-1)
            negative_vectors = model.encode_items(flat_negatives)
            negative_vectors = negative_vectors.reshape(
                negative_ids.shape[0],
                negative_ids.shape[1],
                -1,
            )

            loss = sampled_softmax_loss(
                user_vectors,
                positive_vectors,
                negative_vectors,
                args.temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = history_ids.shape[0]
            running_loss += loss.item() * batch_size
            examples_seen += batch_size

        avg_loss = running_loss / examples_seen
        history_log.append({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch:02d}/{args.epochs} loss={avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_items": len(item_to_idx),
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "max_history": args.max_history,
        },
        ARTIFACT_DIR / "model.pt",
    )

    (ARTIFACT_DIR / "item_to_idx.json").write_text(
        json.dumps(item_to_idx),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "train_history.json").write_text(
        json.dumps(history_log, indent=2),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved -> {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
