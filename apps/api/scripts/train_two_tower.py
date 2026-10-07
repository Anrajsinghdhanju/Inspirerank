from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml_recsys.two_tower import TwoTowerModel


DATA_DIR = Path("data/recsys/arts_crafts_5core")
ARTIFACT_DIR = Path("artifacts/two_tower_v1")


class PairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
    ) -> None:
        self.users = torch.tensor(
            [user_to_idx[user] for user in frame["user_id"].astype(str)],
            dtype=torch.long,
        )
        self.items = torch.tensor(
            [item_to_idx[item] for item in frame["parent_asin"].astype(str)],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocabs(
    train: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    users = sorted(train["user_id"].astype(str).unique())
    items = sorted(train["parent_asin"].astype(str).unique())

    user_to_idx = {user: idx for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    return user_to_idx, item_to_idx


def in_batch_softmax_loss(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Each row's paired item is the positive; all other items in the batch act as
    negatives. This is a standard efficient retrieval-training pattern.
    """
    logits = (user_vectors @ item_vectors.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_DIR / "train.csv")
    train["user_id"] = train["user_id"].astype(str)
    train["parent_asin"] = train["parent_asin"].astype(str)

    user_to_idx, item_to_idx = build_vocabs(train)

    dataset = PairDataset(train, user_to_idx, item_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        embedding_dim=args.embedding_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Users: {len(user_to_idx):,}")
    print(f"Warm items: {len(item_to_idx):,}")
    print(f"Training pairs: {len(dataset):,}")
    print()

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        examples = 0

        for user_ids, item_ids in loader:
            user_ids = user_ids.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)

            user_vectors, item_vectors = model(user_ids, item_ids)
            loss = in_batch_softmax_loss(
                user_vectors,
                item_vectors,
                args.temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = user_ids.shape[0]
            epoch_loss += loss.item() * batch_size
            examples += batch_size

        average_loss = epoch_loss / examples
        history.append({"epoch": epoch, "loss": average_loss})
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={average_loss:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": len(user_to_idx),
            "num_items": len(item_to_idx),
            "embedding_dim": args.embedding_dim,
        },
        ARTIFACT_DIR / "model.pt",
    )

    (ARTIFACT_DIR / "user_to_idx.json").write_text(
        json.dumps(user_to_idx),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "item_to_idx.json").write_text(
        json.dumps(item_to_idx),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "train_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    config = vars(args) | {
        "num_users": len(user_to_idx),
        "num_items": len(item_to_idx),
        "train_pairs": len(dataset),
    }
    (ARTIFACT_DIR / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved model artifacts -> {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
