from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HistoryTwoTower(nn.Module):
    """
    History-aware two-tower retrieval model.

    User tower:
        sequence of historical item IDs
        -> history embedding lookup
        -> masked mean pooling
        -> small MLP
        -> normalized user vector

    Item tower:
        candidate item ID
        -> candidate embedding lookup
        -> small MLP
        -> normalized item vector
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.pad_idx = num_items

        self.history_embedding = nn.Embedding(
            num_items + 1,
            embedding_dim,
            padding_idx=self.pad_idx,
        )
        self.candidate_embedding = nn.Embedding(
            num_items,
            embedding_dim,
        )

        self.user_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.item_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        nn.init.normal_(self.history_embedding.weight, std=0.02)
        nn.init.normal_(self.candidate_embedding.weight, std=0.02)

        with torch.no_grad():
            self.history_embedding.weight[self.pad_idx].zero_()

    def encode_users(
        self,
        history_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.history_embedding(history_ids)

        mask = history_mask.unsqueeze(-1).to(embedded.dtype)
        summed = (embedded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / counts

        projected = self.user_projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.candidate_embedding(item_ids)
        projected = self.item_projection(embedded)
        return F.normalize(projected, p=2, dim=-1)
