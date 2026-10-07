from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """
    First retrieval baseline:
      user_id -> user embedding
      item_id -> item embedding

    The towers are intentionally simple. The next milestone will replace the
    item-ID tower with multimodal SigLIP features so unseen items can be scored.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_embedding(user_ids), p=2, dim=-1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_embedding(item_ids), p=2, dim=-1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_users(user_ids), self.encode_items(item_ids)
