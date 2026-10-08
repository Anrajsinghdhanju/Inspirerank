from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class LearnedMultimodalTwoTower(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.text_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        self.item_output = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.user_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_items(self, text_features, image_features, image_mask):
        text_vec = F.normalize(self.text_projection(text_features), p=2, dim=-1)
        image_vec = F.normalize(self.image_projection(image_features), p=2, dim=-1)
        mask = image_mask.to(text_vec.dtype).unsqueeze(-1)
        text_weight = self.gate(torch.cat([text_vec, image_vec, mask], dim=-1))
        text_weight = torch.where(mask.bool(), text_weight, torch.ones_like(text_weight))
        fused = text_weight * text_vec + (1.0 - text_weight) * image_vec
        return F.normalize(self.item_output(fused), p=2, dim=-1)

    def encode_users(self, history_item_vectors, history_mask):
        mask = history_mask.to(history_item_vectors.dtype).unsqueeze(-1)
        pooled = (history_item_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.user_projection(pooled), p=2, dim=-1)
