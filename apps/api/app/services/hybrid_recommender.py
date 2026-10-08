from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml_recsys.content_two_tower import LearnedMultimodalTwoTower


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data/recsys/arts_crafts_5core"
EMBED_DIR = ROOT / "artifacts/catalog_siglip"
LEARNED_DIR = ROOT / "artifacts/content_two_tower_v1"

ALPHA = 0.5
MIN_INTERACTIONS = 5


def _zscore(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean()) / scores.std().clamp_min(1e-6)


class HybridRecommender:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train = pd.read_csv(DATA_DIR / "train.csv")
        self.items = pd.read_csv(DATA_DIR / "items.csv")

        self.train["user_id"] = self.train["user_id"].astype(str)
        self.train["parent_asin"] = self.train["parent_asin"].astype(str)
        self.items["parent_asin"] = self.items["parent_asin"].astype(str)

        self.item_ids = json.loads(
            (EMBED_DIR / "item_ids.json").read_text(encoding="utf-8")
        )
        self.item_to_idx = {
            item_id: idx
            for idx, item_id in enumerate(self.item_ids)
        }

        self.metadata = (
            self.items.set_index("parent_asin")
            .replace({np.nan: None})
            .to_dict(orient="index")
        )

        self.text_matrix = self._load_float("text_embeddings.npy")
        self.text_matrix = F.normalize(self.text_matrix, p=2, dim=-1)

        self.text_features = self._load_float("text_embeddings.npy")
        self.image_features = self._load_float("image_embeddings.npy")
        self.image_mask = torch.from_numpy(
            np.array(
                np.load(EMBED_DIR / "image_mask.npy", mmap_mode="r"),
                dtype=bool,
                copy=True,
            )
        ).to(self.device)

        checkpoint = torch.load(
            LEARNED_DIR / "model.pt",
            map_location=self.device,
            weights_only=True,
        )

        self.model = LearnedMultimodalTwoTower(
            input_dim=checkpoint["input_dim"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.max_history = checkpoint["max_history"]
        self.learned_catalog = self._encode_learned_catalog()

        self.histories, self.seen = self._build_histories()

        counts = Counter(self.train["parent_asin"])
        self.train_counts = torch.tensor(
            [counts.get(item_id, 0) for item_id in self.item_ids],
            dtype=torch.float32,
            device=self.device,
        )

    def _load_float(self, name: str) -> torch.Tensor:
        array = np.array(
            np.load(EMBED_DIR / name, mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(array).to(self.device)

    def _build_histories(self):
        histories = defaultdict(list)
        seen = defaultdict(set)

        ordered = self.train.sort_values(["user_id", "timestamp"])

        for row in ordered.itertuples(index=False):
            user = str(row.user_id)
            idx = self.item_to_idx.get(str(row.parent_asin))
            if idx is None:
                continue
            histories[user].append(idx)
            seen[user].add(idx)

        return dict(histories), dict(seen)

    @torch.inference_mode()
    def _encode_learned_catalog(self) -> torch.Tensor:
        chunks = []

        for start in range(0, len(self.text_features), 512):
            end = min(start + 512, len(self.text_features))
            chunks.append(
                self.model.encode_items(
                    self.text_features[start:end],
                    self.image_features[start:end],
                    self.image_mask[start:end],
                )
            )

        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def recommend(self, user_id: str, limit: int = 20) -> dict:
        if user_id not in self.histories:
            raise KeyError(user_id)

        history = self.histories[user_id][-self.max_history :]

        history_tensor = torch.tensor(
            history,
            dtype=torch.long,
            device=self.device,
        )

        # Zero-shot semantic profile.
        text_profile = F.normalize(
            self.text_matrix[history_tensor].mean(dim=0),
            p=2,
            dim=0,
        )
        text_scores = _zscore(self.text_matrix @ text_profile)

        # Learned history profile.
        history_vectors = self.model.encode_items(
            self.text_features[history_tensor],
            self.image_features[history_tensor],
            self.image_mask[history_tensor],
        ).unsqueeze(0)

        history_mask = torch.ones(
            (1, len(history)),
            dtype=torch.bool,
            device=self.device,
        )

        user_vector = self.model.encode_users(
            history_vectors,
            history_mask,
        )[0]

        learned_scores = _zscore(
            self.learned_catalog @ user_vector
        )

        trust_mask = (
            self.train_counts >= MIN_INTERACTIONS
        ).to(text_scores.dtype)

        final_scores = (
            text_scores
            + ALPHA * learned_scores * trust_mask
        )

        for seen_idx in self.seen[user_id]:
            final_scores[seen_idx] = -torch.inf

        top_scores, top_indices = torch.topk(
            final_scores,
            k=min(limit, len(final_scores)),
        )

        recommendations = []

        for score, idx in zip(
            top_scores.cpu().tolist(),
            top_indices.cpu().tolist(),
            strict=True,
        ):
            item_id = self.item_ids[idx]
            meta = self.metadata.get(item_id, {})

            recommendations.append(
                {
                    "item_id": item_id,
                    "title": meta.get("title"),
                    "image_url": meta.get("image_url"),
                    "main_category": meta.get("main_category"),
                    "price": meta.get("price"),
                    "average_rating": meta.get("average_rating"),
                    "score": round(float(score), 4),
                    "interaction_support": int(self.train_counts[idx].item()),
                    "strategy": (
                        "semantic+behavioral"
                        if self.train_counts[idx].item() >= MIN_INTERACTIONS
                        else "semantic"
                    ),
                }
            )

        history_examples = []

        for idx in history[-5:]:
            item_id = self.item_ids[idx]
            meta = self.metadata.get(item_id, {})
            history_examples.append(
                {
                    "item_id": item_id,
                    "title": meta.get("title"),
                    "image_url": meta.get("image_url"),
                }
            )

        return {
            "user_id": user_id,
            "strategy": "hybrid_text_plus_behavior_v1",
            "alpha": ALPHA,
            "min_interactions": MIN_INTERACTIONS,
            "history_count": len(self.histories[user_id]),
            "history_examples": history_examples,
            "recommendations": recommendations,
        }

    def demo_users(self, limit: int = 20) -> list[dict]:
        ranked = sorted(
            (
                (user_id, len(history))
                for user_id, history in self.histories.items()
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return [
            {
                "user_id": user_id,
                "history_count": count,
            }
            for user_id, count in ranked[:limit]
        ]


@lru_cache(maxsize=1)
def get_recommender() -> HybridRecommender:
    return HybridRecommender()
