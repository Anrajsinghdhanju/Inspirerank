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
from app.services.realtime_feedback import (
    EVENT_WEIGHTS,
    get_feedback_store,
)
from app.services.semantic_query_encoder import get_query_encoder


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data/recsys/arts_crafts_5core"
EMBED_DIR = ROOT / "artifacts/catalog_siglip"
LEARNED_DIR = ROOT / "artifacts/content_two_tower_v1"

ALPHA = 0.5
MIN_INTERACTIONS = 5
REALTIME_SEMANTIC_STRENGTH = 0.85
MAX_REALTIME_EVENTS = 20

# Search should remain primarily query-driven.
SEARCH_PERSONALIZATION_WEIGHT = 0.25
SEARCH_BEHAVIOR_WEIGHT = 0.10

# Maximal Marginal Relevance diversity.
MMR_LAMBDA = 0.82
MMR_POOL_MULTIPLIER = 6
MMR_MIN_POOL = 80


def _zscore(scores: torch.Tensor) -> torch.Tensor:
    return (
        scores - scores.mean()
    ) / scores.std().clamp_min(1e-6)


class HybridRecommender:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.train = pd.read_csv(DATA_DIR / "train.csv")
        self.items = pd.read_csv(DATA_DIR / "items.csv")

        self.train["user_id"] = self.train["user_id"].astype(str)
        self.train["parent_asin"] = self.train["parent_asin"].astype(str)
        self.items["parent_asin"] = self.items["parent_asin"].astype(str)

        self.item_ids = json.loads(
            (EMBED_DIR / "item_ids.json").read_text(
                encoding="utf-8"
            )
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

        self.text_matrix = self._load_float(
            "text_embeddings.npy"
        )
        self.text_matrix = F.normalize(
            self.text_matrix,
            p=2,
            dim=-1,
        )

        self.text_features = self._load_float(
            "text_embeddings.npy"
        )
        self.image_features = self._load_float(
            "image_embeddings.npy"
        )
        self.image_mask = torch.from_numpy(
            np.array(
                np.load(
                    EMBED_DIR / "image_mask.npy",
                    mmap_mode="r",
                ),
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

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.eval()

        self.max_history = checkpoint["max_history"]
        self.learned_catalog = self._encode_learned_catalog()

        self.histories, self.seen = self._build_histories()

        counts = Counter(
            self.train["parent_asin"].astype(str)
        )
        self.train_counts = torch.tensor(
            [
                counts.get(item_id, 0)
                for item_id in self.item_ids
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def has_user(self, user_id: str) -> bool:
        return user_id in self.histories

    def has_item(self, item_id: str) -> bool:
        return item_id in self.item_to_idx

    def _load_float(self, name: str) -> torch.Tensor:
        array = np.array(
            np.load(
                EMBED_DIR / name,
                mmap_mode="r",
            ),
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(array).to(self.device)

    def _build_histories(self):
        histories = defaultdict(list)
        seen = defaultdict(set)

        ordered = self.train.sort_values(
            ["user_id", "timestamp"]
        )

        for row in ordered.itertuples(index=False):
            user = str(row.user_id)
            idx = self.item_to_idx.get(
                str(row.parent_asin)
            )

            if idx is None:
                continue

            histories[user].append(idx)
            seen[user].add(idx)

        return dict(histories), dict(seen)

    @torch.inference_mode()
    def _encode_learned_catalog(self) -> torch.Tensor:
        chunks = []

        for start in range(
            0,
            len(self.text_features),
            512,
        ):
            end = min(
                start + 512,
                len(self.text_features),
            )
            chunks.append(
                self.model.encode_items(
                    self.text_features[start:end],
                    self.image_features[start:end],
                    self.image_mask[start:end],
                )
            )

        return torch.cat(chunks, dim=0)

    def _latest_feedback_per_item(
        self,
        user_id: str,
    ) -> list[dict]:
        events = get_feedback_store().recent_events(
            user_id,
            limit=MAX_REALTIME_EVENTS,
        )

        latest = []
        seen_items = set()

        for event in events:
            item_id = str(event.get("item_id", ""))

            if (
                not item_id
                or item_id in seen_items
                or item_id not in self.item_to_idx
            ):
                continue

            seen_items.add(item_id)
            latest.append(event)

        return latest

    def _realtime_profile(
        self,
        events: list[dict],
    ) -> torch.Tensor | None:
        weighted_vectors = []
        weights = []

        for event in events:
            item_id = str(event["item_id"])
            event_type = str(event["event_type"])

            idx = self.item_to_idx.get(item_id)
            weight = EVENT_WEIGHTS.get(event_type)

            if idx is None or weight is None:
                continue

            weighted_vectors.append(
                self.text_matrix[idx] * weight
            )
            weights.append(abs(weight))

        if not weighted_vectors:
            return None

        vector = (
            torch.stack(weighted_vectors, dim=0).sum(dim=0)
            / max(sum(weights), 1e-6)
        )

        if torch.linalg.vector_norm(vector) < 1e-8:
            return None

        return F.normalize(vector, p=2, dim=0)

    def _augmented_history(
        self,
        offline_history: list[int],
        events: list[dict],
    ) -> list[int]:
        positive_items = []

        for event in reversed(events):
            if event.get("event_type") not in {"like", "save"}:
                continue

            idx = self.item_to_idx.get(
                str(event.get("item_id"))
            )

            if idx is not None:
                positive_items.append(idx)

        combined = offline_history + positive_items
        return combined[-self.max_history :]

    @torch.inference_mode()
    def _user_profiles(
        self,
        user_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        offline_history = self.histories[user_id]
        recent_events = self._latest_feedback_per_item(user_id)

        semantic_history = torch.tensor(
            offline_history,
            dtype=torch.long,
            device=self.device,
        )

        long_term_profile = F.normalize(
            self.text_matrix[semantic_history].mean(dim=0),
            p=2,
            dim=0,
        )

        realtime_profile = self._realtime_profile(recent_events)

        if realtime_profile is not None:
            semantic_profile = F.normalize(
                long_term_profile
                + REALTIME_SEMANTIC_STRENGTH * realtime_profile,
                p=2,
                dim=0,
            )
        else:
            semantic_profile = long_term_profile

        learned_history = self._augmented_history(
            offline_history,
            recent_events,
        )
        learned_ids = torch.tensor(
            learned_history,
            dtype=torch.long,
            device=self.device,
        )

        history_vectors = self.model.encode_items(
            self.text_features[learned_ids],
            self.image_features[learned_ids],
            self.image_mask[learned_ids],
        ).unsqueeze(0)

        history_mask = torch.ones(
            (1, len(learned_history)),
            dtype=torch.bool,
            device=self.device,
        )

        learned_profile = self.model.encode_users(
            history_vectors,
            history_mask,
        )[0]

        return (
            semantic_profile,
            learned_profile,
            recent_events,
        )

    def _mmr_topk(
        self,
        scores: torch.Tensor,
        limit: int,
        lambda_value: float = MMR_LAMBDA,
    ) -> list[int]:
        """
        Maximal Marginal Relevance.

        First retrieve a relevance-heavy candidate pool, then greedily choose
        items balancing relevance and dissimilarity to already selected items.
        """
        pool_size = min(
            len(scores),
            max(
                MMR_MIN_POOL,
                limit * MMR_POOL_MULTIPLIER,
            ),
        )

        pool_scores, pool_indices = torch.topk(
            scores,
            k=pool_size,
        )

        # Convert relevance into 0..1 inside this candidate pool.
        finite = torch.isfinite(pool_scores)
        if not finite.any():
            return []

        valid_scores = pool_scores[finite]
        valid_indices = pool_indices[finite]

        min_score = valid_scores.min()
        max_score = valid_scores.max()
        relevance = (
            (valid_scores - min_score)
            / (max_score - min_score).clamp_min(1e-8)
        )

        candidate_vectors = self.text_matrix[valid_indices]

        selected_local: list[int] = []
        remaining = torch.ones(
            len(valid_indices),
            dtype=torch.bool,
            device=self.device,
        )

        while len(selected_local) < min(limit, len(valid_indices)):
            if not selected_local:
                best_local = int(torch.argmax(relevance).item())
            else:
                selected_vectors = candidate_vectors[
                    torch.tensor(
                        selected_local,
                        dtype=torch.long,
                        device=self.device,
                    )
                ]

                similarity = (
                    candidate_vectors
                    @ selected_vectors.T
                ).max(dim=1).values

                similarity = torch.clamp(
                    similarity,
                    min=0.0,
                    max=1.0,
                )

                mmr_score = (
                    lambda_value * relevance
                    - (1.0 - lambda_value) * similarity
                )

                mmr_score[~remaining] = -torch.inf
                best_local = int(
                    torch.argmax(mmr_score).item()
                )

            selected_local.append(best_local)
            remaining[best_local] = False

        return [
            int(valid_indices[idx].item())
            for idx in selected_local
        ]

    def _item_payload(
        self,
        idx: int,
        score: float,
    ) -> dict:
        item_id = self.item_ids[idx]
        meta = self.metadata.get(item_id, {})

        return {
            "item_id": item_id,
            "title": meta.get("title"),
            "image_url": meta.get("image_url"),
            "main_category": meta.get("main_category"),
            "price": meta.get("price"),
            "average_rating": meta.get("average_rating"),
            "score": round(float(score), 4),
            "interaction_support": int(
                self.train_counts[idx].item()
            ),
            "strategy": (
                "semantic+behavioral"
                if self.train_counts[idx].item()
                >= MIN_INTERACTIONS
                else "semantic"
            ),
        }

    @torch.inference_mode()
    def recommend(
        self,
        user_id: str,
        limit: int = 20,
    ) -> dict:
        if user_id not in self.histories:
            raise KeyError(user_id)

        (
            semantic_profile,
            learned_profile,
            recent_events,
        ) = self._user_profiles(user_id)

        semantic_scores = _zscore(
            self.text_matrix @ semantic_profile
        )
        learned_scores = _zscore(
            self.learned_catalog @ learned_profile
        )

        trust_mask = (
            self.train_counts >= MIN_INTERACTIONS
        ).to(semantic_scores.dtype)

        final_scores = (
            semantic_scores
            + ALPHA * learned_scores * trust_mask
        )

        for seen_idx in self.seen[user_id]:
            final_scores[seen_idx] = -torch.inf

        for event in recent_events:
            idx = self.item_to_idx.get(
                str(event.get("item_id"))
            )
            if idx is not None:
                final_scores[idx] = -torch.inf

        selected_indices = self._mmr_topk(
            final_scores,
            limit=limit,
        )

        recommendations = [
            self._item_payload(
                idx,
                float(final_scores[idx].item()),
            )
            for idx in selected_indices
        ]

        history_examples = []

        for idx in self.histories[user_id][-5:]:
            item_id = self.item_ids[idx]
            meta = self.metadata.get(item_id, {})
            history_examples.append(
                {
                    "item_id": item_id,
                    "title": meta.get("title"),
                    "image_url": meta.get("image_url"),
                }
            )

        feedback_summary = [
            {
                "item_id": str(event["item_id"]),
                "event_type": str(event["event_type"]),
            }
            for event in recent_events
        ]

        return {
            "user_id": user_id,
            "strategy": "hybrid_realtime_mmr_v1",
            "alpha": ALPHA,
            "min_interactions": MIN_INTERACTIONS,
            "diversity_strategy": "mmr",
            "mmr_lambda": MMR_LAMBDA,
            "history_count": len(self.histories[user_id]),
            "realtime_event_count": len(recent_events),
            "recent_feedback": feedback_summary,
            "history_examples": history_examples,
            "recommendations": recommendations,
        }

    @torch.inference_mode()
    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 24,
    ) -> dict:
        if user_id not in self.histories:
            raise KeyError(user_id)

        query_vector = get_query_encoder().encode(query)

        if query_vector.device != self.device:
            query_vector = query_vector.to(self.device)

        (
            semantic_profile,
            learned_profile,
            recent_events,
        ) = self._user_profiles(user_id)

        query_scores = _zscore(
            self.text_matrix @ query_vector
        )

        personalization_scores = _zscore(
            self.text_matrix @ semantic_profile
        )

        learned_scores = _zscore(
            self.learned_catalog @ learned_profile
        )

        trust_mask = (
            self.train_counts >= MIN_INTERACTIONS
        ).to(query_scores.dtype)

        final_scores = (
            query_scores
            + SEARCH_PERSONALIZATION_WEIGHT
            * personalization_scores
            + SEARCH_BEHAVIOR_WEIGHT
            * learned_scores
            * trust_mask
        )

        # Discovery search avoids repeating already-consumed items.
        for seen_idx in self.seen[user_id]:
            final_scores[seen_idx] = -torch.inf

        for event in recent_events:
            idx = self.item_to_idx.get(
                str(event.get("item_id"))
            )
            if idx is not None:
                final_scores[idx] = -torch.inf

        selected_indices = self._mmr_topk(
            final_scores,
            limit=limit,
        )

        results = [
            self._item_payload(
                idx,
                float(final_scores[idx].item()),
            )
            for idx in selected_indices
        ]

        return {
            "user_id": user_id,
            "query": query,
            "strategy": "siglip_personalized_search_mmr_v1",
            "query_weight": 1.0,
            "personalization_weight": SEARCH_PERSONALIZATION_WEIGHT,
            "behavior_weight": SEARCH_BEHAVIOR_WEIGHT,
            "diversity_strategy": "mmr",
            "mmr_lambda": MMR_LAMBDA,
            "realtime_event_count": len(recent_events),
            "results": results,
        }

    def demo_users(
        self,
        limit: int = 20,
    ) -> list[dict]:
        ranked = sorted(
            (
                (user_id, len(history))
                for user_id, history
                in self.histories.items()
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
