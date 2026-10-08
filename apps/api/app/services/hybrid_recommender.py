from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.ml_recsys.content_two_tower import LearnedMultimodalTwoTower
from app.services.realtime_feedback import (
    EVENT_WEIGHTS,
    get_feedback_store,
)


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data/recsys/arts_crafts_5core"
EMBED_DIR = ROOT / "artifacts/catalog_siglip"
LEARNED_DIR = ROOT / "artifacts/content_two_tower_v1"

ALPHA = 0.5
MIN_INTERACTIONS = 5
REALTIME_SEMANTIC_STRENGTH = 0.85
MAX_REALTIME_EVENTS = 20


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

        # Redis returns newest first. Keep only the latest event
        # for an item so repeated actions do not compound forever.
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

        total_weight = max(sum(weights), 1e-6)

        vector = (
            torch.stack(weighted_vectors, dim=0).sum(dim=0)
            / total_weight
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

        # Redis events are newest-first. Reverse them so the
        # model sees recent positives in chronological order.
        for event in reversed(events):
            if event.get("event_type") not in {"like", "save"}:
                continue

            idx = self.item_to_idx.get(
                str(event.get("item_id"))
            )

            if idx is not None:
                positive_items.append(idx)

        combined = (
            offline_history + positive_items
        )

        return combined[-self.max_history :]

    @torch.inference_mode()
    def recommend(
        self,
        user_id: str,
        limit: int = 20,
    ) -> dict:
        if user_id not in self.histories:
            raise KeyError(user_id)

        offline_history = self.histories[user_id]
        recent_events = self._latest_feedback_per_item(
            user_id
        )

        semantic_history = torch.tensor(
            offline_history,
            dtype=torch.long,
            device=self.device,
        )

        long_term_profile = F.normalize(
            self.text_matrix[
                semantic_history
            ].mean(dim=0),
            p=2,
            dim=0,
        )

        realtime_profile = self._realtime_profile(
            recent_events
        )

        if realtime_profile is not None:
            text_profile = F.normalize(
                long_term_profile
                + REALTIME_SEMANTIC_STRENGTH
                * realtime_profile,
                p=2,
                dim=0,
            )
        else:
            text_profile = long_term_profile

        text_scores = _zscore(
            self.text_matrix @ text_profile
        )

        learned_history = self._augmented_history(
            offline_history,
            recent_events,
        )

        learned_history_tensor = torch.tensor(
            learned_history,
            dtype=torch.long,
            device=self.device,
        )

        history_vectors = self.model.encode_items(
            self.text_features[
                learned_history_tensor
            ],
            self.image_features[
                learned_history_tensor
            ],
            self.image_mask[
                learned_history_tensor
            ],
        ).unsqueeze(0)

        history_mask = torch.ones(
            (1, len(learned_history)),
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
            self.train_counts
            >= MIN_INTERACTIONS
        ).to(text_scores.dtype)

        final_scores = (
            text_scores
            + ALPHA
            * learned_scores
            * trust_mask
        )

        # Never repeat historical items.
        for seen_idx in self.seen[user_id]:
            final_scores[seen_idx] = -torch.inf

        # Also remove items the user just acted on so refreshing
        # visibly surfaces fresh recommendations.
        for event in recent_events:
            idx = self.item_to_idx.get(
                str(event.get("item_id"))
            )
            if idx is not None:
                final_scores[idx] = -torch.inf

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
                    "main_category": meta.get(
                        "main_category"
                    ),
                    "price": meta.get("price"),
                    "average_rating": meta.get(
                        "average_rating"
                    ),
                    "score": round(
                        float(score),
                        4,
                    ),
                    "interaction_support": int(
                        self.train_counts[
                            idx
                        ].item()
                    ),
                    "strategy": (
                        "semantic+behavioral"
                        if self.train_counts[
                            idx
                        ].item()
                        >= MIN_INTERACTIONS
                        else "semantic"
                    ),
                }
            )

        history_examples = []

        for idx in offline_history[-5:]:
            item_id = self.item_ids[idx]
            meta = self.metadata.get(
                item_id,
                {},
            )

            history_examples.append(
                {
                    "item_id": item_id,
                    "title": meta.get("title"),
                    "image_url": meta.get(
                        "image_url"
                    ),
                }
            )

        feedback_summary = [
            {
                "item_id": str(event["item_id"]),
                "event_type": str(
                    event["event_type"]
                ),
            }
            for event in recent_events
        ]

        return {
            "user_id": user_id,
            "strategy": (
                "hybrid_text_behavior_realtime_v1"
            ),
            "alpha": ALPHA,
            "min_interactions": MIN_INTERACTIONS,
            "history_count": len(
                self.histories[user_id]
            ),
            "realtime_event_count": len(
                recent_events
            ),
            "recent_feedback": feedback_summary,
            "history_examples": history_examples,
            "recommendations": recommendations,
        }

    def demo_users(
        self,
        limit: int = 20,
    ) -> list[dict]:
        ranked = sorted(
            (
                (
                    user_id,
                    len(history),
                )
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
            for user_id, count
            in ranked[:limit]
        ]


@lru_cache(maxsize=1)
def get_recommender() -> HybridRecommender:
    return HybridRecommender()
