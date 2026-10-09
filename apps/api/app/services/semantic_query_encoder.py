from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from threading import Lock

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from app.core.config import settings
from app.core.observability import QUERY_CACHE_EVENTS


MODEL_NAME = "google/siglip-base-patch16-224"


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


class SemanticQueryEncoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

        self.cache_size = max(1, settings.query_cache_size)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_lock = Lock()
        self._model_lock = Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    def _cached(self, key: str) -> torch.Tensor | None:
        with self._cache_lock:
            vector = self._cache.get(key)
            if vector is None:
                return None

            self._cache.move_to_end(key)
            self.cache_hits += 1
            QUERY_CACHE_EVENTS.labels(result="hit").inc()
            return vector.to(self.device, non_blocking=True)

    def _store(self, key: str, vector: torch.Tensor) -> None:
        cpu_vector = vector.detach().to(device="cpu", dtype=torch.float32).contiguous()

        with self._cache_lock:
            self._cache[key] = cpu_vector
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    @torch.inference_mode()
    def encode(self, query: str) -> torch.Tensor:
        key = normalize_query(query)
        if not key:
            raise ValueError("Query must not be empty.")

        cached = self._cached(key)
        if cached is not None:
            return cached

        self.cache_misses += 1
        QUERY_CACHE_EVENTS.labels(result="miss").inc()

        with self._model_lock:
            cached = self._cached(key)
            if cached is not None:
                return cached

            inputs = self.processor(
                text=[key],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}

            output = self.model.get_text_features(**inputs)

            if isinstance(output, torch.Tensor):
                vector = output
            elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                vector = output.pooler_output
            else:
                raise RuntimeError("Unexpected SigLIP text feature output.")

            vector = F.normalize(vector.float(), p=2, dim=-1)[0]
            self._store(key, vector)
            return vector

    def cache_info(self) -> dict:
        with self._cache_lock:
            size = len(self._cache)

        total = self.cache_hits + self.cache_misses
        return {
            "size": size,
            "max_size": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }


@lru_cache(maxsize=1)
def get_query_encoder() -> SemanticQueryEncoder:
    return SemanticQueryEncoder()
