from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


MODEL_NAME = "google/siglip-base-patch16-224"


class SemanticQueryEncoder:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, query: str) -> torch.Tensor:
        inputs = self.processor(
            text=[query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        output = self.model.get_text_features(**inputs)

        if isinstance(output, torch.Tensor):
            vector = output
        elif (
            hasattr(output, "pooler_output")
            and output.pooler_output is not None
        ):
            vector = output.pooler_output
        else:
            raise RuntimeError(
                "Unexpected SigLIP text feature output."
            )

        return F.normalize(
            vector.float(),
            p=2,
            dim=-1,
        )[0]


@lru_cache(maxsize=1)
def get_query_encoder() -> SemanticQueryEncoder:
    return SemanticQueryEncoder()
