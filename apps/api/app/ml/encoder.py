from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768


class SiglipEncoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(tensor.float(), p=2, dim=-1)

    @staticmethod
    def _pooled_features(model_output) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output

        if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
            return model_output.pooler_output

        raise RuntimeError(
            "Unexpected SigLIP feature output. Check the installed transformers version."
        )

    @torch.inference_mode()
    def encode_items(
        self,
        images: list[Image.Image],
        texts: list[str],
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        if len(images) != len(texts):
            raise ValueError("images and texts must have the same batch size")

        inputs = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(**inputs)

        image_embeddings = self._normalize(outputs.image_embeds)
        text_embeddings = self._normalize(outputs.text_embeds)
        multimodal_embeddings = self._normalize(image_embeddings + text_embeddings)

        for name, tensor in (
            ("image", image_embeddings),
            ("text", text_embeddings),
            ("multimodal", multimodal_embeddings),
        ):
            if tensor.shape[-1] != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIM}-D {name} embeddings, "
                    f"received {tensor.shape[-1]}."
                )

        return (
            image_embeddings.cpu().tolist(),
            text_embeddings.cpu().tolist(),
            multimodal_embeddings.cpu().tolist(),
        )

    @torch.inference_mode()
    def encode_query(self, query: str) -> list[float]:
        inputs = self.processor(
            text=[query],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        output = self.model.get_text_features(**inputs)
        pooled = self._pooled_features(output)
        embedding = self._normalize(pooled)

        if embedding.shape[-1] != EMBEDDING_DIM:
            raise RuntimeError(
                f"Expected {EMBEDDING_DIM}-D query embedding, "
                f"received {embedding.shape[-1]}."
            )

        return embedding[0].cpu().tolist()


@lru_cache(maxsize=1)
def get_encoder() -> SiglipEncoder:
    return SiglipEncoder()
