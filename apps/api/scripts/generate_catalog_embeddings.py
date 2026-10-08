from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor


DATA_DIR = Path("data/recsys/arts_crafts_5core")
DEFAULT_OUTPUT_DIR = Path("artifacts/catalog_siglip")
MODEL_NAME = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768


def pooled_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    raise RuntimeError("Unexpected SigLIP feature output.")


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), p=2, dim=-1)


def safe_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def build_text(row) -> str:
    text = safe_text(row.search_text).strip()
    if text:
        return text[:4000]

    fallback = " ".join(
        part
        for part in (
            safe_text(row.title),
            safe_text(row.main_category),
            safe_text(row.description),
            safe_text(row.features),
        )
        if part
    ).strip()

    return fallback[:4000] or "arts and crafts product"


def fetch_image(url: str, timeout: float = 15.0) -> Image.Image | None:
    if not url:
        return None

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "InspireRank/0.6 educational-project"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        return image.convert("RGB")
    except (
        httpx.HTTPError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return None


def fetch_images(urls: list[str], workers: int) -> list[Image.Image | None]:
    results: list[Image.Image | None] = [None] * len(urls)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_image, url): idx
            for idx, url in enumerate(urls)
            if url
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None

    return results


def ensure_memmaps(
    output_dir: Path,
    n_items: int,
    overwrite: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "w+" if overwrite else "r+"

    paths = {
        "text": output_dir / "text_embeddings.npy",
        "image": output_dir / "image_embeddings.npy",
        "multimodal": output_dir / "multimodal_embeddings.npy",
        "image_mask": output_dir / "image_mask.npy",
    }

    if overwrite or not all(path.exists() for path in paths.values()):
        text = np.lib.format.open_memmap(
            paths["text"],
            mode="w+",
            dtype="float32",
            shape=(n_items, EMBEDDING_DIM),
        )
        image = np.lib.format.open_memmap(
            paths["image"],
            mode="w+",
            dtype="float32",
            shape=(n_items, EMBEDDING_DIM),
        )
        multimodal = np.lib.format.open_memmap(
            paths["multimodal"],
            mode="w+",
            dtype="float32",
            shape=(n_items, EMBEDDING_DIM),
        )
        image_mask = np.lib.format.open_memmap(
            paths["image_mask"],
            mode="w+",
            dtype="bool",
            shape=(n_items,),
        )
        return text, image, multimodal, image_mask

    text = np.load(paths["text"], mmap_mode=mode)
    image = np.load(paths["image"], mmap_mode=mode)
    multimodal = np.load(paths["multimodal"], mmap_mode=mode)
    image_mask = np.load(paths["image_mask"], mmap_mode=mode)

    expected = (n_items, EMBEDDING_DIM)
    for name, array in (
        ("text", text),
        ("image", image),
        ("multimodal", multimodal),
    ):
        if array.shape != expected:
            raise RuntimeError(
                f"{name} embedding shape is {array.shape}, expected {expected}. "
                "Use --overwrite to rebuild."
            )

    if image_mask.shape != (n_items,):
        raise RuntimeError("image_mask shape mismatch. Use --overwrite to rebuild.")

    return text, image, multimodal, image_mask


@torch.inference_mode()
def encode_text_batch(
    model,
    processor,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output = model.get_text_features(**inputs)
    vectors = normalize(pooled_tensor(output))
    return vectors.cpu().numpy().astype("float32")


@torch.inference_mode()
def encode_image_batch(
    model,
    processor,
    images: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(
        images=images,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output = model.get_image_features(**inputs)
    vectors = normalize(pooled_tensor(output))
    return vectors.cpu().numpy().astype("float32")


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    items = pd.read_csv(DATA_DIR / "items.csv")
    items["parent_asin"] = items["parent_asin"].astype(str)

    if args.limit is not None:
        items = items.head(args.limit).copy()

    items = items.reset_index(drop=True)
    n_items = len(items)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    item_ids = items["parent_asin"].tolist()
    (args.output_dir / "item_ids.json").write_text(
        json.dumps(item_ids),
        encoding="utf-8",
    )

    progress_path = args.output_dir / "progress.json"

    if args.overwrite and progress_path.exists():
        progress_path.unlink()

    next_index = 0
    if progress_path.exists() and not args.overwrite:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("model_name") != MODEL_NAME:
            raise RuntimeError("Stored progress uses a different model.")
        if progress.get("n_items") != n_items:
            raise RuntimeError(
                "Stored progress has a different catalog size. "
                "Use --overwrite."
            )
        next_index = int(progress.get("next_index", 0))

    text_mm, image_mm, multimodal_mm, image_mask_mm = ensure_memmaps(
        args.output_dir,
        n_items,
        args.overwrite or next_index == 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Model: {MODEL_NAME}")
    print(f"Catalog items: {n_items:,}")
    print(f"Resume index: {next_index:,}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    failed_images = int((~np.asarray(image_mask_mm[:next_index], dtype=bool)).sum())

    for start in range(next_index, n_items, args.batch_size):
        end = min(start + args.batch_size, n_items)
        batch = items.iloc[start:end]

        texts = [build_text(row) for row in batch.itertuples(index=False)]
        text_vectors = encode_text_batch(model, processor, texts, device)

        urls = [
            safe_text(row.image_url).strip()
            for row in batch.itertuples(index=False)
        ]
        downloaded = fetch_images(urls, args.download_workers)

        batch_image_vectors = np.zeros(
            (len(batch), EMBEDDING_DIM),
            dtype="float32",
        )
        batch_image_mask = np.zeros(len(batch), dtype=bool)

        valid_positions = [
            idx for idx, image in enumerate(downloaded) if image is not None
        ]

        if valid_positions:
            valid_images = [downloaded[idx] for idx in valid_positions]
            valid_vectors = encode_image_batch(
                model,
                processor,
                valid_images,
                device,
            )

            for position, vector in zip(valid_positions, valid_vectors, strict=True):
                batch_image_vectors[position] = vector
                batch_image_mask[position] = True

        failed_images += int((~batch_image_mask).sum())

        # Text-only fallback for missing/failed images.
        fused = text_vectors.copy()

        if batch_image_mask.any():
            image_plus_text = (
                text_vectors[batch_image_mask]
                + batch_image_vectors[batch_image_mask]
            )
            fused[batch_image_mask] = row_normalize(image_plus_text)

        text_mm[start:end] = text_vectors
        image_mm[start:end] = batch_image_vectors
        multimodal_mm[start:end] = fused
        image_mask_mm[start:end] = batch_image_mask

        text_mm.flush()
        image_mm.flush()
        multimodal_mm.flush()
        image_mask_mm.flush()

        progress = {
            "model_name": MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "n_items": n_items,
            "next_index": end,
        }
        progress_path.write_text(
            json.dumps(progress, indent=2),
            encoding="utf-8",
        )

        print(
            f"Embedded {end:,}/{n_items:,} | "
            f"batch images={int(batch_image_mask.sum())}/{len(batch)} | "
            f"cumulative missing/failed images={failed_images:,}"
        )

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "n_items": n_items,
        "fusion": "normalize(text + image); text fallback when image unavailable",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nCatalog embedding generation complete.")
    print(f"Saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
