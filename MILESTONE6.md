# InspireRank Milestone 6 — Full Catalog SigLIP Embeddings

Catalog coverage is now excellent:

- 27,206 / 27,206 items have metadata
- 27,206 have text
- 27,192 have image URLs

The next step creates content vectors for the entire recommendation catalog.

## 1. Apply this patch

Copy the patch into the existing repository.

## 2. Smoke test first

```powershell
python apps/api/scripts/generate_catalog_embeddings.py --limit 100 --batch-size 8 --overwrite
```

This creates a small 100-item artifact. Because the catalog size is stored in
the checkpoint, the full run should use `--overwrite`.

## 3. Generate the full catalog

```powershell
python apps/api/scripts/generate_catalog_embeddings.py --batch-size 8 --overwrite
```

After the first batch, if the process is interrupted, resume with:

```powershell
python apps/api/scripts/generate_catalog_embeddings.py --batch-size 8
```

Do not use `--overwrite` when resuming.

Artifacts:

```text
artifacts/catalog_siglip/
    item_ids.json
    text_embeddings.npy
    image_embeddings.npy
    multimodal_embeddings.npy
    image_mask.npy
    progress.json
    metadata.json
```

`multimodal_embeddings.npy` uses normalized text + image fusion. If an image is
missing or cannot be downloaded, it falls back to the text embedding.

## 4. Run the zero-shot text content baseline

```powershell
python apps/api/scripts/evaluate_content_baseline.py --mode text
```

## 5. Run the zero-shot multimodal content baseline

```powershell
python apps/api/scripts/evaluate_content_baseline.py --mode multimodal
```

Unlike the ID-only models, this baseline can score items that never appeared in
the training interactions, because every catalog item gets a content vector.

## 6. Send back

Send:

1. the final catalog embedding summary,
2. text-only validation/test metrics,
3. multimodal validation/test metrics.

Then we will train the learned multimodal item tower.
