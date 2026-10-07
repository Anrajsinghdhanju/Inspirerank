# InspireRank Milestone 5 — Build the multimodal catalog

The collaborative baselines underperform popularity and about half of held-out
targets are cold items. The next model therefore needs item content.

This step joins the exact 27K recommendation catalog to the official Amazon
Arts, Crafts & Sewing metadata.

## 1. Apply this patch

Copy the patch contents into the current repository.

## 2. Smoke test on 100 IDs

```powershell
python apps/api/scripts/prepare_catalog_metadata.py --limit 100
```

You should see metadata/text/image coverage statistics.

## 3. Build metadata for the full recommendation catalog

Run again without the limit:

```powershell
python apps/api/scripts/prepare_catalog_metadata.py
```

This writes:

```text
data/recsys/arts_crafts_5core/items.csv
data/recsys/arts_crafts_5core/missing_metadata_items.txt
```

It does NOT download all product images. It only records image URLs for the
exact items used by our recommendation experiment.

## 4. Analyze the resulting catalog

```powershell
python apps/api/scripts/analyze_catalog_metadata.py
```

Send the printed coverage output.

## What comes next

After we verify metadata coverage:

1. Generate SigLIP text/image embeddings for this exact catalog.
2. Evaluate a zero-shot content baseline.
3. Train a multimodal item tower.
4. Compare warm-item and cold-item Recall/NDCG against:
   - popularity
   - ID-only two tower
   - history two tower
