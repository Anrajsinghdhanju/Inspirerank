# InspireRank Milestone 4 — ID-only Two-Tower Baseline

This is intentionally a baseline, not the final recommender.

The dataset has ~50% cold held-out items, so an item-ID model cannot represent
about half of validation/test targets. We keep that limitation visible and
measure it explicitly.

## 1. Apply this patch

Copy the patch contents over the existing project.

## 2. Train

From the project root:

```powershell
$env:PYTHONPATH="apps/api"
python apps/api/scripts/train_two_tower.py --epochs 12 --batch-size 512
```

If PowerShell already has PYTHONPATH configured, the first line is optional.

## 3. Evaluate

```powershell
$env:PYTHONPATH="apps/api"
python apps/api/scripts/evaluate_two_tower.py
```

This reports:

- overall Recall@10 / Recall@20
- overall NDCG@10 / NDCG@20
- MRR
- warm-item-only metrics
- cold-item rate
- the maximum possible overall recall for an ID-only item tower

## 4. Compare with a simple popularity baseline

```powershell
python apps/api/scripts/evaluate_popularity.py
```

## 5. Send back

Send:

1. the final few training losses,
2. the validation block,
3. the test block,
4. the popularity baseline.

Then we will build the hybrid multimodal item tower.
