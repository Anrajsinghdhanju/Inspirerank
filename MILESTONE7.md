# InspireRank Milestone 7 — Learned Multimodal Retrieval

## 1. Warm/cold breakdown for current zero-shot baselines

```powershell
$env:PYTHONPATH="apps/api"

python apps/api/scripts/evaluate_content_breakdown.py --mode text
python apps/api/scripts/evaluate_content_breakdown.py --mode multimodal
```

This also fixes the non-writable NumPy warning.

## 2. Train learned multimodal two-tower

```powershell
python apps/api/scripts/train_content_two_tower.py --epochs 10 --batch-size 128 --negatives 32
```

The model learns:
- text projection
- image projection
- text-vs-image fusion gate
- history-based user representation

Cold catalog items can still be scored because item vectors come from content.

## 3. Evaluate

```powershell
python apps/api/scripts/evaluate_content_two_tower.py
```

Send:
- text warm/cold breakdown
- multimodal warm/cold breakdown
- final training losses
- learned model validation/test metrics
