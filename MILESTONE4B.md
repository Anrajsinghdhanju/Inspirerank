# InspireRank Milestone 4B — History-aware Two-Tower

Why this exists:

The first ID-only model memorized `(user_id, item_id)` pairs and lost to the
popularity baseline. This version makes the user tower from the user's actual
chronological item history and uses sampled negatives that exclude the user's
known positive items.

## 1. Apply patch

Copy this patch over the existing repository.

## 2. Train

```powershell
$env:PYTHONPATH="apps/api"
python apps/api/scripts/train_history_two_tower.py --epochs 15 --batch-size 256 --negatives 64
```

CPU is okay for this experiment.

## 3. Evaluate

```powershell
python apps/api/scripts/evaluate_history_two_tower.py
```

Compare the output to:

- `artifacts/two_tower_v1/metrics.json`
- `artifacts/popularity_baseline/metrics.json`

## 4. Send back

Send the final 5 training losses and the validation/test metrics.

If this baseline is healthy, the next model will replace the item-ID tower with
SigLIP multimodal item features to attack the ~50% cold-item problem.
