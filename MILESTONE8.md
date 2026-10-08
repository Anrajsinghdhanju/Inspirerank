# InspireRank Milestone 8 — Hybrid Retrieval

We now have two complementary systems:

- zero-shot SigLIP content retrieval: strong overall and cold-start performance
- learned two-tower retrieval: much stronger on warm items but weak on cold items

The hybrid keeps semantic retrieval for every catalog item and adds the learned
signal only to candidates with enough interaction support.

## Run

```powershell
$env:PYTHONPATH="apps/api"
python apps/api/scripts/evaluate_hybrid_retrieval.py --base-mode text
```

Default validation grid:

- alpha: 0.25, 0.5, 1.0
- minimum training interactions: 1, 3, 5

The best configuration is selected ONLY using validation NDCG@20. The script
then evaluates that single configuration on the test set.

Optional comparison:

```powershell
python apps/api/scripts/evaluate_hybrid_retrieval.py --base-mode multimodal
```

## Send back

Send:
- validation grid
- selected alpha/min_count
- test overall, warm, and cold metrics
