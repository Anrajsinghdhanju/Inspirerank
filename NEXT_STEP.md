# InspireRank dense recommender dataset

1. Install pandas:

```powershell
python -m pip install pandas
```

2. Prepare a reproducible 5,000-user subset:

```powershell
python apps/api/scripts/prepare_recsys_dataset.py --users 5000 --seed 42
```

3. Analyze it:

```powershell
python apps/api/scripts/analyze_recsys_dataset.py
```

Send the output from step 3 before starting the two-tower model.
