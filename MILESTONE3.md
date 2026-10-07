# InspireRank — Milestone 3

Goal: turn semantic retrieval into actual user-personalized recommendations.

## 1. Embed the rest of the current catalog

The previous 25-item run was only a smoke test.

```powershell
python apps/api/scripts/generate_embeddings.py --limit 2000 --batch-size 8
```

Already embedded items are skipped automatically.

## 2. Install/update dependencies

```powershell
python -m pip install -e "./apps/api[dev,data,ml]"
```

## 3. Inspect recommendation-data coverage

```powershell
python apps/api/scripts/analyze_interactions.py
```

The important numbers are:

- interactions_on_embedded_items
- users_with_2_positive_embedded
- users_with_3_positive_embedded
- users_with_5_positive_embedded

## 4. If user histories are too sparse, ingest a larger interaction slice

This is safe to rerun because interactions have a uniqueness constraint.

```powershell
python apps/api/scripts/ingest_amazon.py --max-items 2000 --max-reviews 50000
```

Then run:

```powershell
python apps/api/scripts/analyze_interactions.py
```

again.

## 5. Restart the API

```powershell
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

## 6. Find a user with enough history

Open:

```text
http://localhost:8000/api/v1/users/eligible?min_interactions=3&limit=10
```

Copy one `user_id`.

## 7. Get personalized recommendations

Open:

```text
http://localhost:8000/api/v1/recommendations/PASTE_USER_ID_HERE?limit=12
```

The endpoint:

1. loads the user's positively rated historical items,
2. creates a recency/rating-weighted multimodal user vector,
3. excludes already-seen items,
4. retrieves nearest unseen items from pgvector.

This is our content-based baseline. The future two-tower model must beat it in offline evaluation.
