# InspireRank Milestone 9 — Productize the Hybrid Recommender

Offline hybrid test result:

- Overall Recall@20: 0.0444
- Warm Recall@20: 0.0614
- Cold Recall@20: 0.0275

Selected on validation:
- alpha = 0.5
- min_interactions = 5

This milestone exposes that exact strategy through FastAPI and adds the first
Next.js frontend.

## 1. Commit Milestone 8

```powershell
git add .
git commit -m "milestone 8: add hybrid semantic and behavioral retrieval"
```

## 2. Apply this patch

Copy this patch over the existing repository.

## 3. Start FastAPI

From the repo root:

```powershell
$env:PYTHONPATH="apps/api"
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

The first feed request loads the model artifacts and catalog into memory.

Useful endpoints:

```text
GET /api/v1/demo-users
GET /api/v1/feed/{user_id}?limit=24
```

## 4. Test the API

Open:

```text
http://localhost:8000/api/v1/demo-users
```

Copy a user ID and then open:

```text
http://localhost:8000/api/v1/feed/PASTE_USER_ID?limit=12
```

## 5. Start the frontend

Open a second terminal:

```powershell
cd apps/web
copy .env.local.example .env.local
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

The site lets you choose a real user from the 5K-user cohort and renders a
Pinterest-style personalized visual feed from the hybrid recommender.

## Next milestone

After the UI works:

- add live user interactions (click/save/hide)
- persist events
- update a short-term Redis preference vector
- add semantic search to the website
- then build a ranking/reranking layer
