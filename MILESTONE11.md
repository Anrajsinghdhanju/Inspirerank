# InspireRank Milestone 11 — Personalized Semantic Search + Diversity

This milestone adds a real discovery/search layer.

## Search architecture

```text
query text
   |
   v
SigLIP text encoder
   |
   +---- semantic relevance
   |
   +---- personalized semantic profile
   |
   +---- learned behavioral score
   |
   v
candidate ranking
   |
   v
MMR diversity reranking
   |
   v
personalized search results
```

Search weights:

```text
query semantic relevance     1.00
personalization              0.25
learned behavioral signal    0.10
```

The query remains dominant so personalization does not hijack explicit intent.

## Diversity

Both home-feed recommendations and search results use Maximal Marginal
Relevance (MMR).

Default:

```text
MMR lambda = 0.82
```

Higher lambda favors relevance. Lower lambda favors diversity.

We first retrieve a larger relevance pool and then greedily choose results that
remain relevant while avoiding near-duplicate semantic vectors.

## 1. Commit Milestone 10

```powershell
git add .
git commit -m "milestone 10: add real-time Redis personalization"
```

## 2. Apply this patch

Copy it over the repository.

No new Python packages are required; Transformers/PyTorch already exist.

## 3. Restart API

```powershell
$env:PYTHONPATH="apps/api"
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

The first semantic search may take longer because the SigLIP text encoder is
loaded lazily on the first query.

## 4. Restart frontend

```powershell
cd apps/web
npm run dev
```

## 5. Try searches

Examples:

```text
watercolor supplies for beginners
minimalist jewelry making tools
pastel scrapbook decorations
knitting gifts
origami paper with Japanese patterns
```

## 6. API test

```text
GET /api/v1/discovery/search?user_id=USER_ID&q=watercolor%20supplies&limit=12
```

## Expected behavior

- natural-language semantic matching instead of literal keyword-only matching
- same query can rank somewhat differently for different users
- likes/saves influence subsequent search ranking
- previously consumed/acted-on items are excluded
- MMR keeps the first page from becoming a wall of near-identical products

## Next milestone

After verifying search:

- benchmark retrieval/search latency
- measure diversity quantitatively
- add observability and request timing
- production Docker setup
- deploy frontend + API
