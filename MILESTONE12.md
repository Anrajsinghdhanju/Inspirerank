# InspireRank Milestone 12 — Two-Stage Adaptive Personalized Reranking

## Why

The first semantic-search implementation mixed query relevance and
personalization across the entire 27K-item catalog. For specific queries, the
query score dominated so strongly that different users often saw nearly the
same ranking.

Milestone 12 changes the search architecture.

## Stage 1 — semantic candidate retrieval

Pure query relevance retrieves the top 250 candidates:

```text
query -> SigLIP -> top 250 semantically relevant products
```

Personalization CANNOT introduce an irrelevant product outside this pool.

## Stage 2 — personalized reranking

Only those 250 candidates are reranked using candidate-local normalized scores:

```text
query relevance
long-term user taste
recent Redis preference
learned behavioral score
```

This makes user signals strong enough to meaningfully alter ranking while
preserving semantic relevance.

## Adaptive weighting

Broad queries get stronger personalization.

Approximate broad-query weights:

```text
query       55%
long-term   24%
realtime    13%
behavior     8%
```

Highly specific queries move toward:

```text
query       78%
long-term   11%
realtime     6%
behavior     5%
```

When no live Redis signal exists, that unused weight is redistributed.

Query specificity uses both:
- semantic score separation inside the candidate pool
- query length as a secondary signal

## MMR

After personalized reranking, MMR still provides diversity on the final page.

## Apply

Commit Milestone 11 first:

```powershell
git add .
git commit -m "milestone 11: add personalized semantic search and diversity reranking"
```

Then copy this patch over the repository.

## Restart

Backend:

```powershell
$env:PYTHONPATH="apps/api"
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd apps/web
npm run dev
```

## Test broad queries

These should show more user-to-user ranking differences:

```text
craft supplies
creative gift
DIY project
beginner art supplies
something colorful
weekend craft
```

Specific queries such as:

```text
origami paper with Japanese patterns
```

should remain more similar between users, because explicit query intent should
win.

The UI now displays:
- semantic candidate pool size
- estimated query specificity
- personalization strength

## Quantitative comparison

From the repository root:

```powershell
$env:PYTHONPATH="apps/api"

python apps/api/scripts/compare_search_personalization.py --query "craft supplies"
```

It runs the query for several demo users and reports:
- top five results per user
- pairwise overlap@20
- mean overlap@20

Compare it with:

```powershell
python apps/api/scripts/compare_search_personalization.py --query "origami paper with Japanese patterns"
```

Expected behavior:

```text
broad query -> lower user-to-user overlap
specific query -> higher user-to-user overlap
```

That is a much better search objective than random shuffling.

## Next

After validating this behavior:

- instrument latency for retrieval / reranking / MMR
- measure diversity and personalization quantitatively
- add OpenTelemetry / Prometheus-ready metrics
- production Docker build
- deployment
