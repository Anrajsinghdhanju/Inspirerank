# InspireRank Milestone 14 — Evaluation, Quality Filtering, and Latency

This milestone stops subjective "looks good" evaluation and adds measurable
system-quality signals.

## 1. Catalog quality filtering

Before retrieval, obvious broken metadata is removed using conservative rules:

- missing/very short titles
- titles with almost no alphabetic content
- suspicious long consonant-only tokens
- malformed-looking records

Every catalog item also receives a soft quality score based on:

- title completeness
- descriptive metadata
- image availability
- category availability

Quality is only a small ranking tie-breaker. Relevance still dominates.

## 2. Latency instrumentation

Search responses now return:

```json
{
  "timings_ms": {
    "query_encode": 0,
    "user_profiles": 0,
    "candidate_retrieval": 0,
    "personalized_rerank": 0,
    "mmr": 0,
    "total": 0
  }
}
```

FastAPI also emits:

```text
X-InspireRank-Process-Time-Ms
```

on every HTTP response.

## 3. Catalog quality audit

Commit Milestone 13:

```powershell
git add .
git commit -m "milestone 13: add query-conditioned personalization"
```

Apply this patch and restart FastAPI.

Then run:

```powershell
$env:PYTHONPATH="apps/api"
python apps/api/scripts/audit_catalog_quality.py
```

Review:
- filtered item count
- reason counts
- sample rejected records

There is also an API endpoint:

```text
GET /api/v1/diagnostics/catalog-quality
```

## 4. Full search-system evaluation

Run:

```powershell
python apps/api/scripts/evaluate_search_system.py
```

Default suite:

```text
craft supplies
creative gift
DIY project
beginner art supplies
watercolor supplies for beginners
origami paper with Japanese patterns
```

Across four demo users it reports:

### Personalization

```text
overlap@5
overlap@10
overlap@20
rank displacement@10
```

Lower overlap for broad queries is expected.
Higher overlap for precise queries is expected.

### Relevance

```text
mean semantic cosine similarity
```

This verifies reranking has not destroyed query relevance.

### Diversity

```text
1 - mean pairwise cosine similarity
```

Higher means the returned set is less repetitive.

### Latency

```text
mean
p50
p95
stage-level means
```

The evaluator warms up the lazy SigLIP query encoder first, so model-loading
time does not pollute normal request latency.

Results are saved to:

```text
artifacts/evaluation/search_system_metrics.json
```

## 5. Optional single-query evaluation

```powershell
python apps/api/scripts/evaluate_search_system.py --query "craft supplies"
```

Multiple queries:

```powershell
python apps/api/scripts/evaluate_search_system.py ^
  --query "craft supplies" ^
  --query "origami paper with Japanese patterns"
```

In PowerShell you may prefer keeping the command on one line.

## 6. Frontend

Restart Next.js. Search now shows:

- candidate count
- query specificity
- personalization %
- low-quality items filtered
- total measured search latency

Cards also expose a small quality score for debugging.

## What success looks like

We are not optimizing for the lowest possible overlap.

We want:

```text
broad query
  -> meaningful personalization differences
  -> good semantic relevance
  -> useful diversity

specific query
  -> stronger user overlap
  -> excellent semantic relevance

all queries
  -> no obvious broken catalog records
  -> measured, explainable latency
```

## Next milestone

Once the numbers look healthy:

- production Docker images
- API/frontend health checks
- environment-specific configuration
- observability / Prometheus metrics
- deployment
