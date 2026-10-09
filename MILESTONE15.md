# InspireRank Milestone 15 — Production Engineering

Milestone 14 measured two important things:

```text
p50 search latency      83.75 ms
p95 search latency      92.81 ms
query encoding          73.06 ms
rest of search          ~10 ms
```

It also exposed an ASCII-only quality-filter false positive for a legitimate
Japanese origami title.

Milestone 15 addresses both and adds production infrastructure.

## 1. Commit Milestone 14

```powershell
git add .
git commit -m "milestone 14: add evaluation quality filtering and latency instrumentation"
```

## 2. Apply this patch

Copy the patch over the repository.

## 3. Install reproducible dependencies

From the repository root:

```powershell
pip install -e "./apps/api[ml,production,dev]"
```

This adds the Prometheus client and records the runtime/ML dependencies in
`pyproject.toml`.

## 4. Run the Unicode-quality tests

```powershell
pytest apps/api/tests/test_catalog_quality.py -q
```

Expected:

```text
4 passed
```

The tests verify that:

- Japanese product titles remain valid.
- Bernina/model-code-heavy titles remain valid.
- obvious corrupted text is filtered.
- empty titles are filtered.

## 5. Re-run the catalog audit

```powershell
$env:PYTHONPATH="apps/api"
python apps/api/scripts/audit_catalog_quality.py
```

The filtered count should change from Milestone 14 because filtering is now
Unicode-aware and deliberately more conservative. Review the rejected sample
again before treating the filter as production-safe.

## 6. Benchmark query caching

```powershell
python apps/api/scripts/benchmark_query_cache.py
```

The first unique query still requires SigLIP inference. Repeated normalized
queries should be dramatically faster.

These all map to the same cache key:

```text
craft supplies
Craft Supplies
  craft   supplies
```

Live cache diagnostics:

```text
GET /api/v1/diagnostics/query-cache
```

## 7. Start the API locally

```powershell
$env:PYTHONPATH="apps/api"
uvicorn app.main:app --reload --app-dir apps/api --host 0.0.0.0 --port 8000
```

Useful endpoints:

```text
GET /health/live
GET /health/ready
GET /metrics
GET /api/v1/diagnostics/catalog-quality
GET /api/v1/diagnostics/query-cache
```

Every request now has an `X-Request-ID` and
`X-InspireRank-Process-Time-Ms` header, while the API terminal emits structured
JSON logs.

## 8. Production Docker test

Create a private environment file:

```powershell
Copy-Item .env.production.example .env.production
```

Edit `POSTGRES_PASSWORD` before continuing. Do not commit this file.

Then run:

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml up --build
```

The stack contains:

```text
Next.js
FastAPI
PostgreSQL + pgvector
Redis
startup model warm-up
liveness/readiness health checks
structured logging
Prometheus metrics
```

Open:

```text
http://localhost:3000
http://localhost:8000/health/ready
http://localhost:8000/metrics
```

## 9. Production build checks

Frontend:

```powershell
cd apps/web
npm audit
npm run build
cd ../..
```

Backend:

```powershell
pytest apps/api/tests -q
```

## Why one Uvicorn worker?

The model is large. Multiple workers would load multiple model copies into
memory. Use one model-loaded process per container and scale horizontally later.

## What this milestone adds

```text
Unicode-safe catalog quality filtering
bounded query-embedding LRU cache
model/query-encoder startup warm-up
liveness + readiness endpoints
structured JSON request logging
request IDs
Prometheus metrics
stage-level search metrics
production API Docker image
production Next.js Docker image
private backend Docker network
environment-specific production config
```

## Next

Once the production Compose stack is healthy, deploy it publicly and finish the
README, architecture diagram, GitHub presentation, and resume bullets using the
measured results from Milestones 8–15.
