# InspireRank Production Architecture

```text
Browser
  |
  v
Next.js container
  |
  v
FastAPI container
  |---- PostgreSQL + pgvector
  |---- Redis
  |---- read-only model/data artifacts
  |
  `---- /metrics -> Prometheus-compatible scraper
```

## One API worker by design

The recommender and SigLIP encoder are memory-heavy singletons. Multiple Uvicorn
workers in the same container would duplicate model memory. The production image
therefore uses one worker. Scale horizontally with multiple API containers when
traffic requires it.

## Health endpoints

- `/health/live` checks whether the API process is alive.
- `/health/ready` checks PostgreSQL, Redis, and required model/data artifacts.

Docker uses readiness for the API health check.

## Startup warm-up

Production Compose enables `WARM_RECOMMENDER=true` and
`WARM_QUERY_ENCODER=true`. Model loading happens during startup rather than on
the first real user request.

## Query embedding cache

Milestone 14 showed SigLIP query encoding dominated request latency. Milestone 15
adds a bounded process-local LRU cache for normalized query embeddings. Default
capacity is 256 queries. The cache is intentionally bounded to prevent unbounded
memory growth.

## Observability

Prometheus endpoint:

```text
GET /metrics
```

Important metrics:

```text
inspirerank_http_requests_total
inspirerank_http_request_duration_seconds
inspirerank_search_stage_duration_seconds
inspirerank_query_embedding_cache_events_total
inspirerank_recommender_ready
inspirerank_query_encoder_ready
```

Every request also emits a JSON log and returns:

```text
X-Request-ID
X-InspireRank-Process-Time-Ms
```

## Secrets

Do not commit `.env.production`. Use your deployment provider's secret manager
for database passwords, credentials, and future API keys.

## Model artifacts

The production Compose file mounts `./data` and `./artifacts` read-only instead
of baking large model files into the image. In cloud deployment, replace these
host mounts with a persistent volume or release-time download from object
storage.
