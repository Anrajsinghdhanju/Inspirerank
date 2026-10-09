# InspireRank — Resume Options

## Recommended three bullets

**InspireRank — Multimodal Recommendation & Discovery Platform**  
*PyTorch, SigLIP, FastAPI, Next.js, TypeScript, PostgreSQL, Redis, Docker*

- Built an end-to-end multimodal recommendation platform across a 27K-item
  catalog, combining SigLIP semantic retrieval with learned behavioral signals;
  improved test Recall@20 by **32%** over the semantic baseline and **3.9×**
  over popularity.
- Developed real-time personalization and query-conditioned semantic search
  using PostgreSQL event logging, Redis short-term preference state, and MMR
  diversity reranking while preserving cold-start content retrieval.
- Profiled and productionized the system with Docker, health checks,
  Prometheus-compatible metrics, structured logging, and bounded embedding
  caching, reducing repeated query-encoding latency from **86 ms to 0.02 ms**
  mean.

## Short two-bullet version

- Built a PyTorch/SigLIP multimodal recommendation system for a 27K-item
  catalog; hybrid semantic-behavioral retrieval improved test Recall@20 by
  **32%** over the content baseline and **3.9×** over popularity.
- Shipped real-time Redis personalization, query-conditioned semantic search,
  MMR diversity, FastAPI/Next.js serving, and Docker observability; reduced
  repeated query-encoding latency from **86 ms to 0.02 ms** via caching.

## Do not overclaim

Use:
- 27K-item catalog
- 5K-user evaluation subset
- 34,886 training interactions

Do not say:
- millions of project interactions
- production users
- deployed at scale
- sub-millisecond end-to-end search

The ~0.02 ms result is the cached **query-encoding stage**.
