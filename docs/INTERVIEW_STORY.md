# InspireRank — Interview Story

## 30-second version

I built InspireRank, a production-style multimodal recommendation and visual
discovery system using PyTorch, SigLIP, FastAPI, Next.js, PostgreSQL, and Redis.
I evaluated several recommendation strategies on a 27K-item catalog and found
that purely behavioral models performed poorly on sparse and cold items. I
built a hybrid semantic-behavioral retriever that improved test Recall@20 by
32% over my semantic baseline, then added real-time feedback, personalized
semantic search, diversity reranking, observability, caching, and Dockerized
production infrastructure.

## Engineering progression

1. **Behavioral baseline failed:** a two-tower ID model learned training data
   but underperformed popularity.
2. **Dataset diagnosis:** moved to an official 5-core chronological benchmark.
3. **Semantic content:** SigLIP enabled cold-item retrieval and reached test
   Recall@20 0.0336 versus 0.0114 popularity.
4. **Learned multimodal tradeoff:** warm retrieval improved but cold retrieval
   degraded.
5. **Hybrid retrieval:** kept semantic retrieval catalog-wide and added
   behavioral score only where interaction support was reliable; test
   Recall@20 reached 0.0444.
6. **Online personalization:** PostgreSQL stores events permanently while Redis
   holds short-term preference state for immediate reranking.
7. **Query-conditioned search:** broad-query personalization improved by
   attending to the user's history items most relevant to the current query.
8. **Production evaluation:** profiling showed query encoding dominated latency;
   cached query encoding fell from 86.03 ms to 0.017 ms mean.
9. **Quality audit:** testing caught a multilingual false-positive in the first
   catalog filter, leading to Unicode-aware filtering and regression tests.
10. **Production engineering:** Docker, health probes, structured logs,
    Prometheus-compatible metrics, CI, model warm-up, and HTTPS deployment
    configuration.

## Main takeaway

The strongest part of the project is the iteration loop:

```text
measure -> discover failure mode -> change architecture -> evaluate -> ship
```

Several failed experiments directly informed the final design.
