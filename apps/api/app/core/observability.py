from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "inspirerank_http_requests_total",
    "HTTP requests handled by InspireRank.",
    ["method", "route", "status"],
)

HTTP_LATENCY = Histogram(
    "inspirerank_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "route"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0),
)

SEARCH_STAGE_LATENCY = Histogram(
    "inspirerank_search_stage_duration_seconds",
    "Latency of each search pipeline stage.",
    ["stage"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)

QUERY_CACHE_EVENTS = Counter(
    "inspirerank_query_embedding_cache_events_total",
    "Query embedding cache hit/miss events.",
    ["result"],
)

MODEL_READY = Gauge(
    "inspirerank_recommender_ready",
    "1 when the recommender has completed warm-up.",
)

QUERY_ENCODER_READY = Gauge(
    "inspirerank_query_encoder_ready",
    "1 when the query encoder has completed warm-up.",
)


def observe_search_timings(timings_ms: dict[str, float]) -> None:
    for stage, milliseconds in timings_ms.items():
        SEARCH_STAGE_LATENCY.labels(stage=stage).observe(milliseconds / 1000.0)
