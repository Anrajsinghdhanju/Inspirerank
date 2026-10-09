from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import Request

from app.core.observability import HTTP_LATENCY, HTTP_REQUESTS


logger = logging.getLogger("inspirerank.request")


async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    start = perf_counter()
    status_code = 500
    response = None

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = perf_counter() - start
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)

        HTTP_REQUESTS.labels(
            method=request.method,
            route=route_path,
            status=str(status_code),
        ).inc()
        HTTP_LATENCY.labels(
            method=request.method,
            route=route_path,
        ).observe(duration)

        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": route_path,
                "status": status_code,
                "duration_ms": round(duration * 1000.0, 2),
            },
        )

        if response is not None:
            response.headers["X-Request-ID"] = request_id
            response.headers["X-InspireRank-Process-Time-Ms"] = f"{duration * 1000.0:.2f}"
