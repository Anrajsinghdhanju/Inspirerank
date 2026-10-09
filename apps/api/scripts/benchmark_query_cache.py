from __future__ import annotations

from time import perf_counter

import numpy as np

from app.services.semantic_query_encoder import get_query_encoder


def timed(encoder, query: str) -> float:
    start = perf_counter()
    encoder.encode(query)
    return (perf_counter() - start) * 1000.0


def main() -> None:
    encoder = get_query_encoder()
    query = "origami paper with Japanese patterns"

    encoder.encode("inspirerank cache warmup")

    first = timed(encoder, query)
    repeats = [timed(encoder, query) for _ in range(20)]

    print("\nInspireRank query-embedding cache benchmark")
    print("=" * 64)
    print(f"uncached query: {first:.2f} ms")
    print(f"cached mean:   {np.mean(repeats):.3f} ms")
    print(f"cached p95:    {np.percentile(repeats, 95):.3f} ms")
    print(f"cache info:    {encoder.cache_info()}")


if __name__ == "__main__":
    main()
