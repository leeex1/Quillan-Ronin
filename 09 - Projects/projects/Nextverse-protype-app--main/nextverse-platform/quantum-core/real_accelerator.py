#!/usr/bin/env python3
"""REAL performance acceleration for the NextVerse core layer.

Nothing in this module is a simulated metric. Every number returned by the
benchmark functions is wall-clock measured on the host. Two mechanisms provide
genuine speed-ups on classical hardware:

1. numpy vectorisation for bulk JQLD/LVVM-style arithmetic;
2. multiprocessing across physical/logical CPU cores for CPU-bound task batches.

The legacy JQLD/LVVM "boost" constants remain simulations for modelling only.
They are NOT used here and must never be reported as measured acceleration.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from typing import Any, Callable, Iterable

import numpy as np

__all__ = [
    "cpu_count",
    "cpu_work",
    "sequential_map",
    "parallel_map",
    "benchmark_parallel",
    "vectorised_jqld_batch",
    "naive_jqld_batch",
    "benchmark_vectorised",
]


def cpu_count() -> int:
    return os.cpu_count() or 1


def cpu_work(limit: int) -> int:
    """Deterministic CPU-bound workload: count primes up to ``limit``.

    Module-level and picklable so multiprocessing can distribute it.
    """
    if limit < 2:
        return 0
    count = 0
    for candidate in range(2, limit + 1):
        root = math.isqrt(candidate)
        prime = True
        for divisor in range(2, root + 1):
            if candidate % divisor == 0:
                prime = False
                break
        if prime:
            count += 1
    return count


def sequential_map(
    items: Iterable[int], worker: Callable[[int], Any] = cpu_work
) -> list:
    return [worker(item) for item in items]


def parallel_map(
    items: Iterable[int],
    workers: int | None = None,
    worker: Callable[[int], Any] = cpu_work,
) -> list:
    """Execute ``worker`` across real OS processes; preserve input order."""
    data = list(items)
    count = workers if workers is not None else cpu_count()
    if count <= 1 or len(data) < 2:
        return sequential_map(data, worker)
    chunk = max(1, len(data) // (count * 4))
    with mp.Pool(processes=count) as pool:
        return pool.map(worker, data, chunksize=chunk)


def benchmark_parallel(
    limit: int = 9000, repeats: int = 240, workers: int | None = None
) -> dict:
    """Measure sequential vs multiprocessing execution and verify equality."""
    items = [limit] * repeats
    start = time.perf_counter()
    sequential = sequential_map(items)
    sequential_seconds = time.perf_counter() - start

    worker_count = workers if workers is not None else cpu_count()
    start = time.perf_counter()
    parallel = parallel_map(items, workers=worker_count)
    parallel_seconds = time.perf_counter() - start

    if sequential != parallel:
        raise RuntimeError("parallel result mismatch - refusing to report speedup")

    return {
        "items": repeats,
        "workers": worker_count,
        "sequential_seconds": round(sequential_seconds, 6),
        "parallel_seconds": round(parallel_seconds, 6),
        "measured_speedup": round(sequential_seconds / max(parallel_seconds, 1e-9), 3),
        "correctness": "verified",
    }


def naive_jqld_batch(
    base: np.ndarray, exponent: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    return np.array(
        [
            float(b) * math.pow(2.0, float(e)) / float(o)
            for b, e, o in zip(base, exponent, omega)
        ],
        dtype=np.float64,
    )


def vectorised_jqld_batch(
    base: np.ndarray, exponent: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(base, dtype=np.float64)
        * np.exp2(np.asarray(exponent, dtype=np.float64))
        / np.asarray(omega, dtype=np.float64)
    )


def benchmark_vectorised(size: int = 400_000, seed: int = 7) -> dict:
    """Measure numpy vectorisation against an element-wise Python loop."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(1e8, 2e9, size)
    exponent = rng.uniform(0.5, 12.0, size)
    omega = rng.uniform(0.5, 3.0, size)

    start = time.perf_counter()
    slow = naive_jqld_batch(base, exponent, omega)
    slow_seconds = time.perf_counter() - start

    start = time.perf_counter()
    fast = vectorised_jqld_batch(base, exponent, omega)
    fast_seconds = time.perf_counter() - start

    if not np.allclose(slow, fast, rtol=1e-12):
        raise RuntimeError("vectorised result mismatch - refusing to report speedup")

    return {
        "items": size,
        "loop_seconds": round(slow_seconds, 6),
        "vectorised_seconds": round(fast_seconds, 6),
        "measured_speedup": round(slow_seconds / max(fast_seconds, 1e-9), 3),
        "correctness": "verified",
    }


if __name__ == "__main__":  # pragma: no cover - multiprocessing guard
    import json

    print(
        json.dumps(
            {
                "cpu_count": cpu_count(),
                "parallel": benchmark_parallel(),
                "vectorised": benchmark_vectorised(),
            },
            indent=2,
        )
    )
