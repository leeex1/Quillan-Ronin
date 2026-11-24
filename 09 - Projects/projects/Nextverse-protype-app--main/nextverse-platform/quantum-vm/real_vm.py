#!/usr/bin/env python3
"""RealQuantumVM - measured, classical-hardware acceleration layer.

This is NOT the simulated 11-copy JQLD/LVVM model. It executes real work using:
  * OS processes across available CPU cores for CPU-bound batches;
  * numpy vectorisation for arithmetic batches.

All reported speed-ups are measured with time.perf_counter and correctness is
verified before any speed-up is returned. On a mismatch, this module raises.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

# Import the real accelerator from the sibling quantum-core package.
_CORE = Path(__file__).resolve().parent.parent / "quantum-core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from real_accelerator import (  # noqa: E402
    benchmark_parallel,
    benchmark_vectorised,
    cpu_count,
    parallel_map,
    sequential_map,
    vectorised_jqld_batch,
)

__all__ = ["RealQuantumVM"]


class RealQuantumVM:
    """Executes real workloads with measurable acceleration."""

    def __init__(self, workers: int | None = None) -> None:
        self.workers = workers or cpu_count()
        self.last_report: dict[str, Any] = {}

    def execute_cpu_batch(self, workloads: Iterable[int]) -> list[int]:
        """Run CPU-bound work in parallel; results are order-identical."""
        data = list(workloads)
        sequential = sequential_map(data)
        parallel = parallel_map(data, workers=self.workers)
        if sequential != parallel:
            raise RuntimeError("parallel VM result mismatch")
        self.last_report = {
            "mode": "multiprocessing",
            "workers": self.workers,
            "items": len(data),
            "correctness": "verified",
        }
        return parallel

    def execute_jqld_batch(self, base, exponent, omega):
        """Vectorised bulk JQLD computation, verified against a Python loop."""
        import numpy as np
        from real_accelerator import naive_jqld_batch

        b = np.asarray(base, dtype=np.float64)
        e = np.asarray(exponent, dtype=np.float64)
        o = np.asarray(omega, dtype=np.float64)
        expected = naive_jqld_batch(b, e, o)
        result = vectorised_jqld_batch(b, e, o)
        if not np.allclose(expected, result, rtol=1e-12):
            raise RuntimeError("vectorised VM result mismatch")
        self.last_report = {
            "mode": "vectorised",
            "items": int(b.size),
            "correctness": "verified",
        }
        return result

    def benchmark(self) -> dict[str, Any]:
        """Return measured speed-ups for both acceleration mechanisms."""
        parallel = benchmark_parallel(workers=self.workers)
        vectorised = benchmark_vectorised()
        self.last_report = {
            "mode": "benchmark",
            "parallel": parallel,
            "vectorised": vectorised,
        }
        return self.last_report


if __name__ == "__main__":  # pragma: no cover
    import json

    vm = RealQuantumVM()
    print(json.dumps(vm.benchmark(), indent=2))
