#!/usr/bin/env python3
"""Verification of the REAL acceleration layer (measured, not simulated)."""

from __future__ import annotations

import sys
from pathlib import Path

VM_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VM_DIR))

from real_vm import RealQuantumVM  # noqa: E402


def main() -> int:
    vm = RealQuantumVM()

    workloads = [9000] * 120
    result = vm.execute_cpu_batch(workloads)
    if len(result) != len(workloads) or len(set(result)) != 1:
        print("[FAIL] parallel batch result incorrect")
        return 1
    print(
        f"[PASS] real parallel batch: {len(result)} items, workers={vm.last_report['workers']}"
    )

    base = [1.0e9] * 50_000
    exponent = [10.0] * 50_000
    omega = [1.0] * 50_000
    jqld = vm.execute_jqld_batch(base, exponent, omega)
    expected = 1.0e9 * 1024.0
    if not all(abs(float(value) - expected) < 1.0 for value in jqld[:10]):
        print("[FAIL] vectorised JQLD batch incorrect")
        return 1
    print(f"[PASS] real vectorised batch: {jqld.size} items, verified values")

    report = vm.benchmark()
    parallel_speedup = report["parallel"]["measured_speedup"]
    vector_speedup = report["vectorised"]["measured_speedup"]
    if parallel_speedup <= 1.0 or vector_speedup <= 1.0:
        print("[FAIL] no measured speedup")
        return 1
    print(f"[PASS] measured multiprocessing speedup = {parallel_speedup}x")
    print(f"[PASS] measured vectorisation speedup = {vector_speedup}x")
    print("[PASS] REAL VM verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
