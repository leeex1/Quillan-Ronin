"""Deep correctness / safety checks for the quantum core reference runtime.

These go beyond the GTest-mirroring suite: numeric invariants, boundary
conditions, concurrency, and deterministic cleanup.
"""

from __future__ import annotations

import math
import threading
import time

import pytest

from quantum_core import (
    AlgorithmOptimization,
    JQLDParams,
    LVVMParams,
    QuantumComputationalCore,
    QuantumSuperVM,
    compound_speedup_report,
)


def test_jqld_exponent_is_additive_and_omega_scales_inversely() -> None:
    params = JQLDParams(
        classical_base_performance_C=1000.0,
        algo_optimizations=[
            AlgorithmOptimization("a", 3, 0.5, 2.0),  # 3 * 0.5 * 2 = 3
            AlgorithmOptimization("b", 1, 1.0, 1.0),  # 1
        ],
    )
    qcc = QuantumComputationalCore(params)
    additive = qcc.calculateJQLD_detailed()
    expected = 1000.0 * (2.0**4.0)
    assert additive == pytest.approx(expected, rel=1e-12)

    params.omega_q_divisor = 4.0
    qcc.setJQLDParams(params)
    scaled = qcc.calculateJQLD_detailed()
    assert scaled == pytest.approx(expected / 4.0, rel=1e-12)


def test_jqld_negative_theta_reduces_exponent() -> None:
    params = JQLDParams(
        classical_base_performance_C=1000.0,
        algo_optimizations=[AlgorithmOptimization("negative", 2, 1.0, -1.5)],
    )
    qcc = QuantumComputationalCore(params)
    assert qcc.calculateJQLD_detailed() == pytest.approx(1000.0 * 2.0**-3.0, rel=1e-12)


def test_qlite_boundary_and_ram_monotonicity() -> None:
    values = []
    for ram in (2.0, 4.0, 8.0, 100.0):
        qcc = QuantumComputationalCore(JQLDParams(free_ram_gb_lite=ram))
        values.append(qcc.calculateJQLD_Qlite())
    assert all(v is not None for v in values)
    assert values == sorted(values)  # more free RAM -> higher simulated value
    # 0.01 + 1/RAM >= 1 when RAM <= ~1.0101; must be rejected.
    qcc = QuantumComputationalCore(JQLDParams(free_ram_gb_lite=1.0))
    assert qcc.calculateJQLD_Qlite() is None


def test_constructor_boundaries() -> None:
    # C must be strictly positive; tiny positive values are valid.
    with pytest.raises(ValueError):
        QuantumComputationalCore(JQLDParams(classical_base_performance_C=0.0))
    tiny = QuantumComputationalCore(JQLDParams(classical_base_performance_C=1e-12))
    assert tiny.calculateJQLD_simplifiedExponent() == pytest.approx(
        1e-12 * 1024.0, rel=1e-12
    )
    for omega in (0.0, 9e-10, -9e-10):
        with pytest.raises(ValueError):
            QuantumComputationalCore(JQLDParams(omega_q_divisor=omega))
    boundary = QuantumComputationalCore(JQLDParams(omega_q_divisor=1e-9))
    assert boundary.params.omega_q_divisor == 1e-9
    qcc = QuantumComputationalCore()
    assert qcc.calculateJQLD_simplifiedExponent() == pytest.approx(
        1.1e9 * 1024.0, rel=1e-12
    )


def test_task_ids_unique_under_threads() -> None:
    qcc = QuantumComputationalCore()
    ids: list[str] = []
    lock = threading.Lock()

    def submit() -> None:
        for _ in range(50):
            task_id = qcc.submitQuantumTask(object())
            with lock:
                ids.append(task_id)

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == 200
    assert len(set(ids)) == 200


def test_qsvm_worker_drains_100_tasks() -> None:
    qcc = QuantumComputationalCore()
    vm = QuantumSuperVM(qcc)
    try:
        counter = {"n": 0}

        def work() -> None:
            counter["n"] += 1

        task_ids = [vm.submitTask(work) for _ in range(100)]
        deadline = time.time() + 5.0
        while time.time() < deadline and any(
            vm.getTaskStatus(t) != "COMPLETED" for t in task_ids
        ):
            time.sleep(0.005)
        assert counter["n"] == 100
        assert all(vm.getTaskStatus(t) == "COMPLETED" for t in task_ids)
    finally:
        vm.shutdown()


def test_qsvm_detailed_lvvm_bounds() -> None:
    qcc = QuantumComputationalCore()
    # tau just above -1 gives a large but finite amplification
    vm = QuantumSuperVM(qcc, LVVMParams(sync_cost_tau_vm=-0.999))
    try:
        result = vm.getAmplifiedPerformance()
        assert result is not None and math.isfinite(result)
        assert result > 0
    finally:
        vm.shutdown()


def test_compound_report_is_self_consistent() -> None:
    report = compound_speedup_report(copies=7)
    assert report["jqld_speedup"] == pytest.approx(1024.0, rel=1e-12)
    assert report["lvvm_detailed_speedup"] == pytest.approx(1129.913725490196, rel=1e-9)
    assert report["eleven_copy_compound_speedup"] == pytest.approx(
        report["lvvm_detailed_speedup"] * report["copies"], rel=1e-12
    )
    # The compound number is a product of simulated metrics, not measured speed.
    assert report["lvvm_detailed_ops"] == pytest.approx(
        report["classical_baseline_ops"] * report["lvvm_detailed_speedup"], rel=1e-9
    )
