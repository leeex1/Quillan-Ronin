#!/usr/bin/env python3
"""Python mirror of the original GTest suites for the NextVerse Quantum Core.

Every numeric expectation and validation contract is transcribed from
``nextverse_alpha_core_design/tests/test_qcc_alpha.cpp`` and
``tests/test_qsvm_alpha.cpp`` so the Python runtime is verified against the
original specification.
"""

from __future__ import annotations

import math
import time

import pytest

from quantum_core import (
    AlgorithmOptimization,
    JQLDParams,
    LVVMParams,
    QuantumComputationalCore,
    QuantumSuperVM,
    QuantumTaskStatus,
    compound_speedup_report,
)


@pytest.fixture
def qcc() -> QuantumComputationalCore:
    return QuantumComputationalCore()


@pytest.fixture
def qsvm(qcc: QuantumComputationalCore) -> QuantumSuperVM:
    vm = QuantumSuperVM(qcc)
    yield vm  # type: ignore[misc]
    vm.shutdown()


# --- QCC: numeric JQLD -------------------------------------------------------


def test_detailed_jqld_default_params(qcc: QuantumComputationalCore) -> None:
    expected = 1.1e9 * 1024.0
    assert qcc.calculateJQLD_detailed() == pytest.approx(expected, abs=1e5)


def test_detailed_jqld_custom_params() -> None:
    params = JQLDParams(
        classical_base_performance_C=1.0e9,
        algo_optimizations=[AlgorithmOptimization("TestOpt", 2, 0.5, 1.5)],
        omega_q_divisor=2.0,
    )
    qcc = QuantumComputationalCore(params)
    expected = (1.0e9 * (2.0**1.5)) / 2.0
    assert qcc.calculateJQLD_detailed() == pytest.approx(expected, abs=1e2)


def test_zero_omega_divisor_raises() -> None:
    params = JQLDParams(
        algo_optimizations=[AlgorithmOptimization("TestOpt", 1, 1.0, 1.0)],
        omega_q_divisor=0.0,
    )
    with pytest.raises(ValueError):
        QuantumComputationalCore(params)


def test_qlite_default_params(qcc: QuantumComputationalCore) -> None:
    expected = 0.25 * 1.1e9 * math.log2(11.0) * 0.6 * 0.74
    assert qcc.calculateJQLD_Qlite() == pytest.approx(expected, abs=1e2)


def test_qlite_low_ram_constraint() -> None:
    qcc = QuantumComputationalCore(JQLDParams(free_ram_gb_lite=0.5))
    assert qcc.calculateJQLD_Qlite() is None


def test_simplified_jqld_default(qcc: QuantumComputationalCore) -> None:
    assert qcc.calculateJQLD_simplifiedExponent() == pytest.approx(
        1.1e9 * 1024.0, abs=1e5
    )


# --- QCC: validation + task flow --------------------------------------------


@pytest.mark.parametrize("bad_c", [0.0, -1.0])
def test_constructor_rejects_nonpositive_c(bad_c: float) -> None:
    with pytest.raises(ValueError):
        QuantumComputationalCore(JQLDParams(classical_base_performance_C=bad_c))


@pytest.mark.parametrize("bad_omega", [0.0, 1e-10, -1e-10])
def test_constructor_rejects_bad_omega(bad_omega: float) -> None:
    with pytest.raises(ValueError):
        QuantumComputationalCore(JQLDParams(omega_q_divisor=bad_omega))


def test_setter_preserves_state_on_invalid_c(qcc: QuantumComputationalCore) -> None:
    original = qcc.params
    with pytest.raises(ValueError):
        qcc.setJQLDParams(JQLDParams(classical_base_performance_C=0.0))
    assert (
        qcc.params.classical_base_performance_C == original.classical_base_performance_C
    )


def test_setter_preserves_state_on_invalid_omega(qcc: QuantumComputationalCore) -> None:
    original = qcc.params
    with pytest.raises(ValueError):
        qcc.setJQLDParams(JQLDParams(omega_q_divisor=1e-12))
    assert qcc.params.omega_q_divisor == original.omega_q_divisor


def test_submit_and_get_task(qcc: QuantumComputationalCore) -> None:
    task = type("Task", (), {"name": "Test Task"})()
    task_id = qcc.submitQuantumTask(task)
    assert task_id.startswith("QCC_Task_")
    result = qcc.getTaskResult(task_id)
    assert result["success"] and result["task_id"] == task_id
    assert "completion" in result["message"]
    missing = qcc.getTaskResult("non_existent_id_123")
    assert not missing["success"] and "not found" in missing["message"]


# --- QSVM: numeric LVVM ------------------------------------------------------


def test_detailed_lvvm_default(qsvm: QuantumSuperVM) -> None:
    q = 1.1e9 * 1024.0
    numerator = 0.98 + 0.15 * (1.0 - 0.03)
    expected = q * (numerator / 1.02)
    assert qsvm.getAmplifiedPerformance() == pytest.approx(expected, abs=1e7)


def test_simplified_lvvm_default(qsvm: QuantumSuperVM) -> None:
    expected = (1.1e9 * 1024.0) * 0.98
    assert qsvm.calculateLVVM_simplified() == pytest.approx(expected, abs=1e7)


def test_vmslim_default(qsvm: QuantumSuperVM) -> None:
    expected = 0.85 * (1.1e9 * 1024.0) * math.sqrt(0.98) * 0.9
    assert qsvm.calculateLVVM_VMSlim() == pytest.approx(expected, abs=1e6)


def test_vmslim_high_swap_returns_none(qcc: QuantumComputationalCore) -> None:
    vm = QuantumSuperVM(qcc, LVVMParams(swap_usage_percent_slim=150.0))
    try:
        assert vm.calculateLVVM_VMSlim() is None
    finally:
        vm.shutdown()


# --- QSVM: validation + task flow -------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replication_factor_Rvm": -0.1},
        {"sync_cost_tau_vm": -1.0},
        {"sync_cost_tau_vm": -2.0},
        {"virtualization_overhead_mu_vm": 1.0},
        {"virtualization_overhead_mu_vm": 1.1},
        {"virtualization_overhead_mu_vm": -0.1},
    ],
)
def test_qsvm_constructor_rejects_invalid_params(
    qcc: QuantumComputationalCore, kwargs: dict
) -> None:
    with pytest.raises(ValueError):
        QuantumSuperVM(qcc, LVVMParams(**kwargs))


def test_qsvm_setter_preserves_state(qsvm: QuantumSuperVM) -> None:
    original = qsvm.params
    with pytest.raises(ValueError):
        qsvm.setLVVMParams(LVVMParams(replication_factor_Rvm=-0.1))
    assert qsvm.params.replication_factor_Rvm == original.replication_factor_Rvm


def test_qsvm_task_submission_and_completion(qcc: QuantumComputationalCore) -> None:
    vm = QuantumSuperVM(qcc)
    try:
        executed = []
        t1 = vm.submitTask(lambda: executed.append("t1"))
        t2 = vm.submitTask(lambda: executed.append("t2"))
        assert t1.startswith("QSVM_Task_") and t1 != t2
        deadline = time.time() + 3.0
        while time.time() < deadline and sorted(executed) != ["t1", "t2"]:
            time.sleep(0.01)
        assert vm.getTaskStatus(t1) == QuantumTaskStatus.COMPLETED
        assert vm.getTaskStatus(t2) == QuantumTaskStatus.COMPLETED
        assert (
            vm.getTaskStatus("non_existent_qsvm_task_id_456")
            == QuantumTaskStatus.UNKNOWN
        )
    finally:
        vm.shutdown()


def test_stubs(qsvm: QuantumSuperVM) -> None:
    qsvm.manageVMInstances(5)
    assert qsvm.syncWithOtherVM(1, "test_sync_message_content") is True


# --- Compound speed-up report ------------------------------------------------


def test_compound_speedup_report() -> None:
    report = compound_speedup_report()
    assert report["jqld_speedup"] == pytest.approx(1024.0, rel=1e-9)
    assert report["lvvm_detailed_speedup"] > 1000.0
    assert report["eleven_copy_compound_speedup"] == pytest.approx(
        report["lvvm_detailed_speedup"] * 11
    )
