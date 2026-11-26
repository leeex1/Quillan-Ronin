#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic regression tests for the WSP Stage B/C/D non-vacuous-progress gates.

Each failure mode is forced with a tiny CPU-only fixture and the corresponding
gate MUST raise; a positive fixture MUST pass. Also verifies that each stage's
run_stage_* actually wires the gate + FAILED-report writer, and that a failed
report is emitted with exported=False and NO success report.

Run:
    python -u test_wsp_gates.py
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import v23_common as c  # noqa: E402
import v23_stage_b_precombine as stage_b  # noqa: E402
import v23_stage_c_consolidate as stage_c  # noqa: E402
import v23_stage_d_deep_wsp as stage_d  # noqa: E402

WSPGateError = stage_b.WSPGateError
SCRIPTS = Path(r"C:\02_QUILLAN\scripts")


def _expect_raise(label: str, fn: Callable[[], Any]) -> None:
    try:
        fn()
    except WSPGateError as exc:
        print(f"PASS {label}: RAISED WSPGateError: {str(exc)[:96]}")
        return
    raise AssertionError(f"{label}: gate FAILED to raise")


def _expect_pass(label: str, fn: Callable[[], Any]) -> Any:
    result = fn()
    print(f"PASS {label}: gate accepted -> {result!r}")
    return result


def _test_stage_b_gates() -> None:
    delta = torch.randn(8, 8)
    fixtures = {(0, "o_proj"): delta}

    _expect_raise(
        "stage_b_alpha_zero",
        lambda: stage_b.check_stage_b_gate(0.0, fixtures),
    )
    _expect_raise(
        "stage_b_zero_delta_norm",
        lambda: stage_b.check_stage_b_gate(0.02, {(0, "o_proj"): torch.zeros(8, 8)}),
    )
    _expect_raise(
        "stage_b_empty_deltas",
        lambda: stage_b.check_stage_b_gate(0.02, {}),
    )
    _expect_raise(
        "stage_b_val_nll_regression",
        lambda: stage_b.check_stage_b_gate(
            0.02, fixtures, final_val_nll=2.10, base_val_nll=2.00
        ),
    )
    _expect_pass(
        "stage_b_positive",
        lambda: stage_b.check_stage_b_gate(
            0.02, fixtures, final_val_nll=1.98, base_val_nll=2.00
        ),
    )


def _test_stage_c_gates() -> None:
    base_nll, base_mc, mc_tol = 2.0, 0.5, 0.025
    real_cfg = {
        "subset": "mid_upper",
        "layers": [0, 1, 2],
        "g_lm": 0.012,
        "g_mem": 0.005,
        "g_ablit": 0.004,
        "g_head": 0.0,
    }

    # baseline-only: every candidate fails the Pareto gate -> selector returns sentinel
    bad_cfg, bad_metrics = stage_c.select_best_verified_candidate(
        [(dict(real_cfg), 2.01, 0.55), (dict(real_cfg), 1.999, 0.40)],
        base_nll,
        base_mc,
        mc_tol,
    )
    assert bad_cfg["subset"] == "baseline", bad_cfg
    _expect_raise(
        "stage_c_baseline_only",
        lambda: stage_c.check_stage_c_gate(bad_cfg, bad_metrics, base_nll, base_mc),
    )

    # non-baseline label but zero real improvement -> still rejected
    _expect_raise(
        "stage_c_no_real_improvement",
        lambda: stage_c.check_stage_c_gate(
            dict(real_cfg), {"val_nll": 2.0, "dev_mc": 0.5}, base_nll, base_mc
        ),
    )

    good_cfg, good_metrics = stage_c.select_best_verified_candidate(
        [(real_cfg, 1.98, 0.52)], base_nll, base_mc, mc_tol
    )
    assert good_cfg["subset"] == "mid_upper", good_cfg
    _expect_pass(
        "stage_c_positive",
        lambda: stage_c.check_stage_c_gate(good_cfg, good_metrics, base_nll, base_mc),
    )


def _test_stage_d_gates() -> None:
    _expect_raise("stage_d_zero_accepted", lambda: stage_d.check_stage_d_gate([]))
    _expect_raise("stage_d_none_type", lambda: stage_d.check_stage_d_gate(None))  # type: ignore[arg-type]
    _expect_pass("stage_d_positive", lambda: stage_d.check_stage_d_gate([{"layer": 3}]))


def _test_failed_report_is_written_without_export() -> None:
    saved_root = c.EXPORT_ROOT
    with tempfile.TemporaryDirectory() as td:
        c.EXPORT_ROOT = Path(td)
        try:
            report = stage_b.write_failed_report(
                "B", "mini", "synthetic fixture failure", {"alpha": 0.0}
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            assert payload["status"] == "FAILED", payload
            assert payload["exported"] is False, payload
            assert not (Path(td) / "v23_stageb_mini_report.json").exists(), (
                "A success report was written for a FAILED gate"
            )
            print(
                f"PASS failed_report_no_export: {report.name} status=FAILED exported=False"
            )
        finally:
            c.EXPORT_ROOT = saved_root


def _test_gate_wiring_in_each_stage() -> None:
    expected: List[Tuple[str, str, str]] = [
        ("v23_stage_b_precombine.py", "run_stage_b", "check_stage_b_gate"),
        ("v23_stage_c_consolidate.py", "run_stage_c", "check_stage_c_gate"),
        ("v23_stage_d_deep_wsp.py", "run_stage_d", "check_stage_d_gate"),
    ]
    for filename, func_name, gate_name in expected:
        tree = ast.parse((SCRIPTS / filename).read_text(encoding="utf-8"))
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == func_name
        )
        called = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
        assert gate_name in called, f"{func_name} never calls {gate_name}"
        assert "write_failed_report" in called, (
            f"{func_name} never writes a FAILED report"
        )
        print(
            f"PASS gate_wiring: {filename}::{func_name} -> {gate_name} + write_failed_report"
        )


def main() -> None:
    _test_stage_b_gates()
    _test_stage_c_gates()
    _test_stage_d_gates()
    _test_failed_report_is_written_without_export()
    _test_gate_wiring_in_each_stage()
    print("ALL_WSP_GATE_TESTS_PASSED")


if __name__ == "__main__":
    main()
