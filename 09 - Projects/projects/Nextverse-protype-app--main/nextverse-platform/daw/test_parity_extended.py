"""Extended DAW parity coverage: curve types, longer plugin chains, BPM sweep."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from automation import AutomationPoint, benchmark_automation, render_automation
from native_plugin_host import NativePlugin, build_gain_plugin
from parity_benchmarks import benchmark_native_chain, benchmark_sequencer_timing


@pytest.fixture(scope="module")
def plugin_path() -> Path:
    local = r"C:\Program Files\LLVM\bin\clang++.exe"
    compiler = local if Path(local).exists() else shutil.which("clang++")
    return build_gain_plugin(
        Path(__file__).parent / "plugins" / "build", compiler=compiler
    )


def test_exponential_and_linear_curves_are_distinct() -> None:
    points_linear = [AutomationPoint(0.0, 0.1), AutomationPoint(1.0, 1.0, "linear")]
    points_exp = [AutomationPoint(0.0, 0.1), AutomationPoint(1.0, 1.0, "exponential")]
    linear = render_automation(points_linear, 100.0, 100)
    exponential = render_automation(points_exp, 100.0, 100)
    assert np.allclose(linear[50], 0.55, atol=0.02)
    # Geometric midpoint between 0.1 and 1.0 is sqrt(0.1) ~ 0.316.
    assert np.allclose(exponential[50], 0.3162, atol=0.01)
    assert exponential[50] < linear[50]
    print("[PASS] linear vs exponential automation curves match their math")


def test_five_plugin_chain_correctness(plugin_path: Path) -> None:
    report = benchmark_native_chain(plugin_path, plugins=5, frames=8192)
    assert report["correctness"] == "verified"
    assert report["plugins"] == 5
    print("[PASS] 5-plugin native chain:", report["native_chain_seconds"], "s")


def test_sequencer_bpm_sweep() -> None:
    for bpm in (60, 90, 120, 174, 200):
        report = benchmark_sequencer_timing(bpm=bpm, bars=2)
        assert report["max_drift_ms"] == 0.0
        assert report["correctness"] == "verified"
    print("[PASS] sequencer timing exact across BPM 60/90/120/174/200")


def test_automation_speedup_across_sizes() -> None:
    for size in (10_000, 100_000, 500_000):
        report = benchmark_automation(size=size)
        assert report["measured_speedup"] > 1.0
        assert report["correctness"] == "verified"
    print("[PASS] automation vectorisation wins at 10k/100k/500k samples")
