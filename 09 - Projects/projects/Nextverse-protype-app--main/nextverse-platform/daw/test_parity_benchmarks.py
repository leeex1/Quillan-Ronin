"""Parity benchmark verification: native chain, automation, sequencer timing."""

from __future__ import annotations

import shutil
from pathlib import Path

from native_plugin_host import build_gain_plugin
from parity_benchmarks import benchmark_native_chain, benchmark_sequencer_timing
from automation import benchmark_automation


def _compiler() -> str | None:
    local = r"C:\Program Files\LLVM\bin\clang++.exe"
    if Path(local).exists():
        return local
    return shutil.which("clang++")


def test_native_plugin_chain_benchmark() -> None:
    plugin = build_gain_plugin(
        Path(__file__).parent / "plugins" / "build", compiler=_compiler()
    )
    report = benchmark_native_chain(plugin, frames=4096)
    assert report["correctness"] == "verified"
    assert report["plugins"] == 3
    assert report["native_vs_numpy"] > 0
    print("[BENCH] native chain:", report)


def test_automation_benchmark() -> None:
    report = benchmark_automation(size=200_000)
    assert report["correctness"] == "verified"
    assert report["measured_speedup"] > 1.0
    print("[BENCH] automation:", report)


def test_sequencer_timing_benchmark() -> None:
    report = benchmark_sequencer_timing()
    assert report["correctness"] == "verified"
    assert report["max_drift_ms"] < 0.001
    assert report["steps"] == 128
    print("[BENCH] sequencer:", report)
