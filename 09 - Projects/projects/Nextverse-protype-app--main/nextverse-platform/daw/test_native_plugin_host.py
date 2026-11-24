"""Native plugin host verification: compile, load, process, benchmark."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from native_plugin_host import NativePlugin, benchmark_native_plugin, build_gain_plugin


@pytest.fixture(scope="module")
def plugin() -> NativePlugin:
    compiler = shutil.which("clang++") or r"C:\Program Files\LLVM\bin\clang++.exe"
    if not Path(compiler).exists():
        compiler = "clang++"
    output = build_gain_plugin(
        Path(__file__).parent / "plugins" / "build", compiler=compiler
    )
    return NativePlugin(output)


def test_plugin_abi_and_name(plugin: NativePlugin) -> None:
    assert "Gain" in plugin.name
    assert plugin.path.exists()


def test_plugin_processes_audio(plugin: NativePlugin) -> None:
    plugin.set_parameter("gain", 0.5)
    block = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    result = plugin.process(block)
    assert result.shape == block.shape
    assert np.allclose(result, [[0.5, -0.5], [0.25, -0.25]], atol=1e-6)


def test_parameter_changes_take_effect(plugin: NativePlugin) -> None:
    plugin.set_parameter("gain", 1.0)
    result = plugin.process(np.array([[0.25, 0.25]], dtype=np.float32))
    assert np.allclose(result, [[0.25, 0.25]], atol=1e-6)
    plugin.set_parameter("gain", 0.5)


def test_invalid_block_rejected(plugin: NativePlugin) -> None:
    with pytest.raises(ValueError):
        plugin.process(np.ones(8, dtype=np.float32))


def test_native_benchmark_measured(plugin: NativePlugin) -> None:
    report = benchmark_native_plugin(plugin, frames=8192)
    assert report["correctness"] == "verified"
    assert report["measured_speedup"] > 0
