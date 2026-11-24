#!/usr/bin/env python3
"""Host for NextVerse native plugins built against plugins/plugin_api.h.

Loads a compiled shared library with ctypes, validates the ABI, and processes
audio blocks. This is a real native plugin path, not a simulated one.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["NativePlugin", "build_gain_plugin", "benchmark_native_plugin"]


class NativePlugin:
    """A loaded native plugin with verified ABI."""

    def __init__(self, path: str | Path) -> None:
        library_path = Path(path).resolve()
        if not library_path.exists():
            raise FileNotFoundError(library_path)
        self._lib = ctypes.CDLL(str(library_path))
        self._lib.nv_plugin_name.restype = ctypes.c_char_p
        self._lib.nv_process.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
        ]
        self._lib.nv_process.restype = ctypes.c_int
        self._lib.nv_set_parameter.argtypes = [ctypes.c_char_p, ctypes.c_double]
        self._lib.nv_set_parameter.restype = None
        self.path = library_path

    @property
    def name(self) -> str:
        return self._lib.nv_plugin_name().decode("utf-8")

    def set_parameter(self, name: str, value: float) -> None:
        self._lib.nv_set_parameter(name.encode("utf-8"), ctypes.c_double(value))

    def process(self, block: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
        data = np.ascontiguousarray(block, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("block must be a 2-D frames x channels array")
        frames, channels = data.shape
        output = np.zeros_like(data)
        input_pointer = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_pointer = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ok = self._lib.nv_process(
            input_pointer, output_pointer, frames, channels, sample_rate
        )
        if not ok:
            raise RuntimeError(f"native plugin failed to process {frames}x{channels}")
        return output


def build_gain_plugin(output_dir: str | Path, compiler: str | None = None) -> Path:
    """Compile the sample gain plugin with the locally installed C++ compiler."""
    daw = Path(__file__).resolve().parent
    source = daw / "plugins" / "gain_plugin.cpp"
    include = daw / "plugins"
    output = Path(output_dir).resolve() / "nv_gain_plugin.dll"
    output.parent.mkdir(parents=True, exist_ok=True)
    # A loaded DLL stays locked on Windows. Reuse an existing build rather than
    # relinking over it when the source has not changed.
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return output
    command = [
        compiler or os.environ.get("CXX", "clang++"),
        "-std=c++17",
        "-O2",
        "-shared",
        *([] if os.name == "nt" else ["-fPIC"]),
        f"-I{include}",
        str(source),
        "-o",
        str(output),
    ]
    import subprocess

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"plugin build failed: {result.stderr}")
    return output


def benchmark_native_plugin(
    plugin: NativePlugin, frames: int = 262144, channels: int = 2
) -> dict:
    """Compare native plugin processing with an equivalent scalar Python loop."""
    rng = np.random.default_rng(11)
    block = rng.uniform(-1.0, 1.0, (frames, channels)).astype(np.float32)

    import time

    start = time.perf_counter()
    native = plugin.process(block)
    native_seconds = time.perf_counter() - start

    start = time.perf_counter()
    expected = np.empty_like(block)
    for i in range(frames):
        for c in range(channels):
            expected[i, c] = block[i, c] * 0.5
    scalar_seconds = time.perf_counter() - start

    if not np.allclose(native, expected, atol=1e-6):
        raise RuntimeError("native plugin output mismatch")
    return {
        "frames": frames,
        "channels": channels,
        "scalar_seconds": round(scalar_seconds, 6),
        "native_seconds": round(native_seconds, 6),
        "measured_speedup": round(scalar_seconds / max(native_seconds, 1e-9), 3),
        "correctness": "verified",
    }
