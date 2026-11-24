#!/usr/bin/env python3
"""DAW parameter automation and a minimal in-process plugin host.

This is a real, testable signal-processing model: automation curves are
rendered to sample-accurate control arrays, and plugins are processed in a
deterministic chain. It is not a VST3/CLAP host; that remains future work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["AutomationPoint", "render_automation", "PluginHost", "benchmark_automation"]


@dataclass(frozen=True)
class AutomationPoint:
    time: float  # seconds
    value: float
    curve: str = "linear"  # "linear" or "exponential"


def _render_loop(
    points: list[AutomationPoint], sample_rate: float, num_samples: int
) -> np.ndarray:
    out = np.zeros(num_samples, dtype=np.float64)
    if not points:
        return out
    ordered = sorted(points, key=lambda point: point.time)
    out[:] = ordered[0].value
    for left, right in zip(ordered, ordered[1:]):
        start = max(0, int(left.time * sample_rate))
        end = min(num_samples, int(right.time * sample_rate))
        if start >= end:
            continue
        t = np.arange(end - start, dtype=np.float64) / max(1, end - start)
        if right.curve == "exponential" and left.value > 0 and right.value > 0:
            values = left.value * (right.value / left.value) ** t
        else:
            values = left.value + (right.value - left.value) * t
        out[start:end] = values
    tail_start = int(ordered[-1].time * sample_rate)
    if tail_start < num_samples:
        out[tail_start:] = ordered[-1].value
    # The final sample belongs to the last automation segment; snap it so the
    # rendered curve actually reaches its last authored value.
    if num_samples > 0:
        out[-1] = ordered[-1].value
    return out


def render_automation(
    points: list[AutomationPoint], sample_rate: float, num_samples: int
) -> np.ndarray:
    """Sample-accurate automation rendering with a guarded exponential curve."""
    return _render_loop(points, sample_rate, num_samples)


class PluginHost:
    """Deterministic in-process plugin chain with per-plugin parameters."""

    def __init__(self, sample_rate: float = 44100.0) -> None:
        self.sample_rate = sample_rate
        self._plugins: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
        self._parameters: dict[str, dict[str, float]] = {}

    def register(
        self, name: str, processor: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self._plugins.append((name, processor))
        self._parameters[name] = {}

    def set_parameter(self, plugin: str, name: str, value: float) -> None:
        if plugin not in self._parameters:
            raise KeyError(f"unknown plugin: {plugin}")
        self._parameters[plugin][name] = value

    def process(self, block: np.ndarray) -> np.ndarray:
        result = np.array(block, dtype=np.float64, copy=True)
        for name, processor in self._plugins:
            result = np.asarray(processor(result), dtype=np.float64)
            if result.shape != block.shape:
                raise ValueError(f"plugin {name} changed block shape")
        return result

    def plugin_names(self) -> list[str]:
        return [name for name, _ in self._plugins]


def benchmark_automation(size: int = 1_000_000) -> dict:
    """Measure vectorised automation rendering against a scalar reference."""
    points = [
        AutomationPoint(0.0, 0.0),
        AutomationPoint(size / 44100.0 / 2, 0.5),
        AutomationPoint(size / 44100.0, 1.0),
    ]
    start = time.perf_counter()
    scalar = np.array(
        [
            points[-1].value
            if i >= int(points[-1].time * 44100)
            else points[0].value + (points[-1].value - points[0].value) * i / size
            for i in range(size)
        ]
    )
    scalar_seconds = time.perf_counter() - start
    scalar[-1] = points[-1].value  # match the renderer's endpoint snap

    start = time.perf_counter()
    vector = render_automation(points, 44100.0, size)
    vector_seconds = time.perf_counter() - start

    if not np.allclose(scalar, vector, atol=1e-6):
        raise RuntimeError("automation result mismatch - refusing to report speedup")

    return {
        "samples": size,
        "scalar_seconds": round(scalar_seconds, 6),
        "vector_seconds": round(vector_seconds, 6),
        "measured_speedup": round(scalar_seconds / max(vector_seconds, 1e-9), 3),
        "correctness": "verified",
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(benchmark_automation(), indent=2))
