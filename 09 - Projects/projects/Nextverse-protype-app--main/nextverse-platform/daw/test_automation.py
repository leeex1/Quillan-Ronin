"""DAW automation + plugin-host verification suite."""

from __future__ import annotations

import numpy as np
import pytest

from automation import (
    AutomationPoint,
    PluginHost,
    benchmark_automation,
    render_automation,
)


def test_linear_automation_endpoints() -> None:
    points = [AutomationPoint(0.0, 0.0), AutomationPoint(1.0, 1.0)]
    result = render_automation(points, 100.0, 100)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result) >= 0)


def test_exponential_automation_is_positive_and_monotonic() -> None:
    points = [AutomationPoint(0.0, 0.1), AutomationPoint(1.0, 1.0, "exponential")]
    result = render_automation(points, 100.0, 100)
    assert np.all(result > 0)
    assert np.all(np.diff(result) >= 0)


def test_empty_automation_returns_silence() -> None:
    assert np.all(render_automation([], 44100.0, 512) == 0.0)


def test_plugin_chain_order_and_parameter_state() -> None:
    host = PluginHost()
    host.register("gain", lambda block: block * 2.0)
    host.register("clip", lambda block: np.clip(block, -0.5, 0.5))
    host.set_parameter("gain", "level", 2.0)
    assert host.plugin_names() == ["gain", "clip"]
    assert host._parameters["gain"]["level"] == 2.0
    out = host.process(np.array([0.4, -0.4]))
    assert out.tolist() == pytest.approx([0.5, -0.5])


def test_plugin_shape_violation_is_rejected() -> None:
    host = PluginHost()
    host.register("bad", lambda block: block[:1])
    with pytest.raises(ValueError):
        host.process(np.ones(4))


def test_automation_benchmark_measures_real_speedup() -> None:
    report = benchmark_automation(size=200_000)
    assert report["correctness"] == "verified"
    assert report["measured_speedup"] > 1.0
