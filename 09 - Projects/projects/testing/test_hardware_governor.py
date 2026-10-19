#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and regression tests for QuillanHardwareTelemetryGovernor.
"""

import sys
import time
import unittest
from pathlib import Path

# Add scripts directory to sys.path
scripts_dir = Path(r"c:\02_QUILLAN\05_Training\scripts")
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from quillan_hardware_governor import (
    HardwareTelemetryGovernor,
    HardwareThresholds,
    TelemetrySnapshot,
    LeeMach6GovernorAdapter,
)


class TestHardwareTelemetryGovernor(unittest.TestCase):
    """Test suite for HardwareTelemetryGovernor."""

    def test_core_lifecycle_and_snapshot(self):
        """Test daemon start, stop, and snapshot generation."""
        thresholds = HardwareThresholds(sample_interval_s=0.05)
        with HardwareTelemetryGovernor(thresholds=thresholds) as gov:
            time.sleep(0.15)
            snap = gov.get_snapshot()
            self.assertIsInstance(snap, TelemetrySnapshot)
            self.assertGreater(snap.timestamp, 0.0)
            self.assertGreaterEqual(snap.host_ram_used_pct, 0.0)

    def test_thermal_critical_throttling(self):
        """Test that thermal critical readings inject throttling and micro-delays."""
        # Force thresholds to low values to simulate thermal critical condition
        thresholds = HardwareThresholds(temp_crit_c=10.0, cooldown_sleep_s=0.01)
        with HardwareTelemetryGovernor(thresholds=thresholds) as gov:
            time.sleep(0.05)
            directives = gov.step_boundary_hook()
            self.assertLessEqual(directives["throttle_factor"], 0.5)
            self.assertGreater(directives["paused_seconds"], 0.0)

    def test_opportunistic_gpu_scheduling(self):
        """Test that idle GPU conditions trigger queued opportunistic tasks."""
        # Set idle threshold to 100% so current GPU usage is guaranteed to be idle
        thresholds = HardwareThresholds(gpu_idle_pct=100.0)
        executed_tasks = []

        def sample_task():
            executed_tasks.append("prefetched_data")

        with HardwareTelemetryGovernor(thresholds=thresholds) as gov:
            gov.register_opportunistic_task("data_prefetch", sample_task)
            time.sleep(0.05)
            directives = gov.step_boundary_hook()
            self.assertEqual(directives["opportunistic_executed"], "data_prefetch")
            self.assertIn("prefetched_data", executed_tasks)

    def test_legacy_leemach6_adapter(self):
        """Test backward-compatibility shim for LeeMach6Governor."""
        thresholds = HardwareThresholds(sample_interval_s=0.05)
        with HardwareTelemetryGovernor(thresholds=thresholds) as gov:
            adapter = LeeMach6GovernorAdapter(target_latency_ms=100, hardware_governor=gov)
            scale, ema, bias = adapter.adjust(latency_ms=150.0)
            self.assertIsInstance(scale, float)
            self.assertIsInstance(ema, float)
            self.assertIsInstance(bias, float)
            self.assertLess(scale, 1.0)
            self.assertEqual(bias, 1.0)


if __name__ == "__main__":
    unittest.main()
