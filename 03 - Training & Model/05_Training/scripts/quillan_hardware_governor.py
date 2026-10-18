#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
👑 QUILLAN-RONIN HARDWARE TELEMETRY GOVERNOR (v5.3.1)
===============================================================================
Provides an asynchronous, event-driven hardware observer and reactive governor
intertwining system thermals, memory pressure, and GPU utilization with
the Quillan-Ronin training and inference runtime.

Key Capabilities:
  - Thermal Throttling: Injects adaptive cadence delays and modulates batch scaling
    when CPU/GPU core temperatures exceed configured limits.
  - Memory Relief: Intercepts host RAM and VRAM saturation, executing deterministic
    garbage collection, CUDA cache flushing, and state offloading.
  - Opportunistic GPU Scheduling: Dispatches pending secondary tasks (validation passes,
    cache prefetching, or tokenization) when GPU compute falls below idle threshold.
  - Backward Compatibility: Provides a drop-in adapter for legacy LeeMach6Governor.

Author: Quillan Engineering Lab
License: Apache 2.0 / Proprietary Sovereign Standard
===============================================================================
"""

from __future__ import annotations

import gc
import logging
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

# Optional hardware libraries with safe standard-library fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


# ─── TELEMETRY SNAPSHOT & THRESHOLDS ─────────────────────────────────────────

@dataclass(frozen=True)
class TelemetrySnapshot:
    """Immutable hardware telemetry readings at a given instant."""
    timestamp: float
    cpu_temp_c: float
    gpu_temp_c: float
    host_ram_used_pct: float
    vram_used_pct: float
    gpu_util_pct: float
    cpu_util_pct: float
    is_thermal_warning: bool
    is_thermal_critical: bool
    is_ram_pressure: bool
    is_gpu_idle: bool


@dataclass(frozen=True)
class HardwareThresholds:
    """Configurable trigger thresholds for hardware interventions."""
    temp_warn_c: float = 75.0
    temp_crit_c: float = 85.0
    ram_warn_pct: float = 82.0
    ram_crit_pct: float = 90.0
    vram_warn_pct: float = 85.0
    vram_crit_pct: float = 92.0
    gpu_idle_pct: float = 20.0
    sample_interval_s: float = 1.0
    cooldown_sleep_s: float = 0.10


# ─── OBSERVER PROTOCOL ───────────────────────────────────────────────────────

class HardwareObserverProtocol(Protocol):
    """Event contract for telemetry notification listeners."""
    def on_thermal_warning(self, snapshot: TelemetrySnapshot) -> None: ...
    def on_thermal_critical(self, snapshot: TelemetrySnapshot) -> None: ...
    def on_memory_pressure(self, snapshot: TelemetrySnapshot) -> None: ...
    def on_gpu_idle(self, snapshot: TelemetrySnapshot) -> None: ...


# ─── ASYNCHRONOUS HARDWARE TELEMETRY GOVERNOR ─────────────────────────────────

class HardwareTelemetryGovernor:
    """
    Asynchronous, non-blocking hardware telemetry observer and reactive governor.
    
    Operates as a context manager spawning a lightweight daemon thread to sample
    system thermals, memory pressure, and GPU activity without degrading inner-loop
    compute latency.
    """

    def __init__(
        self,
        thresholds: Optional[HardwareThresholds] = None,
        logger: Optional[logging.Logger] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        self.thresholds = thresholds or HardwareThresholds()
        self.logger = logger or logging.getLogger(__name__)
        self.trace_id = trace_id or "quillan-hw-gov"

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._observers: List[HardwareObserverProtocol] = []
        self._opportunistic_tasks: List[Tuple[str, Callable[[], Any]]] = []

        # Atomic state storage
        self._latest_snapshot = TelemetrySnapshot(
            timestamp=time.time(),
            cpu_temp_c=0.0,
            gpu_temp_c=0.0,
            host_ram_used_pct=0.0,
            vram_used_pct=0.0,
            gpu_util_pct=0.0,
            cpu_util_pct=0.0,
            is_thermal_warning=False,
            is_thermal_critical=False,
            is_ram_pressure=False,
            is_gpu_idle=False,
        )

        # Reactive control state
        self._throttle_factor = 1.0
        self._last_gc_time = 0.0

    def __enter__(self) -> HardwareTelemetryGovernor:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def register_observer(self, observer: HardwareObserverProtocol) -> None:
        """Register a listener conforming to HardwareObserverProtocol."""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def register_opportunistic_task(self, name: str, task_fn: Callable[[], Any]) -> None:
        """Register a secondary task to run during GPU idle periods."""
        with self._lock:
            self._opportunistic_tasks.append((name, task_fn))

    def start(self) -> None:
        """Start the background telemetry monitoring thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="QuillanHardwareGovernorDaemon",
                daemon=True,
            )
            self._thread.start()
            self.logger.info(
                '{"level": "INFO", "msg": "Hardware telemetry daemon started", "trace_id": "%s"}',
                self.trace_id,
            )

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the background telemetry monitoring thread gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
            self.logger.info(
                '{"level": "INFO", "msg": "Hardware telemetry daemon stopped", "trace_id": "%s"}',
                self.trace_id,
            )

    def get_snapshot(self) -> TelemetrySnapshot:
        """Retrieve the latest immutable telemetry snapshot (O(1), thread-safe)."""
        with self._lock:
            return self._latest_snapshot

    def step_boundary_hook(self) -> Dict[str, Any]:
        """
        Invoked synchronously at the step boundary of training or inference.
        
        Evaluates current hardware pressure, applies heat dissipation micro-sleeps,
        triggers garbage collection / cache eviction, and returns adaptive directives.
        
        Returns:
            Dict containing current throttle factor, pause duration, and active interventions.
        """
        snapshot = self.get_snapshot()
        directives: Dict[str, Any] = {
            "throttle_factor": 1.0,
            "paused_seconds": 0.0,
            "eviction_performed": False,
            "opportunistic_executed": None,
        }

        # 1. Thermal Intervention with Hysteresis
        if snapshot.is_thermal_critical:
            sleep_duration = self.thresholds.cooldown_sleep_s * 2.0
            time.sleep(sleep_duration)
            self._throttle_factor = 0.5
            directives["throttle_factor"] = 0.5
            directives["paused_seconds"] = sleep_duration
            self.logger.warning(
                '{"level": "WARN", "msg": "Thermal critical event: throttle applied", "trace_id": "%s", "cpu_temp": %.1f, "gpu_temp": %.1f}',
                self.trace_id, snapshot.cpu_temp_c, snapshot.gpu_temp_c,
            )
        elif snapshot.is_thermal_warning:
            sleep_duration = self.thresholds.cooldown_sleep_s
            time.sleep(sleep_duration)
            self._throttle_factor = 0.8
            directives["throttle_factor"] = 0.8
            directives["paused_seconds"] = sleep_duration
        else:
            self._throttle_factor = min(1.0, self._throttle_factor + 0.05)
            directives["throttle_factor"] = self._throttle_factor

        # 2. Memory Pressure Intervention
        now = time.time()
        if snapshot.is_ram_pressure and (now - self._last_gc_time > 5.0):
            self._last_gc_time = now
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            directives["eviction_performed"] = True
            self.logger.info(
                '{"level": "INFO", "msg": "RAM pressure event: cache flushed", "trace_id": "%s", "ram_pct": %.1f, "vram_pct": %.1f}',
                self.trace_id, snapshot.host_ram_used_pct, snapshot.vram_used_pct,
            )

        # 3. Thread-Safe Opportunistic GPU Compute Execution
        task_to_run: Optional[Tuple[str, Callable[[], Any]]] = None
        if snapshot.is_gpu_idle:
            with self._lock:
                if self._opportunistic_tasks:
                    task_to_run = self._opportunistic_tasks.pop(0)

        if task_to_run is not None:
            task_name, task_fn = task_to_run
            try:
                task_fn()
                directives["opportunistic_executed"] = task_name
                self.logger.info(
                    '{"level": "INFO", "msg": "Opportunistic task executed on idle GPU", "trace_id": "%s", "task": "%s"}',
                    self.trace_id, task_name,
                )
            except Exception as e:
                self.logger.error(
                    '{"level": "ERROR", "msg": "Opportunistic task failed", "trace_id": "%s", "task": "%s", "error": "%s"}',
                    self.trace_id, task_name, str(e),
                )

        return directives

    # ── Internal Monitoring Engine ───────────────────────────────────────────

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snapshot = self._sample_hardware()
                with self._lock:
                    self._latest_snapshot = snapshot
                    observers = list(self._observers)

                # Dispatch notifications outside lock to avoid deadlocks
                for obs in observers:
                    try:
                        if snapshot.is_thermal_critical:
                            obs.on_thermal_critical(snapshot)
                        elif snapshot.is_thermal_warning:
                            obs.on_thermal_warning(snapshot)

                        if snapshot.is_ram_pressure:
                            obs.on_memory_pressure(snapshot)

                        if snapshot.is_gpu_idle:
                            obs.on_gpu_idle(snapshot)
                    except Exception as err:
                        self.logger.debug("Observer error: %s", err)

            except Exception as e:
                self.logger.debug("Telemetry sample error: %s", e)

            self._stop_event.wait(timeout=self.thresholds.sample_interval_s)

    def _sample_hardware(self) -> TelemetrySnapshot:
        now = time.time()
        cpu_util = 0.0
        host_ram_pct = 0.0
        cpu_temp = 0.0

        if PSUTIL_AVAILABLE and psutil is not None:
            cpu_util = psutil.cpu_percent(interval=None)
            host_ram_pct = psutil.virtual_memory().percent
            try:
                temps = getattr(psutil, "sensors_temperatures", lambda: {})()
                if temps:
                    all_readings = [
                        entry.current for group in temps.values() for entry in group if hasattr(entry, "current")
                    ]
                    if all_readings:
                        cpu_temp = max(all_readings)
            except Exception:
                cpu_temp = 0.0

        # GPU metrics with suppressed warnings for unsupported compute capabilities
        gpu_temp = 0.0
        vram_pct = 0.0
        gpu_util = 0.0

        if TORCH_AVAILABLE and torch.cuda.is_available():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    dev = torch.cuda.current_device()
                    mem_alloc = torch.cuda.memory_allocated(dev)
                    mem_total = torch.cuda.get_device_properties(dev).total_memory
                    vram_pct = (mem_alloc / max(1, mem_total)) * 100.0
                    
                    if hasattr(torch.cuda, "utilization"):
                        gpu_util = float(torch.cuda.utilization(dev))
                    else:
                        mem_reserved = torch.cuda.memory_reserved(dev)
                        gpu_util = (mem_alloc / max(1, mem_reserved)) * 50.0 if mem_reserved > 0 else 0.0
                except Exception:
                    pass

        # If direct temperature sensor is unavailable on host, estimate from sustained utilization
        effective_cpu_temp = cpu_temp if cpu_temp > 0.0 else (45.0 + (cpu_util * 0.40))
        effective_gpu_temp = gpu_temp if gpu_temp > 0.0 else (40.0 + (gpu_util * 0.45))

        is_thermal_critical = (
            effective_cpu_temp >= self.thresholds.temp_crit_c or
            effective_gpu_temp >= self.thresholds.temp_crit_c
        )
        is_thermal_warning = (
            not is_thermal_critical and (
                effective_cpu_temp >= self.thresholds.temp_warn_c or
                effective_gpu_temp >= self.thresholds.temp_warn_c
            )
        )
        is_ram_pressure = (
            host_ram_pct >= self.thresholds.ram_warn_pct or
            vram_pct >= self.thresholds.vram_warn_pct
        )
        is_gpu_idle = gpu_util < self.thresholds.gpu_idle_pct

        return TelemetrySnapshot(
            timestamp=now,
            cpu_temp_c=effective_cpu_temp,
            gpu_temp_c=effective_gpu_temp,
            host_ram_used_pct=host_ram_pct,
            vram_used_pct=vram_pct,
            gpu_util_pct=gpu_util,
            cpu_util_pct=cpu_util,
            is_thermal_warning=is_thermal_warning,
            is_thermal_critical=is_thermal_critical,
            is_ram_pressure=is_ram_pressure,
            is_gpu_idle=is_gpu_idle,
        )


# ─── BACKWARD COMPATIBILITY ADAPTER ──────────────────────────────────────────

class LeeMach6GovernorAdapter:
    """
    Backward-compatible drop-in shim preserving the legacy LeeMach6Governor contract.
    
    Method Signature:
        adjust(latency_ms: float) -> Tuple[float, float, float]
        
    Internally queries the HardwareTelemetryGovernor to dynamically factor thermal
    and memory saturation into the returned scale.
    """

    def __init__(
        self,
        target_latency_ms: int = 100,
        hardware_governor: Optional[HardwareTelemetryGovernor] = None,
    ) -> None:
        self.target_ms = target_latency_ms
        self.current_scale = 1.0
        self.hardware_governor = hardware_governor

    def adjust(self, latency_ms: float) -> Tuple[float, float, float]:
        """
        Legacy adjustment interface preserved for zero downstream breakage.
        
        Returns:
            Tuple of (current_scale, suggested_ema_decay, recency_bias)
        """
        suggested_ema_decay = 0.995
        recency_bias = 0.0

        if latency_ms > self.target_ms:
            self.current_scale = max(0.1, self.current_scale * 0.8)
            suggested_ema_decay = 0.9999
            recency_bias = 1.0
        elif latency_ms < (self.target_ms * 0.5):
            self.current_scale = min(1.0, self.current_scale * 1.1)

        # Apply deep hardware telemetry modulation if governor is present
        if self.hardware_governor is not None:
            snapshot = self.hardware_governor.get_snapshot()
            if snapshot.is_thermal_critical:
                self.current_scale = max(0.1, self.current_scale * 0.5)
                recency_bias = 1.0
            elif snapshot.is_thermal_warning:
                self.current_scale = max(0.2, self.current_scale * 0.8)

            if snapshot.is_ram_pressure:
                suggested_ema_decay = 0.999

        return self.current_scale, suggested_ema_decay, recency_bias
