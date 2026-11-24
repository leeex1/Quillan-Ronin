#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
booster_bridge.py -- Opt-in bridge: host governor  <->  sm_61 booster kernel.

WHY THIS EXISTS
---------------
The box has two halves that were never connected:

  1. HOST CONTROL: quillan_governor_daemon.py computes a live `gov_scale`
     (0.70..1.00) from Core-0 load + foreground-window detection and writes
     it to governor_state.json every 10s.

  2. GPU MATH: quillan_cuda_sm61.dll exposes bitnet_gemv_dp4a_kernel and
     fused_34expert_rank8_kernel (real Pascal sm_61 __dp4a instructions),
     dispatched by quillan_cuda_dispatcher.py via D.install(model).

The model's QuillanBlock.forward already accepts `gov_scale`, but no runtime
caller ever passes the live value, and no inference entry point ever calls
D.install(model). Both fast paths are orphaned.

This module closes both gaps WITHOUT touching model files, MCP config, or
any daemon. It is strictly opt-in: import is inert, nothing runs until the
calling session explicitly calls install().

DESIGN INVARIANTS
-----------------
- No torch / CUDA / ctypes import at module import time.
- No background threads, no timers, no daemons, no file writes.
- governor_state.json is read on demand (cached 2s) with a neutral fallback.
- If any dependency is missing, install() returns False and the model runs
  its original code path. Failure is always safe.
- Reversible: uninstall() restores both the dispatcher patches and the
  forward wrapper installed here.

USAGE (from the model-owning session, one line):
    import sys; sys.path.insert(0, r"C:\\02_QUILLAN\\kernels")
    import booster_bridge
    booster_bridge.install(model)      # returns True if both paths active

    # ... inference ...

    booster_bridge.uninstall()         # restore original forwards
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

__all__ = ["get_live_gov_scale", "install", "uninstall", "status"]

_REPO = Path(r"C:\02_QUILLAN")
_GOV_STATE = _REPO / "07 - Memory & LanceDB" / "governor_state.json"
_DISPATCHER_DIR = _REPO / "03 - Training & Model" / "scripts"

_CACHE_SECONDS = 2.0
_cache_value: float = 1.0
_cache_stamp: float = 0.0

_dispatcher: Any = None
_model: Any = None
_original_forward: Any = None
_installed: bool = False


def _read_governor_state() -> dict[str, Any]:
    """Read governor_state.json defensively. Never raises."""
    try:
        raw = _GOV_STATE.read_text(encoding="ascii")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def get_live_gov_scale() -> float:
    """Return the governor's live scale, clamped, with neutral fallback.

    Missing/stale/corrupt state -> 1.0 (full throughput), because an
    absent signal must never silently throttle inference.
    """
    global _cache_value, _cache_stamp
    now = time.monotonic()
    if _cache_stamp and (now - _cache_stamp) < _CACHE_SECONDS:
        return _cache_value

    state = _read_governor_state()
    # FRESHNESS GATE 2026-11-22: a dead governor latches its last gov_scale
    # forever. Without this, a stale 0.90 silently costs 10% throughput.
    # Missing/invalid timestamp or age > 30s -> neutral 1.0 (full speed).
    try:
        ts = float(state.get("timestamp", 0.0))
    except (TypeError, ValueError):
        ts = 0.0
    age = time.time() - ts
    if ts <= 0 or age < 0 or age > 30.0:
        value = 1.0
    else:
        try:
            value = float(state.get("gov_scale", 1.0))
        except (TypeError, ValueError):
            value = 1.0
        if not (0.5 <= value <= 1.0):
            value = 1.0

    _cache_value = value
    _cache_stamp = now
    return value


def _wrap_forward(model: Any) -> bool:
    """Inject live gov_scale into QuillanBlock.forward only when accepted.

    Detects the keyword from the signature; if the rebuilt model drops it,
    this is a no-op rather than a TypeError at inference time.
    """
    global _original_forward
    import inspect

    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        params = inspect.signature(forward).parameters
        accepts = "gov_scale" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        return False
    if not accepts:
        return False

    _original_forward = forward

    def forward_with_governor(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("gov_scale", get_live_gov_scale())
        return _original_forward(*args, **kwargs)

    model.forward = forward_with_governor
    return True


def install(model: Any) -> bool:
    """Install both booster paths. Idempotent. Always safe on failure.

    Returns True only if the CUDA dispatcher installed successfully.
    Governor injection is best-effort and reported via status().
    """
    global _dispatcher, _model, _installed
    if _installed:
        return True

    if str(_DISPATCHER_DIR) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_DISPATCHER_DIR))

    try:
        import quillan_cuda_dispatcher as dispatcher
    except Exception:
        _dispatcher = None
        return False

    ok = False
    try:
        ok = bool(dispatcher.install(model))
    except Exception:
        ok = False

    _dispatcher = dispatcher if ok else None
    _model = model if ok else None

    if ok:
        _wrap_forward(model)
        _installed = True
    return ok


def uninstall() -> None:
    """Restore original forwards. Safe to call when not installed."""
    global _dispatcher, _model, _original_forward, _installed
    if _original_forward is not None and _model is not None:
        try:
            _model.forward = _original_forward
        except (AttributeError, TypeError):
            pass
    if _dispatcher is not None:
        try:
            _dispatcher.uninstall()
        except Exception:
            pass
    _dispatcher = None
    _model = None
    _original_forward = None
    _installed = False


def status() -> dict[str, Any]:
    """Non-mutating diagnostics."""
    state = _read_governor_state()
    return {
        "installed": _installed,
        "dispatcher_active": _dispatcher is not None,
        "forward_wrapped": _original_forward is not None,
        "governor_scale": state.get("gov_scale"),
        "governor_mode": state.get("smoothness_mode"),
        "governor_policy": state.get("autoboost_policy"),
        "governor_age_seconds": round(
            time.time() - float(state.get("timestamp", 0.0)), 1
        )
        if state.get("timestamp")
        else None,
        "module": "booster_bridge",
    }
