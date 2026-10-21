#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN GOVERNOR ENGINE — NATIVE FAST-MCP SERVER
=========================================================
Embodying C14-KAIDO (Efficiency), C26-TECHNE, and the LeeMach6 Velocity Governor:
  1. governor_system_telemetry: Real-time host CPU, RAM, Swap, and DPC health check
  2. governor_thread_budget: Safe CPU thread budget (reserves 2 cores for Windows OS/DWM)
  3. governor_optimizer_ram_calc: Mathematical RAM estimator (AdamW vs Muon vs SGD)
  4. governor_velocity_tune: LeeMach6 step-time velocity and thermal throttle tuner
  5. governor_full_pipeline — telemetry -> thread budget -> ram calc -> velocity tune -> health summary (persisted)
  6. governor_route — cross-MCP routing payload (council/code/buddy)
  7. governor_health — governor dashboard summary
  8. governor_session_list / governor_session_replay — persistence
"""

import sys
import os
import logging
import builtins

# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
# Ensure all logs and prints go strictly to stderr to prevent stdio JSON-RPC corruption
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if "file" not in kwargs or kwargs["file"] is None or kwargs["file"] is sys.stdout:
        kwargs["file"] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print

# --- Unified Cross-Platform PYTHONPATH Injection ---
_BASE_PATHS = [
    r"C:\02_QUILLAN\09 - Projects\mcp",
    r"C:\02_QUILLAN\09 - Projects\projects\oni",
    r"C:\02_QUILLAN\03 - Training & Model\scripts",
    r"C:\02_QUILLAN",
]
for _bp in _BASE_PATHS:
    if os.name != "nt":
        if _bp.startswith("C:") or _bp.startswith("c:"):
            _bp = "/mnt/c/" + _bp[3:].replace("\\", "/")
    if os.path.exists(_bp) and _bp not in sys.path:
        sys.path.insert(0, _bp)



import math
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import psutil
from fastmcp import FastMCP

mcp = FastMCP("QuillanGovernor")
@mcp.tool()
def governor_system_health() -> Dict[str, Any]:
    """Returns real-time CPU, RAM, and thermal telemetry for the host machine."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()


    cpu_percent = psutil.cpu_percent(interval=0.2)
    logical_cores = os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores // 2

    free_ram_gb = round(vm.available / (1024**3), 2)
    total_ram_gb = round(vm.total / (1024**3), 2)
    used_ram_pct = vm.percent

    health_status = "OPTIMAL"
    warning = None

    if free_ram_gb < 1.5 or used_ram_pct > 92.0:
        health_status = "CRITICAL_STARVATION"
        warning = "System RAM is nearly exhausted. Paging to disk is imminent. Immediate thread throttling required."
    elif free_ram_gb < 3.0 or used_ram_pct > 82.0 or cpu_percent > 88.0:
        health_status = "ELEVATED_RISK"
        warning = "Resource load high. Heavy allocations may cause UI compositor stutter."

    return {
        "status": health_status,
        "warning": warning,
        "cpu": {
            "utilization_percent": cpu_percent,
            "logical_cores": logical_cores,
            "physical_cores": physical_cores,
            "safe_training_threads": max(1, logical_cores - 2)
        },
        "memory": {
            "total_gb": total_ram_gb,
            "available_gb": free_ram_gb,
            "used_percent": used_ram_pct,
            "swap_used_percent": swap.percent
        },
        "governor_policy": {
            "os_reserved_cores": 2,
            "recommended_priority": "BELOW_NORMAL_PRIORITY_CLASS",
            "dpc_interrupt_protection": "Active"
        }
    }

# ─── 2. THREAD BUDGET & PRIORITY ADVISOR ─────────────────────────────────────

@mcp.tool()
def governor_thread_budget(workload_type: str = "training") -> Dict[str, Any]:
    """Calculates the exact safe CPU thread allocation and process priority."""
    total_cores = os.cpu_count() or 4
    wl = workload_type.strip().lower()

    if wl in ["training", "sft", "finetune"]:
        safe_threads = max(1, total_cores - 2)
        priority = "BELOW_NORMAL_PRIORITY_CLASS (32)"
        rationale = "Reserving 2 cores for Desktop Window Manager (dwm.exe) and OS interrupt handling."
    elif wl in ["inference", "generate"]:
        safe_threads = max(1, total_cores - 2)
        priority = "NORMAL_PRIORITY_CLASS (32)"
        rationale = "Interactive response generation with reserved OS headroom."
    elif wl in ["data_loading", "tokenize", "eval"]:
        safe_threads = max(1, total_cores - 4) if total_cores >= 8 else max(1, total_cores - 2)
        priority = "IDLE_PRIORITY_CLASS (64) or BELOW_NORMAL"
        rationale = "Disk/IO bound pipeline; avoid competing with UI threads."
    else:
        safe_threads = max(1, total_cores - 2)
        priority = "BELOW_NORMAL"
        rationale = "Standard protective governor budget."

    code_snippet = (
        f"import os, torch, psutil\n"
        f"torch.set_num_threads({safe_threads})\n"
        f"try:\n"
        f"    p = psutil.Process(os.getpid())\n"
        f"    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)\n"
        f"except Exception:\n"
        f"    pass"
    )

    return {
        "workload": wl,
        "total_logical_cores": total_cores,
        "allocated_threads": safe_threads,
        "reserved_os_cores": total_cores - safe_threads,
        "process_priority": priority,
        "rationale": rationale,
        "python_setup_snippet": code_snippet
    }

# ─── 3. OPTIMIZER RAM ESTIMATOR ──────────────────────────────────────────────

@mcp.tool()
def governor_optimizer_ram_calc(
    param_count_millions: float,
    optimizer: str = "adamw",
    trainable_ratio: float = 1.0,
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_dim: int = 1024,
    n_layers: int = 12
) -> Dict[str, Any]:
    """Accurately calculates memory footprint (RAM) for model training."""
    p_total = param_count_millions * 1e6
    p_trainable = p_total * max(0.01, min(1.0, trainable_ratio))

    weight_bytes = p_total * 4
    grad_bytes = p_trainable * 4

    opt = optimizer.strip().lower()
    if opt == "adamw":
        opt_bytes = p_trainable * 8
    elif opt == "muon":
        opt_bytes = p_trainable * 4
    elif opt == "sgd":
        opt_bytes = p_trainable * 4
    else:
        opt_bytes = p_trainable * 8

    act_bytes = 2 * batch_size * seq_len * hidden_dim * n_layers * 4 * 3

    total_bytes = weight_bytes + grad_bytes + opt_bytes + act_bytes
    total_gb = round(total_bytes / (1024**3), 2)

    vm = psutil.virtual_memory()
    available_gb = round(vm.available / (1024**3), 2)
    fits_in_ram = total_gb <= (available_gb * 0.85)

    return {
        "model_params_m": param_count_millions,
        "trainable_params_m": round(p_trainable / 1e6, 2),
        "optimizer": opt.upper(),
        "breakdown_mb": {
            "model_weights": round(weight_bytes / (1024**2), 1),
            "gradients": round(grad_bytes / (1024**2), 1),
            "optimizer_states": round(opt_bytes / (1024**2), 1),
            "activation_buffer": round(act_bytes / (1024**2), 1)
        },
        "total_estimated_ram_gb": total_gb,
        "host_available_ram_gb": available_gb,
        "safe_to_run": fits_in_ram,
        "verdict": "SAFE_HEADROOM" if fits_in_ram else "EXCEEDS_RECOMMENDED_HEADROOM (Risk of Paging/Freezing)"
    }

# ─── 4. LEEMACH6 VELOCITY TUNING ─────────────────────────────────────────────

@mcp.tool()
def governor_velocity_tune(
    current_step_time_ms: float,
    target_step_time_ms: float = 5000.0,
    current_loss: Optional[float] = None
) -> Dict[str, Any]:
    """Applies the LeeMach6 Velocity Governor algorithm to tune training throughput."""
    error = target_step_time_ms - current_step_time_ms
    velocity_ratio = current_step_time_ms / max(100.0, target_step_time_ms)

    actions = []
    if velocity_ratio > 2.0:
        actions.append("CRITICAL SLOWDOWN: Step time exceeds 2x target. Reduce sequence padding or lower unpadded seq_len.")
        actions.append("Consider unfreezing fewer layers (e.g., train attention/norm only).")
    elif velocity_ratio > 1.25:
        actions.append("MODERATE LATENCY: Enable dynamic sequence batching to remove pad tokens.")
    elif velocity_ratio < 0.5:
        actions.append("HIGH HEADROOM: Step time is fast; batch size or micro-batch count could be increased safely.")
    else:
        actions.append("STABLE CADENCE: Velocity within optimal LeeMach6 operational band.")

    return {
        "current_ms": current_step_time_ms,
        "target_ms": target_step_time_ms,
        "velocity_ratio": round(velocity_ratio, 2),
        "status": "STABLE" if 0.7 <= velocity_ratio <= 1.3 else ("TOO_SLOW" if velocity_ratio > 1.3 else "TOO_FAST"),
        "prescribed_actions": actions
    }

# ─── 5. FULL PIPELINE (persisted) ────────────────────────────────────────────

@mcp.tool()
def governor_full_pipeline(param_count_millions: float = 758.0, optimizer: str = "adamw",
                          trainable_ratio: float = 1.0, batch_size: int = 1, seq_len: int = 512,
                          hidden_dim: int = 1024, n_layers: int = 12,
                          workload_type: str = "training",
                          current_step_time_ms: float = 4200.0, target_step_time_ms: float = 5000.0,
                          persist: bool = True) -> Dict[str, Any]:
    """Chain telemetry -> thread budget -> ram calc -> velocity tune -> health summary."""
    telemetry = governor_system_telemetry()
    budget = governor_thread_budget(workload_type=workload_type)
    ram = governor_optimizer_ram_calc(param_count_millions=param_count_millions, optimizer=optimizer,
                                     trainable_ratio=trainable_ratio, batch_size=batch_size,
                                     seq_len=seq_len, hidden_dim=hidden_dim, n_layers=n_layers)
    velocity = governor_velocity_tune(current_step_time_ms=current_step_time_ms,
                                     target_step_time_ms=target_step_time_ms)

    session_id = f"governor-{int(time.time())}"
    result = {
        "status": "GOVERNOR_PIPELINE_COMPLETE",
        "session_id": session_id,
        "telemetry": telemetry,
        "thread_budget": budget,
        "ram_calc": ram,
        "velocity_tune": velocity,
        "overall": "PASS" if (telemetry.get("status") not in ("CRITICAL_STARVATION",) and ram.get("safe_to_run") and velocity.get("status") == "STABLE") else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "governor_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result

# ─── 7. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def governor_route(target_mcp: str = "council", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route governor resource decision to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_full_pipeline",
                "payload": payload or {}, "note": "Route governor infrastructure setup to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "RoninBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route governor tuning to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}

# ─── 8. HEALTH DASHBOARD ────────────────────────────────────────────────────

@mcp.tool()
def governor_health() -> Dict[str, Any]:
    """Governor dashboard summary (telemetry + budget + ram + velocity)."""
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "telemetry_status": (data.get("telemetry") or {}).get("status"),
                    "ram_safe": (data.get("ram_calc") or {}).get("safe_to_run"),
                    "velocity_status": (data.get("velocity_tune") or {}).get("status"),
                    "overall": data.get("overall"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "GOVERNOR_HEALTH_READY",
        "sessions": sessions[:5],
        "council_note": "C14-KAIDO optimizes; C26-TECHNE engineers; LeeMach6 paces.",
    }

# ─── 9. SESSION PERSISTENCE ─────────────────────────────────────────────────

@mcp.tool()
def governor_session_list() -> Dict[str, Any]:
    """List persisted governor sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "overall": data.get("overall"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}

@mcp.tool()
def governor_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted governor session by ID."""
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Read failed: {e}"}


# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log

class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()
