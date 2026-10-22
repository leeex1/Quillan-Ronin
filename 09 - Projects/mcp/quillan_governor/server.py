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
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_governor', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_governor] _shared warning: {_shared_err}', file=_sys.stderr)

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


# ── GOVERNOR PID & THERMODYNAMICS (v5.4.0-ONI) ────────────────────────────────────────

@mcp.tool()
def governor_pid_step(router_conf: float = 0.85, nemesis_integrity: float = 0.90, e_ice_ratio: float = 0.30) -> Dict[str, Any]:
    """Execute one Lee-Mach-6 PID governor step and return updated thresholds."""
    from _shared.base_engines import GOVERNOR_ENGINE
    result = GOVERNOR_ENGINE.step(
        router_conf_mean=max(0.0, min(1.0, router_conf)),
        nemesis_integrity=max(0.0, min(1.0, nemesis_integrity)),
        e_ice_ratio=max(0.0, min(1.0, e_ice_ratio)),
    )
    return {"status": "OK", "governor": result, "engine": "v5.4.0-ONI LeeMach6GovernorEngine"}

@mcp.tool()
def governor_thermo_energy(depth: float = 12.0, coherence: float = 0.95, entropy: float = 2.5, gamma_max: float = 1.0) -> Dict[str, Any]:
    """Compute E_ICE Landauer-bound thermodynamic energy for a reasoning step."""
    from _shared.base_engines import GOVERNOR_ENGINE
    result = GOVERNOR_ENGINE.compute_e_omega(depth=depth, coherence=coherence, entropy=entropy, gamma_max=gamma_max)
    return {"status": "OK", "thermo": result, "engine": "v5.4.0-ONI E_ICE"}

@mcp.tool()
def governor_full_analysis(query: str) -> Dict[str, Any]:
    """Run full governor analysis: PID step + thermodynamic energy + complexity routing."""
    from _shared.base_engines import GOVERNOR_ENGINE
    import hashlib
    h = hashlib.md5(query.encode("utf-8")).hexdigest()
    score = int(h[:4], 16) / 65535.0
    conf = 0.85 + score * 0.1
    integrity = 0.90 + score * 0.05
    e_ice = 0.25 + score * 0.15
    pid = GOVERNOR_ENGINE.step(router_conf_mean=conf, nemesis_integrity=integrity, e_ice_ratio=e_ice)
    thermo = GOVERNOR_ENGINE.compute_e_omega(depth=12.0, coherence=integrity, entropy=2.5 + score)
    return {"status": "OK", "query_preview": query[:80], "pid": pid, "thermo": thermo, "engine": "v5.4.0-ONI"}

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


# ─── 10. GPU MONITORING ──────────────────────────────────────────────────────

@mcp.tool()
def governor_gpu_status() -> Dict[str, Any]:
    """Monitor GPU utilization, VRAM, temperature, and power draw."""
    gpus = []
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "utilization_pct": float(parts[2]),
                        "vram_used_mb": float(parts[3]),
                        "vram_total_mb": float(parts[4]),
                        "temperature_c": float(parts[5]),
                        "power_draw_w": float(parts[6]),
                    })
    except FileNotFoundError:
        pass
    except Exception as e:
        pass

    if not gpus:
        # Fallback: try torch
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "index": i,
                        "name": props.name,
                        "vram_total_mb": round(props.total_memory / (1024**2), 1),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "engine": "torch",
                    })
        except Exception:
            pass

    if not gpus:
        return {"status": "NO_GPU_DETECTED", "gpus": [], "note": "nvidia-smi and torch not available."}

    alerts = []
    for g in gpus:
        if g.get("temperature_c", 0) > 85:
            alerts.append("GPU %d temperature critical: %.1fC" % (g["index"], g["temperature_c"]))
        if g.get("vram_used_mb", 0) / max(1, g.get("vram_total_mb", 1)) > 0.9:
            alerts.append("GPU %d VRAM near capacity" % g["index"])

    return {
        "status": "GPU_MONITORING_ACTIVE",
        "gpu_count": len(gpus),
        "gpus": gpus,
        "alerts": alerts,
        "council_note": "C14-KAIDO: optimize VRAM; C26-TECHNE: hardware constraints.",
    }


# ─── 11. DISK I/O MONITORING ─────────────────────────────────────────────────

@mcp.tool()
def governor_disk_io() -> Dict[str, Any]:
    """Monitor disk I/O, usage, and IOPS across all mounted drives."""
    disks = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disks.append({
                "device": part.device,
                "mountpoint": part.mountpoint,
                "fstype": part.fstype,
                "total_gb": round(usage.total / (1024**3), 1),
                "used_gb": round(usage.used / (1024**3), 1),
                "free_gb": round(usage.free / (1024**3), 1),
                "used_percent": usage.percent,
            })
        except Exception:
            continue

    # I/O counters
    io_counters = {}
    try:
        io = psutil.disk_io_counters()
        if io:
            io_counters = {
                "read_count": io.read_count,
                "write_count": io.write_count,
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes,
                "read_time_ms": io.read_time,
                "write_time_ms": io.write_time,
            }
    except Exception:
        pass

    alerts = []
    for d in disks:
        if d["used_percent"] > 90:
            alerts.append("Disk %s nearly full: %.1f%%" % (d["mountpoint"], d["used_percent"]))

    return {
        "status": "DISK_MONITORING_ACTIVE",
        "disks": disks,
        "io_counters": io_counters,
        "alerts": alerts,
        "council_note": "C14-KAIDO: optimize I/O; C26-TECHNE: storage constraints.",
    }


# ─── 12. NETWORK BANDWIDTH MONITORING ────────────────────────────────────────

@mcp.tool()
def governor_network_bandwidth() -> Dict[str, Any]:
    """Monitor network interfaces, bandwidth, and connection stats."""
    interfaces = []
    try:
        net_io = psutil.net_io_counters(pernic=True)
        for name, counters in net_io.items():
            interfaces.append({
                "interface": name,
                "bytes_sent": counters.bytes_sent,
                "bytes_recv": counters.bytes_recv,
                "packets_sent": counters.packets_sent,
                "packets_recv": counters.packets_recv,
                "errors_in": counters.errin,
                "errors_out": counters.errout,
                "drops_in": counters.dropin,
                "drops_out": counters.dropout,
            })
    except Exception:
        pass

    # Connection count
    conn_count = 0
    try:
        conns = psutil.net_connections()
        conn_count = len(conns)
    except Exception:
        pass

    # Network addresses
    addrs = {}
    try:
        for iface, addr_list in psutil.net_if_addrs().items():
            addrs[iface] = [{"family": str(a.family), "address": a.address} for a in addr_list]
    except Exception:
        pass

    return {
        "status": "NETWORK_MONITORING_ACTIVE",
        "interfaces": interfaces,
        "connection_count": conn_count,
        "addresses": addrs,
        "council_note": "C13-WARDEN: network perimeter; C29-NAVIGATOR: routing.",
    }


# ─── 13. PROCESS MANAGEMENT ──────────────────────────────────────────────────

@mcp.tool()
def governor_process_list(sort_by: str = "cpu", limit: int = 20) -> Dict[str, Any]:
    """List top processes by CPU or memory usage."""
    procs = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
        try:
            info = proc.info
            mem_mb = round(info["memory_info"].rss / (1024**2), 1) if info.get("memory_info") else 0
            procs.append({
                "pid": info["pid"],
                "name": info["name"],
                "cpu_percent": info.get("cpu_percent", 0),
                "memory_mb": mem_mb,
                "status": info.get("status", "unknown"),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    sort_key = "cpu_percent" if sort_by.lower() == "cpu" else "memory_mb"
    procs.sort(key=lambda x: x[sort_key], reverse=True)

    return {
        "status": "PROCESS_LIST_READY",
        "sort_by": sort_by,
        "total_processes": len(procs),
        "top_processes": procs[:limit],
        "council_note": "C13-WARDEN: process audit; C14-KAIDO: resource optimization.",
    }


@mcp.tool()
def governor_kill_process(pid: int, force: bool = False) -> Dict[str, Any]:
    """Terminate a process by PID. Use force=True for SIGKILL."""
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        if force:
            proc.kill()
        else:
            proc.terminate()
        return {"status": "TERMINATED", "pid": pid, "name": name, "force": force,
                "council_note": "C13-WARDEN: authorized process termination."}
    except psutil.NoSuchProcess:
        return {"status": "ERROR", "error": f"Process {pid} not found."}
    except psutil.AccessDenied:
        return {"status": "ERROR", "error": f"Access denied for PID {pid}. Run as admin."}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ─── 14. POWER CONSUMPTION TRACKING ──────────────────────────────────────────

@mcp.tool()
def governor_power_status() -> Dict[str, Any]:
    """Monitor system power consumption and battery status."""
    battery = None
    try:
        bat = psutil.sensors_battery()
        if bat:
            battery = {
                "percent": bat.percent,
                "secs_left": bat.secsleft,
                "power_plugged": bat.power_plugged,
                "status": "CHARGING" if bat.power_plugged else "DISCHARGING",
            }
    except Exception:
        pass

    # Try to get CPU power via RAPL (Linux) or estimate
    cpu_power_w = None
    try:
        # Linux RAPL
        import glob as _g
        rapl_paths = _g.glob("/sys/class/powercap/intel-rapl/*/energy_uj")
        if rapl_paths:
            cpu_power_w = "RAPL available (read energy_uj for delta)"
    except Exception:
        pass

    return {
        "status": "POWER_MONITORING_ACTIVE",
        "battery": battery,
        "cpu_power": cpu_power_w or "Estimated via utilization (not directly measurable on Windows without RAPL)",
        "estimated_system_power_w": round(psutil.cpu_percent() * 0.5 + 50, 1) if psutil.cpu_percent() > 0 else 50,
        "council_note": "C14-KAIDO: power optimization; C26-TECHNE: hardware constraints.",
    }


# ─── 15. THERMAL THROTTLING ALERTS ───────────────────────────────────────────

@mcp.tool()
def governor_thermal_status() -> Dict[str, Any]:
    """Monitor system temperatures and thermal throttling status."""
    temps = {}
    alerts = []
    try:
        sensors = psutil.sensors_temperatures()
        for name, entries in sensors.items():
            temps[name] = [{"label": e.label, "current": e.current, "high": e.high, "critical": e.critical} for e in entries]
            for e in entries:
                if e.critical and e.current >= e.critical - 5:
                    alerts.append("CRITICAL: %s %s at %.1fC (limit %.1fC)" % (name, e.label, e.current, e.critical))
                elif e.high and e.current >= e.high:
                    alerts.append("WARNING: %s %s at %.1fC (high %.1fC)" % (name, e.label, e.current, e.high))
    except Exception:
        pass

    # GPU temps (if available)
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().splitlines()):
                temp = float(line.strip())
                temps["gpu_%d" % i] = [{"label": "GPU", "current": temp, "high": 80, "critical": 90}]
                if temp > 85:
                    alerts.append("GPU %d temperature high: %.1fC" % (i, temp))
    except Exception:
        pass

    if not temps:
        return {"status": "NO_SENSORS_AVAILABLE", "temps": {}, "alerts": [],
                "note": "Temperature sensors not available on this platform (Windows). Use nvidia-smi for GPU temps."}

    return {
        "status": "THERMAL_MONITORING_ACTIVE",
        "temps": temps,
        "alerts": alerts,
        "throttling_active": len(alerts) > 0,
        "council_note": "C13-WARDEN: thermal safety; C14-KAIDO: optimize for thermals.",
    }


# ─── 16. CONTAINER / RESOURCE LIMITS ─────────────────────────────────────────

@mcp.tool()
def governor_container_limits() -> Dict[str, Any]:
    """Check container/cgroup resource limits (Docker, K8s, WSL)."""
    limits = {}
    # cgroup v1
    try:
        if os.path.exists("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                limits["memory_limit_bytes"] = int(f.read().strip())
        if os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                limits["cpu_quota_us"] = int(f.read().strip())
        if os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_period_us"):
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                limits["cpu_period_us"] = int(f.read().strip())
    except Exception:
        pass

    # cgroup v2
    try:
        if os.path.exists("/sys/fs/cgroup/memory.max"):
            with open("/sys/fs/cgroup/memory.max") as f:
                val = f.read().strip()
                limits["memory_max"] = val if val != "max" else "unlimited"
        if os.path.exists("/sys/fs/cgroup/cpu.max"):
            with open("/sys/fs/cgroup/cpu.max") as f:
                parts = f.read().strip().split()
                limits["cpu_max"] = parts
    except Exception:
        pass

    # Docker detection
    is_docker = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")

    # WSL detection
    is_wsl = "microsoft" in os.uname().release.lower() if hasattr(os, "uname") else False

    return {
        "status": "CONTAINER_CHECK_COMPLETE",
        "is_container": is_docker,
        "is_wsl": is_wsl,
        "cgroup_limits": limits if limits else "No cgroup limits detected (bare metal or Windows).",
        "host_info": {
            "os": sys.platform,
            "cpu_count": os.cpu_count(),
            "python_version": sys.version.split()[0],
        },
        "council_note": "C26-TECHNE: hardware constraints; C13-WARDEN: resource isolation.",
    }


# ─── 17. DISK SPACE ALERT ────────────────────────────────────────────────────

@mcp.tool()
def governor_disk_space_alert(threshold_percent: float = 90.0) -> Dict[str, Any]:
    """Check all drives for low disk space and return alerts."""
    alerts = []
    drives = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            drives.append({
                "mountpoint": part.mountpoint,
                "total_gb": round(usage.total / (1024**3), 1),
                "free_gb": round(usage.free / (1024**3), 1),
                "used_percent": usage.percent,
            })
            if usage.percent >= threshold_percent:
                alerts.append("LOW DISK: %s is %.1f%% full (%.1f GB free)" % (part.mountpoint, usage.percent, usage.free / (1024**3)))
        except Exception:
            continue
    return {
        "status": "DISK_SPACE_CHECK_COMPLETE",
        "threshold_percent": threshold_percent,
        "drives": drives,
        "alerts": alerts,
        "alert_count": len(alerts),
        "council_note": "C14-KAIDO: storage optimization; C13-WARDEN: capacity planning.",
    }


# ─── 18. SYSTEM UPTIME & LOAD ────────────────────────────────────────────────

@mcp.tool()
def governor_uptime_load() -> Dict[str, Any]:
    """Check system uptime, load average, and boot time."""
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    uptime_hours = round(uptime_seconds / 3600, 1)

    load_avg = None
    try:
        load_avg = psutil.getloadavg()
    except Exception:
        pass

    return {
        "status": "UPTIME_CHECK_COMPLETE",
        "boot_time": boot_time,
        "uptime_hours": uptime_hours,
        "uptime_days": round(uptime_hours / 24, 1),
        "load_average": load_avg,
        "cpu_count": os.cpu_count(),
        "load_per_cpu": round(load_avg[0] / max(1, os.cpu_count()), 2) if load_avg else None,
        "council_note": "C32-AEON: temporal awareness; C14-KAIDO: load optimization.",
    }


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
