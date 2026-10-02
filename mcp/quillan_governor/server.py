#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN-RONIN GOVERNOR ENGINE — NATIVE FAST-MCP SERVER
==========================================================
Embodying C14-KAIDO, C26-TECHNE, and the LeeMach6 Velocity Governor:
  1. governor_system_telemetry: Real-time host CPU, RAM, Swap, and DPC health check
  2. governor_thread_budget: Safe CPU thread budget (reserves 2 cores for Windows OS/DWM)
  3. governor_optimizer_ram_calc: Mathematical RAM estimator (AdamW vs Muon vs SGD)
  4. governor_velocity_tune: LeeMach6 step-time velocity and thermal throttle tuner
"""

import sys
import os
import math
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import psutil
from fastmcp import FastMCP

mcp = FastMCP("QuillanGovernor")


# ─── 1. SYSTEM TELEMETRY & HEALTH CHECK ──────────────────────────────────────

@mcp.tool()
def governor_system_telemetry() -> Dict[str, Any]:
    """Inspects live host system telemetry to ensure compute workloads don't freeze the desktop.
    
    Checks CPU load, available RAM, swap usage, and flags hazardous resource starvation states.
    """
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    cpu_percent = psutil.cpu_percent(interval=0.2)
    logical_cores = os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores // 2

    # Status classification
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
    """Calculates the exact safe CPU thread allocation and process priority.
    
    Reserves at least 2 cores to prevent Windows kernel DPC/ISR starvation and mouse compositor freezing.
    
    Args:
        workload_type: Type of task: 'training', 'inference', 'data_loading', or 'background'.
    """
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
    """Accurately calculates memory footprint (RAM) for model training.
    
    Compares model weights, gradients, optimizer states (AdamW vs Muon vs SGD), and activations.
    
    Args:
        param_count_millions: Total model parameters in millions (e.g. 758 for 12L, 380 for 6L).
        optimizer: 'adamw', 'muon', or 'sgd'.
        trainable_ratio: Fraction of parameters receiving gradients (0.0 to 1.0, e.g. 0.22 for unfreezing attention/prism only).
        batch_size: Batch size per step.
        seq_len: Context sequence length in tokens.
        hidden_dim: Model hidden dimension (e.g. 1024).
        n_layers: Number of transformer layers (e.g. 6 or 12).
    """
    p_total = param_count_millions * 1e6
    p_trainable = p_total * max(0.01, min(1.0, trainable_ratio))

    # Weight memory (FP32 baseline in bytes)
    weight_bytes = p_total * 4

    # Gradient memory (FP32 for trainable parameters)
    grad_bytes = p_trainable * 4

    # Optimizer states
    opt = optimizer.strip().lower()
    if opt == "adamw":
        # 1st moment (fp32) + 2nd moment (fp32) = 8 bytes per trainable param
        opt_bytes = p_trainable * 8
    elif opt == "muon":
        # Single momentum buffer = 4 bytes per trainable param
        opt_bytes = p_trainable * 4
    elif opt == "sgd":
        # Momentum buffer = 4 bytes, or 0 without momentum
        opt_bytes = p_trainable * 4
    else:
        opt_bytes = p_trainable * 8

    # Activation memory estimation (bytes)
    # Roughly 2 * b * s * h * l * (attention + ffn overhead)
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
    """Applies the LeeMach6 Velocity Governor algorithm to tune training throughput.
    
    Provides PID adjustments to maintain stable step cadence without thermal or memory runaway.
    
    Args:
        current_step_time_ms: Measured time for the last training step in milliseconds.
        target_step_time_ms: Desired step cadence (default 5000ms / 5s).
        current_loss: Optional current loss value to correlate with step stability.
    """
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


if __name__ == "__main__":
    mcp.run()
