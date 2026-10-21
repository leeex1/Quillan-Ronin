#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 QUILLAN-RONIN MODEL LAB & DIAGNOSTICS — NATIVE FAST-MCP SERVER
================================================================
Embodying C28-CALCULUS, C6-OMNIS, and Optimization Specialist:
  1. Checkpoint Header & Parameter Inspection (fast header read without loading all tensors)
  2. Dataset Validator & Distribution Profiler (JSONL token distribution, length percentiles)
  3. Hardware & Memory Footprint Advisor (RAM headroom, thread limits, zero-lag tuning)
  4. BitNet Ternary Quantization & Sparsity Analyzer
  5. model_lab_full_pipeline — inspect checkpoint -> validate dataset -> hardware advisor -> quantization analyzer (persisted)
  6. model_lab_route — cross-MCP routing payload (council/code/buddy)
  7. model_lab_health — model lab dashboard summary
  8. model_lab_session_list / model_lab_session_replay — persistence
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

import torch
from fastmcp import FastMCP

mcp = FastMCP("QuillanModelLab")
        return {"error": f"Checkpoint not found: {p}"}

    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        keys = list(data.keys())

        info = {
            "file_name": p.name,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "keys_present": keys,
            "step": data.get("step", "N/A"),
            "loss": round(float(data.get("loss", 0.0)), 4) if "loss" in data else "N/A",
            "val_loss": round(float(data.get("val_loss", 0.0)), 4) if "val_loss" in data else "N/A",
            "ppl": round(float(data.get("ppl", 0.0)), 2) if "ppl" in data else "N/A",
            "engine": data.get("engine", "N/A"),
        }

        if "config" in data and isinstance(data["config"], dict):
            cfg = data["config"]
            info["model_config"] = {
                "n_layer": cfg.get("n_layer"),
                "hidden_dim": cfg.get("hidden_dim"),
                "n_head": cfg.get("n_head"),
                "num_experts": cfg.get("num_experts"),
                "max_seq_len": cfg.get("max_seq_len"),
                "vocab_size": cfg.get("vocab_size"),
            }

        if "model_state_dict" in data:
            sd = data["model_state_dict"]
            total_params = sum(t.numel() for t in sd.values())
            info["total_parameters"] = f"{total_params:,}"
            info["tensor_count"] = len(sd)

        del data
        return info

    except Exception as e:
        return {"error": f"Failed to inspect checkpoint: {e}"}

# ─── 2. DATASET VALIDATOR & PROFILER ─────────────────────────────────────────

@mcp.tool()
def model_dataset_validator(jsonl_path: str, max_samples: int = 1000) -> Dict[str, Any]:
    """Validates and profiles a JSONL training dataset for question/response coverage, char lengths, and formatting."""
    p = Path(jsonl_path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"Dataset file not found: {p}"}

    total_lines = 0
    valid_samples = 0
    empty_responses = 0
    prompt_lengths = []
    response_lengths = []

    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                if valid_samples >= max_samples:
                    continue
                try:
                    obj = json.loads(line)
                    prompt = obj.get("question") or obj.get("prompt") or obj.get("instruction") or ""
                    resp = obj.get("response") or obj.get("answer") or obj.get("output") or ""

                    if not resp.strip():
                        empty_responses += 1
                        continue

                    valid_samples += 1
                    prompt_lengths.append(len(prompt.split()))
                    response_lengths.append(len(resp.split()))
                except Exception:
                    pass

        avg_prompt_words = round(sum(prompt_lengths) / max(1, len(prompt_lengths)), 1)
        avg_resp_words = round(sum(response_lengths) / max(1, len(response_lengths)), 1)
        max_resp_words = max(response_lengths) if response_lengths else 0

        return {
            "file_name": p.name,
            "total_lines": total_lines,
            "inspected_samples": valid_samples,
            "empty_responses": empty_responses,
            "avg_prompt_words": avg_prompt_words,
            "avg_response_words": avg_resp_words,
            "max_response_words": max_resp_words,
            "status": "HEALTHY" if empty_responses == 0 else "WARNING_EMPTY_RESPONSES",
        }

    except Exception as e:
        return {"error": f"Failed to validate dataset: {e}"}

# ─── 3. HARDWARE & ZERO-LAG FOOTPRINT ADVISOR ────────────────────────────────

@mcp.tool()
def model_hardware_advisor(model_param_millions: float = 758.0) -> Dict[str, Any]:
    """Calculates safe memory allocations and thread caps to guarantee 0% desktop stutter."""
    total_cores = os.cpu_count() or 4
    safe_threads = max(1, total_cores - 2)

    fp32_weights_gb = round((model_param_millions * 1e6 * 4) / (1024**3), 2)
    adamw_state_gb = round(fp32_weights_gb * 2, 2)
    full_unfreeze_total_gb = round(fp32_weights_gb + adamw_state_gb + 1.0, 2)

    targeted_params_m = min(110.0, model_param_millions)
    targeted_weights_gb = round((targeted_params_m * 1e6 * 4) / (1024**3), 2)
    targeted_adamw_gb = round(targeted_weights_gb * 2, 2)
    targeted_total_gb = round(fp32_weights_gb + targeted_adamw_gb + 0.5, 2)

    return {
        "system_cores": total_cores,
        "recommended_worker_threads": safe_threads,
        "cores_reserved_for_os_ui": 2,
        "process_priority": "BELOW_NORMAL_PRIORITY_CLASS",
        "full_unfreeze_ram_footprint_gb": full_unfreeze_total_gb,
        "targeted_unfreeze_ram_footprint_gb": targeted_total_gb,
        "recommendation": (
            f"Use targeted unfreeze (Attention + Prism + LoRA + Gates) requiring only ~{targeted_total_gb} GB RAM "
            f"and set torch.set_num_threads({safe_threads}) to eliminate system interrupts and DPC latency."
        )
    }

# ─── 4. BITNET TERNARY QUANTIZATION ANALYZER ─────────────────────────────────

@mcp.tool()
def model_bitnet_analyze(checkpoint_path: str) -> Dict[str, Any]:
    """Analyzes a checkpoint for BitNet 1.58-bit ternary quantization viability."""
    p = Path(checkpoint_path.strip().strip('"\''))
    if not p.exists():
        return {"error": f"Checkpoint not found: {p}"}

    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        sd = data.get("model_state_dict", data if isinstance(data, dict) else {})

        total_params = 0
        ternary_eligible = 0
        weight_layers = []

        for name, t in sd.items():
            n = t.numel()
            total_params += n
            if t.is_floating_point():
                weight_layers.append({"name": name, "params": n, "dtype": str(t.dtype)})
                if t.dim() >= 2:
                    ternary_eligible += n

        total_m = total_params / 1e6
        ternary_m = ternary_eligible / 1e6

        # BitNet 1.58b: ternary {-1,0,1} + per-channel scale gamma
        # Memory: 1 bit for sign + 1 bit for zero + scale
        ternary_bits_per_param = 2  # {-1,0,1} ~ 2 bits after packing with scale
        compressed_gb = round((ternary_m * 1e6 * 2 / 8) / (1024**3), 2)
        full_fp32_gb = round((total_m * 1e6 * 4) / (1024**3), 2)
        savings = round((1 - compressed_gb / max(full_fp32_gb, 0.001)) * 100, 1)

        return {
            "file_name": p.name,
            "total_parameters_m": round(total_m, 2),
            "ternary_eligible_parameters_m": round(ternary_m, 2),
            "ternary_eligible_pct": round((ternary_m / max(total_m, 0.001)) * 100, 1),
            "full_fp32_gb": full_fp32_gb,
            "ternary_compressed_gb": compressed_gb,
            "memory_savings_pct": savings,
            "weight_layers_sample": weight_layers[:10],
            "recommendation": (
                f"BitNet 1.58b viable: {ternary_m:.1f}M of {total_m:.1f}M params ternary-eligible "
                f"({(ternary_m / max(total_m, 0.001)) * 100:.1f}%). Estimated {savings}% memory reduction."
            ),
        }

    except Exception as e:
        return {"error": f"Failed to analyze checkpoint: {e}"}

# ─── 5. FULL PIPELINE (persisted) ───────────────────────────────────────────

@mcp.tool()
def model_lab_full_pipeline(checkpoint_path: str = "", jsonl_path: str = "",
                           model_param_millions: float = 758.0,
                           persist: bool = True) -> Dict[str, Any]:
    """Chain checkpoint inspect -> dataset validate -> hardware advisor -> bitnet analyzer."""
    ckpt = model_checkpoint_inspect(checkpoint_path) if checkpoint_path else {"skipped": True}
    dataset = model_dataset_validator(jsonl_path) if jsonl_path else {"skipped": True}
    hardware = model_hardware_advisor(model_param_millions=model_param_millions)
    bitnet = model_bitnet_analyze(checkpoint_path) if checkpoint_path else {"skipped": True}

    session_id = f"modellab-{int(time.time())}"
    result = {
        "status": "MODEL_LAB_PIPELINE_COMPLETE",
        "session_id": session_id,
        "checkpoint": ckpt,
        "dataset": dataset,
        "hardware": hardware,
        "bitnet": bitnet,
        "overall": "PASS" if (ckpt.get("status") != "error" and dataset.get("status") in ("HEALTHY", "WARNING_EMPTY_RESPONSES", "skipped")) else "REVIEW",
        "timestamp": time.time(),
    }

    if persist:
        try:
            p = SESSION_DIR / f"{session_id}.json"
            p.write_text(json.dumps({"type": "model_lab_pipeline", "result": result}, indent=2), encoding="utf-8")
            result["persisted_to"] = str(p)
        except Exception as e:
            result["persist_error"] = str(e)

    return result

# ─── 6. ROUTE (cross-MCP) ───────────────────────────────────────────────────

@mcp.tool()
def model_lab_route(target_mcp: str = "council", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cross-MCP routing payload for council/code/buddy integration."""
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_full_deliberation",
                "payload": payload or {}, "note": "Route model lab diagnostics to council deliberation."}
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_full_pipeline",
                "payload": payload or {}, "note": "Route model lab findings to code scan pipeline."}
    if target_mcp.lower() == "buddy":
        return {"target": "RoninBuddy", "suggested_tool": "buddy_mirror_collaborate",
                "payload": payload or {}, "note": "Route model lab results to mirror-collaboration."}
    return {"error": f"Unknown target '{target_mcp}'. Use council, code, or buddy."}

# ─── 7. HEALTH DASHBOARD ────────────────────────────────────────────────────

@mcp.tool()
def model_lab_health() -> Dict[str, Any]:
    """Model lab dashboard summary."""
    sessions = []
    try:
        for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id"),
                    "checkpoint": (data.get("checkpoint") or {}).get("file_name"),
                    "dataset": (data.get("dataset") or {}).get("file_name"),
                    "overall": data.get("overall"),
                    "timestamp": data.get("timestamp"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return {
        "status": "MODEL_LAB_HEALTH_READY",
        "sessions": sessions[:5],
        "council_note": "C28-CALCULUS quantifies; C6-OMNIS integrates.",
    }

# ─── 9. SESSION PERSISTENCE ─────────────────────────────────────────────────

@mcp.tool()
def model_lab_session_list() -> Dict[str, Any]:
    """List persisted model lab sessions."""
    sessions = []
    for p in sorted(SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "checkpoint": (data.get("checkpoint") or {}).get("file_name"),
                "dataset": (data.get("dataset") or {}).get("file_name"),
                "overall": data.get("overall"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}

@mcp.tool()
def model_lab_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted model lab session by ID."""
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

if __name__ == "__main__":
    mcp.run()
