#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan hybrid runtime — single source of truth for compute placement.

EVERYTHING (train, distill student, inference, RAG embed, benchmarks) calls
hybrid_runtime.init() at startup. Policy lives in
configs/hybrid_compute.json and is only ever changed by measured benchmark
output (oni/toolkit_benchmark.py), never by hand-tuning.

Current honest policy (GTX 1050 sm_61 + torch cu130):
  compute=CPU, optimizer=CPU, vGPU split=0.0 (all-CPU).
GPU compute unlocks only when cuda_status.sm_61_supported_by_torch flips,
proven by the benchmark's real CUDA probe — not by editing this file.
"""
import json
import os
import sys
from pathlib import Path

HYBRID_CONFIG = Path(r"C:\02_QUILLAN\00 - Meta\configs\hybrid_compute.json")


def load_hybrid_config(path: Path = HYBRID_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_ram_headroom(min_free_gb: float) -> dict:
    """Returns measured RAM state; warns (never kills) when below floor."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        free_gb = vm.available / (1024 ** 3)
        return {
            "total_gb": round(vm.total / (1024 ** 3), 1),
            "free_gb": round(free_gb, 2),
            "ok": free_gb >= min_free_gb,
        }
    except Exception as e:
        return {"total_gb": 0.0, "free_gb": 0.0, "ok": True, "note": f"psutil unavailable: {e}"}


def ensure_offload_dirs(cfg: dict) -> None:
    for key in ("root", "activations", "optimizer"):
        Path(cfg["offload"][key]).mkdir(parents=True, exist_ok=True)


def init() -> dict:
    """Apply hybrid policy process-wide. Call once at startup. Returns cfg."""
    import torch

    cfg = load_hybrid_config()
    t = cfg["torch"]
    torch.set_num_threads(int(t["num_threads"]))
    torch.set_num_interop_threads(int(t["num_interop_threads"]))
    try:
        torch.set_float32_matmul_precision(t.get("matmul_precision", "high"))
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = bool(t.get("cudnn_benchmark", True))
    except Exception:
        pass

    ensure_offload_dirs(cfg)

    ram = check_ram_headroom(float(cfg["host"]["ram_min_free_gb"]))
    if not ram["ok"]:
        print(f"[HYBRID] WARNING: RAM free {ram['free_gb']}GB < floor "
              f"{cfg['host']['ram_min_free_gb']}GB — free memory before training "
              f"(total {ram['total_gb']}GB). Continuing, no processes touched.",
              flush=True)
    else:
        print(f"[HYBRID] mode={cfg['mode']} compute={cfg['routing']['train_compute']} "
              f"threads={torch.get_num_threads()} ram_free={ram['free_gb']}GB "
              f"vgpu_split_gpu={cfg['vgpu']['split_ratio_gpu']}", flush=True)
    return cfg


def resolve_device(requested: str, cfg: dict) -> str:
    """Map a requested device through hybrid policy.

    'cuda' is honored ONLY when the benchmark has proven sm_61 torch support.
    Otherwise it falls back to CPU with a loud log line (never silent).
    """
    if requested is None or requested == "cpu":
        return "cpu"
    if cfg["cuda_status"].get("sm_61_supported_by_torch"):
        return requested
    print(f"[HYBRID] CUDA requested but torch lacks sm_61 support "
          f"(torch {__import__('torch').version.cuda}) — falling back to CPU. "
          f"See cuda_status in hybrid_compute.json.", flush=True)
    return "cpu"


if __name__ == "__main__":
    c = init()
    print(json.dumps({k: c[k] for k in ("mode", "routing", "vgpu")}, indent=2))
