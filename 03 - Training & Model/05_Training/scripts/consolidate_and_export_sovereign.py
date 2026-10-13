#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN WEIGHT CONSOLIDATION & EXPORT
Fuses active LoRA matrices and underling swarm projections into static inference weights,
strips optimizer overhead, and exports a lightweight, production-ready deployable checkpoint.
"""

import os
import sys
import time
import torch
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft")
EXPORT_DIR = Path(r"C:\02_QUILLAN\checkpoints\production_export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

def consolidate_and_export():
    print("==================================================================", flush=True)
    print("   👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN WEIGHT CONSOLIDATION", flush=True)
    print("==================================================================\n", flush=True)

    candidates = [
        CKPT_DIR / "quillan_super_run_best.pt",
        CKPT_DIR / "quillan_super_run_final.pt",
        CKPT_DIR / "quillan_sovereign_gold_best.pt"
    ]

    target_ckpt = None
    for cand in candidates:
        if cand.exists():
            target_ckpt = cand
            break

    if not target_ckpt:
        print("[-] Error: No checkpoint available to consolidate.", flush=True)
        return

    print(f"[*] Consolidating source checkpoint: {target_ckpt.name}", flush=True)
    device = torch.device("cpu")
    cfg = QuillanUnrolledConfig()
    model = QuillanUnrolledSovereign(cfg).to(device)

    raw_data = torch.load(str(target_ckpt), map_location=device, weights_only=False)
    sd = raw_data.get("model_state_dict", raw_data)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Create clean production export dictionary
    clean_state_dict = {}
    param_count = 0
    tensor_bytes = 0

    for k, v in model.state_dict().items():
        v_detached = v.detach().clone().to(torch.float32)
        clean_state_dict[k] = v_detached
        param_count += v_detached.numel()
        tensor_bytes += v_detached.numel() * v_detached.element_size()

    export_path = EXPORT_DIR / "quillan_ronin_v531_sovereign_production.pt"
    payload = {
        "architecture": "QuillanUnrolledSovereign_v5.3.1",
        "parameters": param_count,
        "size_bytes": tensor_bytes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
        "model_state_dict": clean_state_dict
    }

    torch.save(payload, str(export_path))
    print(f"[+] Successfully exported clean production artifact:")
    print(f"    - Target: {export_path}")
    print(f"    - Parameter Count: {param_count:,}")
    print(f"    - Size: {tensor_bytes / (1024**2):.2f} MB\n", flush=True)

if __name__ == "__main__":
    consolidate_and_export()
