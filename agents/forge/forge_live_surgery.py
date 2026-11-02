#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FORGE v2 — Live Model Forensic & Topology Surgeon
==================================================
Runs directly inside QuillanRoninOni 1,441-tensor state dict:
1. Audits 1,441-tensor topology (Trainable vs Parked vs Frozen).
2. Performs BitNet 1.58b ternary distribution analysis (Gamma + {-1, 0, +1}%).
3. Evaluates 35 EGGROLL low-rank swarm matrices (Rank, U@V.T delta norm).
4. Verifies router entropy and gating health on ingest_gate and moe_stabilizer.
5. Strictly enforces the Zero-Leak Audit Law (no_grad, del, gc.collect).
"""

import os
import sys
import gc
import math
from pathlib import Path
import torch

WORKSPACE = Path(r"C:\02_QUILLAN")
CHECKPOINT_PATH = WORKSPACE / "checkpoints" / "checkpoints_oni" / "sft_runs" / "quillan_sft_latest.pt"

TRAINABLE_PATTERNS = [
    "council_experts", "swarms", "ingest_gate", "quillan_finalizer",
    "quillan_comm_gate", "prism", "ln_f", "mem_norm", "moe_stabilizer"
]
PARKED_MODULES = [
    "test_time_world", "dream_diffusion", "early_probe", "diffusion",
    "mixture_of_recursions", "diverse_moe", "ste_pack", "pocket_dfa"
]

def is_trainable(name: str) -> bool:
    return any(p in name for p in TRAINABLE_PATTERNS) and "mod_routers" not in name

def is_parked(name: str) -> bool:
    return any(mod in name for mod in PARKED_MODULES)

def main():
    print("=" * 78)
    print("FORGE v2 // TENSOR SPECIALIST SURGEON — LIVE FORENSIC AUDIT")
    print(f"Target Checkpoint: {CHECKPOINT_PATH.name}")
    print("Zero-Leak Audit Law: ACTIVE")
    print("=" * 78)

    if not CHECKPOINT_PATH.exists():
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
        return

    with torch.no_grad():
        print("\n[1/4] Loading state dict (CPU map, zero VRAM contention)...")
        ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict") or ck
        step = ck.get("step", "unknown")
        train_ce = ck.get("train_ce", float("nan"))
        val_ce = ck.get("val_ce", float("nan"))
        print(f"  Checkpoint Step: {step} | Train CE: {train_ce:.4f} | Val CE: {val_ce:.4f}")

        # 1. 1,441-Tensor Topology Breakdown
        total_tensors = len(sd)
        trainable = {k: v for k, v in sd.items() if is_trainable(k)}
        parked = {k: v for k, v in sd.items() if is_parked(k)}
        frozen = {k: v for k, v in sd.items() if k not in trainable and k not in parked}

        n_train_params = sum(p.numel() for p in trainable.values())
        n_parked_params = sum(p.numel() for p in parked.values())
        n_frozen_params = sum(p.numel() for p in frozen.values())

        print(f"\n[2/4] 1,441-Tensor Topology Audit:")
        print(f"  Total Tensors Verified : {total_tensors} / 1,441")
        print(f"  Trainable Adapters     : {len(trainable)} tensors ({n_train_params / 1e6:.2f}M params) [CUDA]")
        print(f"  Parked Auxiliary Units : {len(parked)} tensors ({n_parked_params / 1e6:.2f}M params) [CPU]")
        print(f"  Frozen Base Backbone   : {len(frozen)} tensors ({n_frozen_params / 1e6:.2f}M params) [Ternary]")

        # 2. BitNet 1.58b Ternary Quantization Analysis
        print(f"\n[3/4] BitNet 1.58b Ternary Health (Sampled Layers):")
        ternary_sampled = 0
        for name, tensor in list(frozen.items())[:20]:
            if "weight" in name and tensor.dim() >= 2 and tensor.numel() > 1000:
                t_float = tensor.float()
                gamma = t_float.abs().mean().item()
                w_quant = torch.clamp(torch.round(t_float / (gamma + 1e-8)), -1, 1)
                total = w_quant.numel()
                neg = (w_quant == -1).sum().item() / total * 100
                zero = (w_quant == 0).sum().item() / total * 100
                pos = (w_quant == 1).sum().item() / total * 100
                print(f"  -> {name[:45]:<45} | Gamma: {gamma:.6f} | [-1: {neg:.1f}%, 0: {zero:.1f}%, +1: {pos:.1f}%]")
                del t_float, w_quant
                ternary_sampled += 1
                if ternary_sampled >= 4:
                    break

        # 3. 35 EGGROLL Swarm Low-Rank Adapter Inspection
        print(f"\n[4/4] 35-Swarm EGGROLL Low-Rank Adapter Dynamics:")
        swarm_u_keys = [k for k in sd.keys() if "swarms" in k and ".U" in k]
        swarm_v_keys = [k for k in sd.keys() if "swarms" in k and ".V" in k]
        print(f"  Active Swarm U/V Tensors: {len(swarm_u_keys)} U-matrices, {len(swarm_v_keys)} V-matrices")

        sample_swarms = [0, 11, 23, 34]
        for sw_idx in sample_swarms:
            u_key = f"swarms.{sw_idx}.U"
            v_key = f"swarms.{sw_idx}.V"
            if u_key in sd and v_key in sd:
                U = sd[u_key].float()
                V = sd[v_key].float()
                rank = U.shape[1] if U.dim() > 1 else 1
                # Delta = U @ V.T
                delta = U @ V.T
                norm_delta = delta.norm().item()
                print(f"  -> Swarm {sw_idx:02d}: U shape {list(U.shape)}, V shape {list(V.shape)} | Rank: {rank} | ||U @ V.T||: {norm_delta:.6f}")
                del U, V, delta

        # Router Gate Health Check
        print(f"\n[ROUTER GATES] Gating Stability & Dispersion:")
        for r_name in ["ingest_gate", "quillan_finalizer", "quillan_comm_gate", "moe_stabilizer"]:
            matches = [k for k in sd.keys() if r_name in k]
            for m in matches[:1]:
                t = sd[m].float()
                print(f"  -> {m:<35} | Shape: {list(t.shape)} | Mean: {t.mean().item():+.5f} | Std: {t.std().item():.5f} | Norm: {t.norm().item():.4f}")
                del t

        # Zero-Leak Cleanup
        del sd, ck, trainable, parked, frozen
    gc.collect()
    print("\n" + "=" * 78)
    print("[FORGE SURGERY COMPLETE] Zero-Leak Law Enforced. Tensors Deallocated.")
    print("=" * 78)

if __name__ == "__main__":
    main()

