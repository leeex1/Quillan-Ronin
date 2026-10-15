#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4.0 — 3-STAGE PROGRESSIVE CURRICULUM TRAINING ORCHESTRATOR
=============================================================================
Sequentially trains Quillan through the 3 standardized epochs:
  - Epoch 1: Identity & Foundation (19.10M tokens)
  - Epoch 2: Instruction Following & Dialogue (13.75M tokens)
  - Epoch 3: Math, Code, STEM & High-Tier System Architecture (17.82M tokens)

Carries forward weights and optimizer states seamlessly across all 3 stages.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import argparse
import time
from pathlib import Path
import torch

REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
SCRIPTS_DIR = MODEL_DIR / "scripts"
DATA_DIR = MODEL_DIR / "training_data" / "canonical_standardized"
CKPT_DIR = MODEL_DIR / "checkpoints" / "checkpoints_oni"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(SCRIPTS_DIR), str(MODEL_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_train_pipeline import QuillanTrainingOrchestrator

STAGES_MAP = [
    {
        "epoch": 1,
        "name": "Identity & Foundation Pre-training",
        "file": DATA_DIR / "stage1_identity_foundation.pt",
        "default_steps": 250,
        "lr": 2.5e-5,
    },
    {
        "epoch": 2,
        "name": "Instruction Following & Conversational Alignment",
        "file": DATA_DIR / "stage2_instructions_alignment.pt",
        "default_steps": 250,
        "lr": 2.0e-5,
    },
    {
        "epoch": 3,
        "name": "Math, Code, STEM & System Architecture",
        "file": DATA_DIR / "stage3_math_code_science.pt",
        "default_steps": 250,
        "lr": 1.5e-5,
    }
]

def main():
    parser = argparse.ArgumentParser(description="Quillan 3-Stage Progressive Curriculum Runner")
    parser.add_argument("--layers", type=int, choices=[6, 12], default=6, help="Model depth: 6 (Mini) or 12 (Main)")
    parser.add_argument("--steps-per-stage", type=int, default=100, help="Number of training steps per curriculum epoch (default: 100)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per forward pass (default: 1 for max throughput)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--resume-stage", type=int, default=1, choices=[1, 2, 3], help="Start from stage 1, 2, or 3")
    args = parser.parse_args()

    print("=" * 72)
    print("  👑 QUILLAN-RONIN: 3-STAGE CURRICULUM TRAINING PIPELINE")
    print("=" * 72)
    print(f"[*] Target Architecture : QuillanRoninOni ({args.layers} Layers)")
    print(f"[*] Steps per Stage     : {args.steps_per_stage} steps")
    print(f"[*] Total Steps Planned : {args.steps_per_stage * 3} steps")
    print(f"[*] Batch Config        : batch_size={args.batch_size}, seq_len={args.seq_len}, grad_accum={args.grad_accum}")
    print(f"[*] Checkpoint Output   : {CKPT_DIR}")

    orchestrator = QuillanTrainingOrchestrator(n_layer=args.layers, device_str="cpu", router_mode="dense_pull")

    # Resume from existing checkpoint if available
    latest_ckpt = CKPT_DIR / f"quillan_{args.layers}l_latest.pt"
    if latest_ckpt.exists():
        print(f"[*] Resuming from existing baseline: {latest_ckpt.name}...")
        orchestrator.load_checkpoint(latest_ckpt)

    total_start_time = time.time()

    for stage_info in STAGES_MAP:
        epoch_num = stage_info["epoch"]
        if epoch_num < args.resume_stage:
            print(f"\n[>>>] Skipping Stage {epoch_num} (resume-stage={args.resume_stage})")
            continue

        stage_name = stage_info["name"]
        data_file = stage_info["file"]
        stage_lr = stage_info["lr"]

        print("\n" + "=" * 72)
        print(f"  🚀 STARTING EPOCH {epoch_num}/3: {stage_name.upper()}")
        print(f"     Dataset: {data_file.name}")
        print("=" * 72)

        if not data_file.exists():
            print(f"[!] Error: Stage dataset {data_file} not found. Skipping.")
            continue

        stage_ckpt_out = CKPT_DIR / f"quillan_{args.layers}l_curriculum_epoch{epoch_num}.pt"

        res = orchestrator.run_training_loop(
            steps=args.steps_per_stage,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=stage_lr,
            warmup_steps=max(1, args.steps_per_stage // 5),
            grad_accum_steps=args.grad_accum,
            aux_alpha=0.01,
            eval_interval=max(10, args.steps_per_stage // 4),
            patience=3,
            checkpoint_dir=CKPT_DIR,
            data_file=data_file,
            optimizer_type="fused",
            lr_muon=0.015,
            ccrl_limit=4.0,
            val_split=0.10,
        )

        # Save stage checkpoint
        orchestrator.save_checkpoint(stage_ckpt_out, orchestrator.global_step, res["final_loss"])
        print(f"\n[✓] Completed Epoch {epoch_num}: Final Loss = {res['final_loss']:.4f}")
        print(f"[✓] Stage Checkpoint Saved: {stage_ckpt_out.name}")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 72)
    print(f"  👑 ALL 3 CURRICULUM EPOCHS COMPLETE IN {total_elapsed / 60:.1f} MINUTES!")
    print("=" * 72)

    # Final native export
    final_qbin = CKPT_DIR / f"quillan_{args.layers}l_curriculum_final.qbin"
    orchestrator.export_native_qbin(final_qbin)
    print(f"[✓] Native C++ Binary Exported: {final_qbin}")

if __name__ == "__main__":
    main()
