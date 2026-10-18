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
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import faulthandler
faulthandler.enable()
import gc
import argparse
from pathlib import Path
import time
import torch

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
SCRIPTS_DIR = MODEL_DIR / "scripts"
DATA_DIR = MODEL_DIR / "training_data" / "canonical_standardized"
CKPT_DIR = MODEL_DIR / "checkpoints" / "checkpoints_oni"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(SCRIPTS_DIR), str(MODEL_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_train_pipeline import QuillanTrainingOrchestrator, NativeQuillanExporter

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
    parser.add_argument("--steps-per-stage", type=int, default=0, help="Number of training steps per stage. Set to 0 for automatic FULL EPOCH calculation based on dataset size.")
    parser.add_argument("--epochs-per-stage", type=float, default=1.0, help="Number of full dataset epochs per stage if --steps-per-stage is 0 (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per forward pass (default: 1 for max throughput)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15 evaluations before halting; prevents premature stopping)")
    parser.add_argument("--eval-interval", type=int, default=0, help="Evaluation interval in steps. Set to 0 for auto-calculation.")
    parser.add_argument("--resume-stage", type=int, default=1, choices=[1, 2, 3], help="Start from stage 1, 2, or 3")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom baseline checkpoint to initialize or resume from")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save trained checkpoints (e.g. on external drive)")
    parser.add_argument("--reset-step", action="store_true", help="Reset global step to 0 for fresh curriculum baseline")
    args = parser.parse_args()

    global CKPT_DIR
    if args.checkpoint_dir:
        CKPT_DIR = Path(args.checkpoint_dir)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  👑 QUILLAN-RONIN: 3-STAGE CURRICULUM TRAINING PIPELINE")
    print("=" * 72)
    print(f"[*] Target Architecture : QuillanRoninOni ({args.layers} Layers)")
    print(f"[*] Mode                : {'Full Epoch (' + str(args.epochs_per_stage) + 'x)' if args.steps_per_stage <= 0 else str(args.steps_per_stage) + ' steps/stage'}")
    print(f"[*] Batch Config        : batch_size={args.batch_size}, seq_len={args.seq_len}, grad_accum={args.grad_accum}")
    print(f"[*] Checkpoint Output   : {CKPT_DIR}")

    orchestrator = QuillanTrainingOrchestrator(n_layer=args.layers, device_str="cpu", router_mode="dense_pull")

    if args.checkpoint:
        resume_ckpt = Path(args.checkpoint)
        print(f"[*] Using explicitly specified baseline: {resume_ckpt}...")
        orchestrator.load_checkpoint(resume_ckpt)
        if args.reset_step:
            orchestrator.global_step = 0
            print("[*] Global step reset to 0 for fresh curriculum training.")
    else:
        # Resume from existing checkpoint if available (prefer best.pt if more advanced)
        best_ckpt = CKPT_DIR / f"quillan_{args.layers}l_best.pt"
        latest_ckpt = CKPT_DIR / f"quillan_{args.layers}l_latest.pt"
        resume_ckpt = None
        if latest_ckpt.exists():
            resume_ckpt = latest_ckpt
        elif best_ckpt.exists():
            resume_ckpt = best_ckpt

        if resume_ckpt is not None:
            print(f"[*] Resuming from existing baseline: {resume_ckpt.name}...")
            orchestrator.load_checkpoint(resume_ckpt)

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
        print(f"  🚀 STAGE {epoch_num}/3: {stage_name.upper()}")
        print(f"     Dataset: {data_file.name}")
        print("=" * 72)

        if not data_file.exists():
            print(f"[!] Error: Stage dataset {data_file} not found. Skipping.")
            continue

        # Load metadata/tensor to compute true epoch step count
        d_raw = torch.load(data_file, map_location="cpu", weights_only=False)
        if isinstance(d_raw, dict) and "input_ids" in d_raw:
            num_samples = len(d_raw["input_ids"])
            total_tokens = d_raw.get("meta", {}).get("total_tokens", num_samples * args.seq_len)
        elif isinstance(d_raw, torch.Tensor) and d_raw.dim() == 2:
            num_samples = len(d_raw)
            total_tokens = num_samples * d_raw.shape[1]
        else:
            num_samples = 50000
            total_tokens = num_samples * args.seq_len
        del d_raw
        gc.collect()

        # Compute full-epoch steps or use explicit override
        effective_batch = args.batch_size * args.grad_accum
        steps_for_one_epoch = max(1, num_samples // effective_batch)
        if args.steps_per_stage > 0:
            stage_steps = args.steps_per_stage
            print(f"[*] Running explicit {stage_steps:,} steps ({stage_steps * effective_batch * args.seq_len:,} tokens | {stage_steps / steps_for_one_epoch:.2%} of 1 epoch)")
        else:
            stage_steps = max(1, int(steps_for_one_epoch * args.epochs_per_stage))
            print(f"[*] AUTO FULL EPOCH: {stage_steps:,} steps ({args.epochs_per_stage:.1f} epoch(s) = {num_samples:,} samples = {total_tokens:,} tokens)")

        eval_interval = args.eval_interval if args.eval_interval > 0 else min(100, max(25, stage_steps // 50))
        warmup_steps = min(100, max(5, stage_steps // 20))

        stage_ckpt_out = CKPT_DIR / f"quillan_{args.layers}l_curriculum_epoch{epoch_num}.pt"

        res = orchestrator.run_training_loop(
            steps=stage_steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=stage_lr,
            warmup_steps=warmup_steps,
            grad_accum_steps=args.grad_accum,
            aux_alpha=0.01,
            eval_interval=eval_interval,
            save_interval=50,
            patience=args.patience,
            checkpoint_dir=CKPT_DIR,
            data_path=data_file,
            optimizer_type="fused",
            lr_muon=0.015,
            ccrl_limit=4.0,
            val_split=0.10,
        )

        # Save stage checkpoint
        orchestrator.save_checkpoint(stage_ckpt_out, orchestrator.global_step, res["final_loss"])
        print(f"\n[✓] Completed Stage {epoch_num}: Final Loss = {res['final_loss']:.4f}")
        print(f"[✓] Stage Checkpoint Saved: {stage_ckpt_out.name}")

        # Prune superseded step checkpoints to maintain ample disk headroom
        for old_step_file in CKPT_DIR.glob(f"quillan_{args.layers}l_step_*.pt"):
            try:
                old_step_file.unlink()
            except Exception:
                pass

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 72)
    print(f"  👑 ALL 3 CURRICULUM EPOCHS COMPLETE IN {total_elapsed / 60:.1f} MINUTES!")
    print("=" * 72)

    # Final native export
    final_qbin = CKPT_DIR / f"quillan_{args.layers}l_curriculum_final.qbin"
    NativeQuillanExporter.export(orchestrator.model, orchestrator.cfg, final_qbin)
    print(f"[✓] Native C++ Binary Exported: {final_qbin}")

if __name__ == "__main__":
    main()
