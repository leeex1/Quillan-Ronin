#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_6l_gold_finetune.py — Precision Instruction & Conversational Alignment Fine-Tuning
Fine-tunes the 6-Layer model on quillan_gold_sft_master.pt with unfrozen token embeddings (wte/wpe)
to achieve natural sentence coherence and conversational fluency.
"""
import sys, os, time, math, gc, shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
SCRIPTS_DIR = MODEL_DIR / "scripts"
DATA_DIR = MODEL_DIR / "training_data" / "canonical_standardized"
CKPT_DIR = Path(r"E:\quillan_checkpoints_oni")

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(MODEL_DIR))

import torch
from quillan_train_pipeline import QuillanTrainingOrchestrator, NativeQuillanExporter

def main():
    print("=" * 72)
    print("  [*] QUILLAN-RONIN: 6-LAYER GOLD CONVERSATIONAL FINE-TUNING")
    print("=" * 72)

    data_file = DATA_DIR / "quillan_gold_sft_master.pt"
    if not data_file.exists():
        raise FileNotFoundError(f"Gold SFT dataset not found: {data_file}")

    trinity_ckpt = Path(r"E:\quillan_merged_models\quillan_trinity_base_6l.pt")
    if trinity_ckpt.exists():
        baseline_ckpt = trinity_ckpt
    else:
        baseline_ckpt = CKPT_DIR / "quillan_6l_curriculum_epoch3.pt"
        if not baseline_ckpt.exists():
            baseline_ckpt = CKPT_DIR / "quillan_6l_best.pt"
    print(f"[*] Baseline Checkpoint : {baseline_ckpt}")
    print(f"[*] Alignment Dataset   : {data_file.name}")
    print(f"[*] Checkpoint Output   : {CKPT_DIR}")

    # Initialize orchestrator
    orchestrator = QuillanTrainingOrchestrator(n_layer=6, device_str="cpu", router_mode="dense_pull")

    # Load baseline model weights
    orchestrator.load_checkpoint(baseline_ckpt)

    # UNFREEZE word token embeddings (wte) and position embeddings (wpe)
    # This enables token-level gradient adaptation to conversational English
    print("[*] Unfreezing wte (token embeddings) and wpe (position embeddings)...")
    for name, p in orchestrator.model.named_parameters():
        if "wte" in name or "wpe" in name:
            p.requires_grad = True

    trainable = sum(p.numel() for p in orchestrator.model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in orchestrator.model.parameters() if not p.requires_grad)
    print(f"[*] Active Trainable Params : {trainable / 1e6:.2f}M | Frozen: {frozen / 1e6:.2f}M")

    out_ckpt = CKPT_DIR / "quillan_6l_gold_sft_aligned.pt"
    steps = 100

    print(f"[*] Launching {steps}-step precision alignment run (batch_size=1, seq_len=256, grad_accum=2)...")
    t0 = time.time()

    res = orchestrator.run_training_loop(
        steps=steps,
        batch_size=1,
        seq_len=256,
        lr=2.0e-5,
        warmup_steps=10,
        grad_accum_steps=2,
        aux_alpha=0.01,
        eval_interval=25,
        save_interval=25,
        patience=20,
        checkpoint_dir=CKPT_DIR,
        data_path=data_file,
        optimizer_type="fused",
        lr_muon=0.015,
        ccrl_limit=4.0,
        val_split=0.10,
    )

    elapsed = time.time() - t0
    orchestrator.save_checkpoint(out_ckpt, orchestrator.global_step, res["final_loss"])
    print("\n" + "=" * 72)
    print(f"[✓] Gold Conversational Alignment Complete in {elapsed / 60:.1f} minutes!")
    print(f"[✓] Final SFT Loss: {res['final_loss']:.4f}")
    print(f"[✓] Saved Aligned Checkpoint: {out_ckpt.name}")
    print("=" * 72)

    # Export C++ binary
    out_qbin = CKPT_DIR / "quillan_6l_gold_sft_final.qbin"
    NativeQuillanExporter.export(orchestrator.model, orchestrator.cfg, out_qbin)
    print(f"[✓] Exported Aligned Native C++ Binary: {out_qbin}")

if __name__ == "__main__":
    main()
