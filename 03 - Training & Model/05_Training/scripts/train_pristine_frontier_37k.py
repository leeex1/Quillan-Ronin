#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — PRISTINE FRONTIER 37K MASTER SFT TRAINER
Trains the 12-Layer Unrolled Sovereign Architecture directly on the 37,463 pristine SFT samples
in c:\\02_QUILLAN\\training_data\\pristine_frontier_gold_37k.pt.

Features:
  - 37,463 diverse multi-domain samples (Math, CS, Physics, Systems, Algorithms, Logic)
  - Zero token noise or prompt leak
  - Target Masking (labels=-100 on prompt tokens)
  - Memory-efficient SGD with momentum over 223M trainable parameters
  - Multi-threaded CPU execution (4 threads)
"""

import os
import sys
import time
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
DATA_DIR = Path(r"C:\02_QUILLAN\training_data")
CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def log_msg(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def train_pristine_37k():
    log_msg("==================================================================")
    log_msg("   👑 PRISTINE FRONTIER 37K MULTI-DOMAIN SFT")
    log_msg("==================================================================")

    device = torch.device("cpu")
    cfg = QuillanUnrolledConfig()
    model = QuillanUnrolledSovereign(cfg).to(device)

    # Resume from clean direct factual checkpoint
    resume_ckpt = CKPT_DIR / "quillan_direct_factual_best.pt"
    if resume_ckpt.exists():
        log_msg(f"Resuming from baseline: {resume_ckpt.name}")
        sd = torch.load(str(resume_ckpt), map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd)
        model.load_state_dict(sd, strict=False)

    # Freeze base unrolled weights, train LoRAs, Underling Swarms, Bridges, Prism, LayerNorms
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        if any(k in name for k in ['lora', 'swarm', 'expert_swarms', 'q1_bridge', 'q2_bridge', 'ingest_gate', 'prism', 'ln_']):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False

    log_msg(f"Trainable Parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.1f}%)")

    dataset_path = DATA_DIR / "pristine_frontier_gold_37k.pt"
    log_msg(f"Loading pristine dataset from {dataset_path.name}...")
    data_dict = torch.load(str(dataset_path), map_location="cpu", weights_only=False)
    input_ids_all = data_dict["input_ids"]
    labels_all = data_dict["labels"]
    num_samples = input_ids_all.size(0)
    log_msg(f"Loaded {num_samples:,} pristine sequence pairs (seq_len={input_ids_all.size(1)}).")

    num_steps = 300
    peak_lr = 2.0e-2
    min_lr = 1.0e-3
    weight_decay = 0.01
    grad_clip = 1.0

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=peak_lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=min_lr)

    model.train()
    best_loss = float("inf")
    running_loss = 0.0
    steps_accum = 0
    t0 = time.time()

    log_msg(f"Starting Pristine 37K Pass ({num_steps} steps across {num_samples:,} samples)...")

    for step in range(1, num_steps + 1):
        idx = random.randint(0, num_samples - 1)
        inp = input_ids_all[idx : idx + 1].to(device)
        lbl = labels_all[idx : idx + 1].to(device)

        optimizer.zero_grad()
        _, loss = model(inp, labels=lbl)

        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        running_loss += loss_val
        steps_accum += 1

        if step % 10 == 0 or step == num_steps:
            avg_loss = running_loss / steps_accum
            elapsed = time.time() - t0
            sps = step / max(0.001, elapsed)
            lr_curr = scheduler.get_last_lr()[0]

            log_msg(
                f"Step {step:3d}/{num_steps} | Loss: {avg_loss:.4f} | LR: {lr_curr:.2e} | GNorm: {gnorm.item():.2f} | {sps:.2f} st/s"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_ckpt_path = CKPT_DIR / "quillan_pristine_37k_best.pt"
                torch.save({
                    "step": step,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                }, str(best_ckpt_path))
                log_msg(f"--> Saved pristine 37K checkpoint (Loss: {best_loss:.4f}) to {best_ckpt_path.name}")

            running_loss = 0.0
            steps_accum = 0

    final_ckpt_path = CKPT_DIR / "quillan_pristine_37k_final.pt"
    torch.save({
        "step": num_steps,
        "loss": best_loss,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, str(final_ckpt_path))

    log_msg("==================================================================")
    log_msg(f"   🏆 PRISTINE 37K SFT COMPLETE (Best Loss: {best_loss:.4f})")
    log_msg("==================================================================")

if __name__ == "__main__":
    train_pristine_37k()
