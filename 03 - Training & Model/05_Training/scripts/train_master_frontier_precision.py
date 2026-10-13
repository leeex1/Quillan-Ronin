#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — MASTER FRONTIER ACCELERATED PRECISION SFT
Trains the 12-Layer Unrolled Sovereign Architecture on 21,642 Master Harvested Samples:
  - 2,797 Advanced Coding, Algorithms & Distributed Systems (code_train.jsonl)
  - 8,648 Multi-Turn Science, Mathematics & Logic (instruct_train.jsonl)
  - 1,140 Quantum Mechanics, Physics & Derivations (quillan_science_absolute.jsonl)
  - 7,857 Council Expert Domain Slices (experts_34/)
  - 1,200 Direct Gold Precision Anchors

Optimizations:
  - CPU 4-Thread parallelization (torch.set_num_threads(4))
  - Sequence-length optimized for rapid convergence (max_len = 160)
  - Target Masking (ignore_index = -100 on prompt tokens)
  - Flushed 5-step telemetry logging and dynamic checkpointing
"""

import os
import sys
import math
import time
import json
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Enable multi-threaded CPU execution
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
from sovereign_inference_engine import SovereignTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def log_msg(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_master_dataset(tokenizer: SovereignTokenizer, data_path: Path, max_samples: int = 5000, max_seq_len: int = 160):
    log_msg(f"Loading master multi-horizon dataset from {data_path.name}...")
    tokenized_pairs = []
    
    with open(data_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
        
    random.seed(42)
    random.shuffle(lines)
    
    for line in lines[:max_samples]:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            p_text = item.get("prompt", "")
            r_text = item.get("response", "")
            if not p_text or not r_text:
                continue
                
            p_tokens = tokenizer.encode(p_text)[:64]
            max_r_len = max_seq_len - len(p_tokens)
            r_tokens = tokenizer.encode(r_text)[:max_r_len]
            
            total_tokens = p_tokens + r_tokens
            labels = [-100] * len(p_tokens) + list(r_tokens)
            
            if len(total_tokens) <= max_seq_len and len(total_tokens) == len(labels) and len(r_tokens) > 3:
                tokenized_pairs.append({
                    "input_ids": total_tokens,
                    "labels": labels
                })
        except Exception:
            continue
            
    log_msg(f"Loaded {len(tokenized_pairs)} target-masked master multi-horizon samples (max_seq_len={max_seq_len}).")
    return tokenized_pairs

def train_master_frontier():
    log_msg("==================================================================")
    log_msg("   👑 MASTER FRONTIER ACCELERATED PRECISION SFT (21,642 SAMPLES)")
    log_msg("==================================================================")

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanUnrolledConfig()

    model = QuillanUnrolledSovereign(cfg).to(device)

    resume_ckpt = CKPT_DIR / "quillan_direct_factual_best.pt"
    if not resume_ckpt.exists():
        resume_ckpt = CKPT_DIR / "quillan_gold_precision_best.pt"
        
    if resume_ckpt.exists():
        log_msg(f"Resuming from baseline: {resume_ckpt.name}")
        sd = torch.load(str(resume_ckpt), map_location=device, weights_only=False)
        sd = sd.get("model_state_dict", sd)
        model.load_state_dict(sd, strict=False)

    # Freeze base unrolled weights, train only LoRAs, Underling Swarms, Bridges, Prism, and LayerNorms
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

    master_dataset_file = DATA_DIR / "Quillan_Universal_100_Percent_Master_Gold.jsonl"
    dataset = load_master_dataset(tokenizer, master_dataset_file, max_samples=30000, max_seq_len=160)

    num_steps = 200
    peak_lr = 1.5e-2
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

    log_msg(f"Starting Master Frontier Pass ({num_steps} steps, Peak LR: {peak_lr:.2e}, Threads: 4)...")

    for step in range(1, num_steps + 1):
        sample = random.choice(dataset)
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([sample["labels"]], dtype=torch.long, device=device)

        optimizer.zero_grad()
        _, loss = model(input_ids, labels=labels)
        
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        running_loss += loss_val
        steps_accum += 1

        if step % 5 == 0 or step == num_steps:
            avg_loss = running_loss / steps_accum
            elapsed = time.time() - t0
            sps = step / max(0.001, elapsed)
            lr_curr = scheduler.get_last_lr()[0]
            
            log_msg(
                f"Step {step:3d}/{num_steps} | Loss: {avg_loss:.4f} | LR: {lr_curr:.2e} | GNorm: {gnorm.item():.2f} | {sps:.2f} st/s"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_ckpt_path = CKPT_DIR / "quillan_master_frontier_best.pt"
                torch.save({
                    "step": step,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                }, str(best_ckpt_path))
                log_msg(f"--> Saved master frontier checkpoint (Loss: {best_loss:.4f}) to {best_ckpt_path.name}")

            running_loss = 0.0
            steps_accum = 0

    final_ckpt_path = CKPT_DIR / "quillan_master_frontier_final.pt"
    torch.save({
        "step": num_steps,
        "loss": best_loss,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, str(final_ckpt_path))

    log_msg("==================================================================")
    log_msg(f"   🏆 MASTER FRONTIER PRECISION PASS COMPLETE (Best Loss: {best_loss:.4f})")
    log_msg("==================================================================")

if __name__ == "__main__":
    train_master_frontier()
