#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — 34-EXPERT DOMAIN ALIGNMENT ENGINE
---------------------------------------------------------------------------------------
Executes targeted Supervised Fine-Tuning across all 34 Council Expert channels
using domain-partitioned datasets in `training_data/experts_34/` and `router_training_dataset.jsonl`.

Features:
- Target-masked Cross-Entropy (ignore_index = -100 on prompts).
- Safe parameter updates with AdamW (LR = 2.0e-5 with Cosine Annealing).
- Gradient accumulation & clipping (max_norm = 1.0) to prevent catastrophic forgetting.
- Periodic checkpoint saves with step & loss metadata.
"""

import os
import sys
import time
import math
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Setup directories
REPO_ROOT = Path(r"C:\02_QUILLAN")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
TRAIN_DATA_DIR = REPO_ROOT / "training_data"
EXPERTS_DIR = TRAIN_DATA_DIR / "experts_34"
CKPT_DIR = REPO_ROOT / "checkpoints" / "checkpoints_sft"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRATCH_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train_34_experts")

def load_expert_dataset(tokenizer: SovereignTokenizer, max_seq_len: int = 512) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Loads and tokenizes QA pairs with target masking from all 34 expert datasets."""
    packed_samples = []
    
    expert_files = sorted(list(EXPERTS_DIR.glob("*.jsonl")))
    LOGGER.info("Loading domain pairs from %d expert datasets...", len(expert_files))
    
    for fpath in expert_files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                
                q = data.get("question") or data.get("prompt") or ""
                a = data.get("response") or data.get("answer") or ""
                if not q or not a:
                    continue
                
                # Format with standardized delimiter boundaries
                prompt_text = f"Question: {q.strip()}\nAnswer:\n"
                response_text = f"{a.strip()}<|im_end|>"
                
                prompt_ids = tokenizer.encode(prompt_text)
                response_ids = tokenizer.encode(response_text)
                
                full_sequence = prompt_ids + response_ids
                labels = [-100] * len(prompt_ids) + list(response_ids)
                
                if len(full_sequence) > max_seq_len:
                    full_sequence = full_sequence[:max_seq_len]
                    labels = labels[:max_seq_len]
                
                # Check that we have at least 1 response token in target
                num_target_tokens = sum(1 for x in labels if x != -100)
                if len(full_sequence) > 1 and num_target_tokens > 0:
                    inp_t = torch.tensor(full_sequence, dtype=torch.long)
                    lbl_t = torch.tensor(labels, dtype=torch.long)
                    assert inp_t.shape == lbl_t.shape, f"Length mismatch: {inp_t.shape} vs {lbl_t.shape}"
                    packed_samples.append((inp_t, lbl_t))
                    
    LOGGER.info("Total packed and target-masked samples prepared: %d", len(packed_samples))
    return packed_samples

def main():
    LOGGER.info("==================================================================")
    LOGGER.info("   👑 34-COUNCIL EXPERT DOMAIN ALIGNMENT TRAINING")
    LOGGER.info("==================================================================")

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanUnrolledConfig()

    master_ckpt_path = CKPT_DIR / "quillan_thinking_reasoning_master.pt"
    backup_ckpt_path = CKPT_DIR / "quillan_34_experts_aligned_best.pt"

    model = QuillanUnrolledSovereign(cfg).to(device)

    if master_ckpt_path.exists():
        LOGGER.info("Resuming from existing checkpoint: %s", master_ckpt_path.name)
        ckpt = torch.load(str(master_ckpt_path), map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
    else:
        LOGGER.info("Starting fresh from initialized sovereign architecture.")

    model.train()
    dataset = load_expert_dataset(tokenizer, max_seq_len=256)
    if not dataset:
        LOGGER.error("No training data found in %s", EXPERTS_DIR)
        return

    # Optimizer & Scheduler
    peak_lr = 2.0e-5
    min_lr = 1.0e-6
    total_steps = 1000
    accum_steps = 4
    warmup_steps = 50

    optimizer = AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), weight_decay=0.01)

    LOGGER.info("Beginning 34-expert alignment pass for %d steps (accum_steps=%d, peak_lr=%.2e)...", total_steps, accum_steps, peak_lr)

    best_loss = float("inf")
    running_loss = 0.0
    t0 = time.time()

    optimizer.zero_grad()
    for step in range(1, total_steps + 1):
        # Sample mini-batch
        inp_t, lbl_t = random.choice(dataset)
        inp_t = inp_t.unsqueeze(0).to(device)
        lbl_t = lbl_t.unsqueeze(0).to(device)

        logits, loss = model(inp_t, labels=lbl_t)
        loss = loss / accum_steps
        loss.backward()
        running_loss += loss.item() * accum_steps

        if step % accum_steps == 0:
            # Gradient clipping
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Learning rate schedule
            curr_step = step // accum_steps
            if curr_step < warmup_steps:
                lr = peak_lr * (curr_step / max(1, warmup_steps))
            else:
                progress = (curr_step - warmup_steps) / max(1, (total_steps // accum_steps) - warmup_steps)
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()

        # Periodic logging & checkpointing
        if step % 50 == 0:
            avg_loss = running_loss / 50.0
            elapsed = time.time() - t0
            sps = 50.0 / max(0.001, elapsed)
            current_lr = optimizer.param_groups[0]['lr']
            
            LOGGER.info("Step %4d/%d | Loss: %.4f | LR: %.2e | GNorm: %.2f | %.2f st/s", step, total_steps, avg_loss, current_lr, gnorm.item() if 'gnorm' in locals() else 0.0, sps)
            running_loss = 0.0
            t0 = time.time()

            if avg_loss < best_loss:
                best_loss = avg_loss
                LOGGER.info("--> Saving new best checkpoint (Loss: %.4f) to %s", best_loss, backup_ckpt_path.name)
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                }, str(backup_ckpt_path))

        if step % 250 == 0:
            LOGGER.info("--> Periodic master checkpoint snapshot at step %d to %s", step, master_ckpt_path.name)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
            }, str(master_ckpt_path))

    LOGGER.info("==================================================================")
    LOGGER.info("   🏆 34-EXPERT DOMAIN ALIGNMENT TRAINING COMPLETE (Best Loss: %.4f)", best_loss)
    LOGGER.info("==================================================================")

if __name__ == "__main__":
    main()
