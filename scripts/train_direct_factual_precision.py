#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — DIRECT FACTUAL PRECISION TRAINER
---------------------------------------------------------------------------------------
Executes targeted SFT strictly on clean, direct question-answer completions.
Eliminates meta-header boilerplate, token fragmentation, and prompt leakage.
"""

import os
import sys
import time
import math
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = Path(r"C:\02_QUILLAN")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
TRAIN_DATA_DIR = REPO_ROOT / "training_data"
CKPT_DIR = REPO_ROOT / "checkpoints" / "checkpoints_sft"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRATCH_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("direct_factual_trainer")

def load_clean_dataset(tokenizer: SovereignTokenizer, max_seq_len: int = 384) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    data_file = TRAIN_DATA_DIR / "Quillan_Direct_Answers_Gold.jsonl"
    LOGGER.info("Loading clean factual corpus from %s...", data_file.name)
    
    samples = []
    with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            
            prompt_text = data["prompt"]
            response_text = data["response"]
            
            prompt_ids = tokenizer.encode(prompt_text)
            response_ids = tokenizer.encode(response_text)
            
            full_sequence = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + list(response_ids)
            
            if len(full_sequence) > max_seq_len:
                full_sequence = full_sequence[:max_seq_len]
                labels = labels[:max_seq_len]
            
            if len(full_sequence) > 1 and any(x != -100 for x in labels):
                inp_t = torch.tensor(full_sequence, dtype=torch.long)
                lbl_t = torch.tensor(labels, dtype=torch.long)
                samples.append((inp_t, lbl_t))
                
    LOGGER.info("Loaded %d target-masked clean samples.", len(samples))
    return samples

def main():
    LOGGER.info("==================================================================")
    LOGGER.info("   👑 DIRECT FACTUAL PRECISION PASS (ZERO DRIFT / ZERO SALAD)")
    LOGGER.info("==================================================================")

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanUnrolledConfig()

    master_ckpt_path = CKPT_DIR / "quillan_thinking_reasoning_master.pt"
    precision_best_path = CKPT_DIR / "quillan_gold_precision_best.pt"
    direct_best_path = CKPT_DIR / "quillan_direct_factual_best.pt"

    model = QuillanUnrolledSovereign(cfg).to(device)

    # Resume from the precision best checkpoint
    resume_path = precision_best_path if precision_best_path.exists() else master_ckpt_path
    LOGGER.info("Resuming from checkpoint: %s", resume_path.name)
    ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)

    model.train()
    dataset = load_clean_dataset(tokenizer, max_seq_len=384)

    peak_lr = 3.0e-5
    min_lr = 1.0e-6
    total_steps = 300
    accum_steps = 4
    warmup_steps = 20

    optimizer = AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), weight_decay=0.05)

    LOGGER.info("Starting Direct Precision Pass (%d steps, Peak LR: %.2e)...", total_steps, peak_lr)

    best_loss = float("inf")
    running_loss = 0.0
    t0 = time.time()

    optimizer.zero_grad()
    for step in range(1, total_steps + 1):
        inp_t, lbl_t = random.choice(dataset)
        inp_t = inp_t.unsqueeze(0).to(device)
        lbl_t = lbl_t.unsqueeze(0).to(device)

        logits, loss = model(inp_t, labels=lbl_t)
        loss = loss / accum_steps
        loss.backward()
        running_loss += loss.item() * accum_steps

        if step % accum_steps == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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

        if step % 25 == 0:
            avg_loss = running_loss / 25.0
            elapsed = time.time() - t0
            sps = 25.0 / max(0.001, elapsed)
            current_lr = optimizer.param_groups[0]['lr']
            
            LOGGER.info("Step %3d/%d | Loss: %.4f | LR: %.2e | GNorm: %.2f | %.2f st/s", step, total_steps, avg_loss, current_lr, gnorm.item() if 'gnorm' in locals() else 0.0, sps)
            running_loss = 0.0
            t0 = time.time()

            if avg_loss < best_loss:
                best_loss = avg_loss
                LOGGER.info("--> Saving new best direct factual checkpoint (Loss: %.4f) to %s", best_loss, direct_best_path.name)
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                }, str(direct_best_path))
                # Also update the master checkpoint
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                }, str(master_ckpt_path))

    LOGGER.info("==================================================================")
    LOGGER.info("   🏆 DIRECT FACTUAL PRECISION PASS COMPLETE (Best Loss: %.4f)", best_loss)
    LOGGER.info("==================================================================")

if __name__ == "__main__":
    main()
