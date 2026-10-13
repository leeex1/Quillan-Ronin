#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — DEEP MUON-K2 + ADAMW + CCRL ANNEALING ENGINE
---------------------------------------------------------------------------------------
Executes 1,000 steps of deep multi-domain SFT over the 37,463-sample Pristine Multi-Domain
Corpus + Direct Factual Gold anchors.

Features:
  - Resumes from best loss baseline (5.33)
  - Muon (5th-Order Newton-Schulz) for 2D Council Swarms & LoRA matrices
  - AdamW for Ingestion Bridges, Embeddings, and LayerNorms
  - CCRL Dynamic Curvature Regularization
  - Live Out-of-Domain reasoning probes every 100 steps
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
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "training_data"
CKPT_DIR = REPO_ROOT / "checkpoints" / "checkpoints_sft"
PROD_DIR = REPO_ROOT / "checkpoints" / "production_export"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PROD_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignTokenizer
from quillan_muonk2_optimizer import create_quillan_muonk2_optimizer

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("deep_muonk2_engine")

def load_combined_corpus(tokenizer: SovereignTokenizer, max_seq_len: int = 256):
    LOGGER.info("[*] Loading 37k Pristine Frontier dataset...")
    pristine_path = DATA_DIR / "pristine_frontier_gold_37k.pt"
    data_dict = torch.load(str(pristine_path), map_location="cpu", weights_only=False)
    pristine_ids = data_dict["input_ids"]
    pristine_lbls = data_dict["labels"]
    
    samples = []
    for i in range(pristine_ids.size(0)):
        samples.append((pristine_ids[i], pristine_lbls[i]))
        
    LOGGER.info("[+] Loaded %d pristine multi-domain sequences.", len(samples))
    
    # Load anchor factual Q&A to preserve formal proofs
    factual_path = DATA_DIR / "Quillan_Direct_Answers_Gold.jsonl"
    factual_count = 0
    if factual_path.exists():
        with open(factual_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                    p_ids = tokenizer.encode(f"<|user|>\n{d['prompt'].strip()}\n<|assistant|>\n")
                    r_ids = tokenizer.encode(f"{d['response'].strip()}<|im_end|>\n")
                    seq = p_ids + r_ids
                    lbls = [-100] * len(p_ids) + list(r_ids)
                    if len(seq) > max_seq_len:
                        seq = seq[:max_seq_len]
                        lbls = lbls[:max_seq_len]
                    else:
                        pad_len = max_seq_len - len(seq)
                        seq = seq + [50256] * pad_len
                        lbls = lbls + [-100] * pad_len
                    samples.append((torch.tensor(seq, dtype=torch.long), torch.tensor(lbls, dtype=torch.long)))
                    factual_count += 1
                except Exception:
                    pass
        LOGGER.info("[+] Added %d anchor factual QA samples.", factual_count)
        
    random.shuffle(samples)
    return samples

def train_deep_muonk2():
    LOGGER.info("==================================================================")
    LOGGER.info("   👑 QUILLAN-RONIN v5.3.1 — 1,000-STEP DEEP MUON-K2 SFT ENGINE")
    LOGGER.info("==================================================================")

    device = torch.device("cpu")
    tokenizer = SovereignTokenizer("gpt2")
    cfg = QuillanUnrolledConfig()

    model = QuillanUnrolledSovereign(cfg).to(device)

    # Resume from best generalization baseline
    best_ckpt = CKPT_DIR / "quillan_frontier_generalization_best.pt"
    direct_ckpt = CKPT_DIR / "quillan_direct_factual_best.pt"
    resume_path = best_ckpt if best_ckpt.exists() else direct_ckpt

    LOGGER.info("Resuming weights from %s...", resume_path.name)
    data = torch.load(str(resume_path), map_location=device, weights_only=False)
    sd = data.get("model_state_dict", data.get("state_dict", data))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    LOGGER.info("Loaded baseline (missing: %d, unexpected: %d)", len(missing), len(unexpected))

    # Parameter targeting: LoRAs, Swarms, Ingestion Bridges, Prism, and LayerNorms (223.6M active parameters)
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        if any(k in name for k in ['lora', 'swarm', 'expert_swarms', 'q1_bridge', 'q2_bridge', 'ingest_gate', 'prism', 'ln_', 'quillan_finalizer', 'quillan_comm_gate']):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False

    LOGGER.info("Trainable Parameters: %s / %s (%.1f%% Sovereign Active Layers)", f"{trainable_count:,}", f"{total_count:,}", (trainable_count/total_count)*100)

    dataset = load_combined_corpus(tokenizer, max_seq_len=256)
    total_samples = len(dataset)

    # Initialize Custom MuonK2 + AdamW + CCRL Optimizer with smooth annealing rates
    optimizer = create_quillan_muonk2_optimizer(
        model,
        lr_muon=0.012,
        lr_adamw=2.0e-4,
        weight_decay=0.01,
        ccrl_limit=5.0
    )
    LOGGER.info("[+] Sovereign MuonK2 + AdamW + CCRL Optimizer Initialized.")

    num_steps = 1000
    batch_size = 2
    accum_steps = 2  # Effective batch size = 4
    grad_clip = 1.0

    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

    LOGGER.info("Starting Deep Training: %d steps, Effective Batch Size: %d", num_steps, batch_size * accum_steps)

    model.train()
    best_loss = 5.3302
    running_loss = 0.0
    t0 = time.time()

    data_idx = 0
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(accum_steps):
            inp_batch = []
            lbl_batch = []
            for _ in range(batch_size):
                inp, lbl = dataset[data_idx % total_samples]
                inp_batch.append(inp)
                lbl_batch.append(lbl)
                data_idx += 1

            inp_tensor = torch.stack(inp_batch).to(device)
            lbl_tensor = torch.stack(lbl_batch).to(device)

            logits, loss = model(inp_tensor, labels=lbl_tensor)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            step_loss += scaled_loss.item()

        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += step_loss

        if step % 10 == 0 or step == 1:
            avg_loss = running_loss / (10 if step > 1 else 1)
            running_loss = 0.0
            elapsed = time.time() - t0
            sps = step / max(0.001, elapsed)
            lr_curr = scheduler.get_last_lr()[0]
            LOGGER.info("Step [%4d/%4d] | MuonK2 Loss: %.4f | LR: %.2e | Speed: %.2f step/s", step, num_steps, avg_loss, lr_curr, sps)

            if avg_loss < best_loss and step >= 20:
                best_loss = avg_loss
                best_path = CKPT_DIR / "quillan_frontier_generalization_best.pt"
                torch.save({"model_state_dict": model.state_dict(), "step": step, "loss": best_loss}, str(best_path))
                torch.save({"model_state_dict": model.state_dict(), "step": step, "loss": best_loss}, str(PROD_DIR / "quillan_ronin_v531_sovereign_production.pt"))
                LOGGER.info("🏆 Saved New Best Checkpoint (Loss: %.4f)", best_loss)

        if step % 100 == 0:
            # Live Generation Probe across OOD prompts
            model.eval()
            probe_prompts = [
                "<|user|>\nWhat is the difference between synchronous and asynchronous programming in distributed computing?\n<|assistant|>\n",
                "<|user|>\nExplain how photosynthesis converts light energy into chemical glucose.\n<|assistant|>\n"
            ]
            for p_text in probe_prompts:
                toks = tokenizer.encode(p_text)
                gen = model.generate(toks, max_tokens=70, temp=0.25, frequency_penalty=0.5, presence_penalty=0.3)
                gen_text = tokenizer.decode(gen[len(toks):]).strip().split("<|im_end|>")[0].split("<|endoftext|>")[0]
                LOGGER.info("[Probe @ Step %d] Response:\n%s\n", step, gen_text)
            model.train()

    LOGGER.info("==================================================================")
    LOGGER.info("   🏆 DEEP MUON-K2 TRAINING COMPLETE")
    LOGGER.info("   Final Best Loss: %.4f", best_loss)
    LOGGER.info("==================================================================")

if __name__ == "__main__":
    train_deep_muonk2()
