#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — PRISTINE FRONTIER SFT MASTER ENGINE (37K+ SAMPLES)
---------------------------------------------------------------------------------------
Trains 100% Unrolled Sovereign Architecture on 37,463 Pristine Reasoning & Coding Pairs.
- Zero OOM memory footprint (Momentum SGD + set_to_none=True + proactive GC)
- Prompt-masked target cross-entropy loss (ignore_index=-100)
- Live generation audits every 50 steps
- Auto-saves to quillan_thinking_reasoning_master.pt
"""

import os
import sys
import time
import math
import gc
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

DATASET_PATH = Path(r"C:\02_QUILLAN\training_data\pristine_frontier_gold_37k.pt")
CHECKPOINT_PATH = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")

TOTAL_STEPS = 600
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 0.005
MOMENTUM = 0.85
WEIGHT_DECAY = 1e-4

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — MASTER SFT ANNEALING (37K+ SAMPLES)", flush=True)
print("==================================================================", flush=True)

enc = tiktoken.get_encoding("gpt2")
cfg = QuillanUnrolledConfig()

print(f"[*] Initializing Unrolled Sovereign Model...", flush=True)
model = QuillanUnrolledSovereign(cfg).to("cpu")

if CHECKPOINT_PATH.exists():
    print(f"[*] Loading weights from {CHECKPOINT_PATH.name}...", flush=True)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    print(f"[+] Model loaded successfully ({sum(p.numel() for p in model.parameters())*4/1e6:.1f} MB)!\n", flush=True)
else:
    print(f"[!] Warning: Checkpoint not found at {CHECKPOINT_PATH}, initializing from scratch.", flush=True)

print(f"[*] Loading Pristine Dataset from {DATASET_PATH.name}...", flush=True)
ds = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
input_ids_all = ds['input_ids']
labels_all = ds['labels']
num_samples = input_ids_all.size(0)
print(f"[+] Loaded {num_samples:,} samples (Tensor Shape: {input_ids_all.shape})\n", flush=True)

# Momentum SGD optimizer - extremely light on memory, prevents OOM
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

AUDIT_PROMPTS = [
    "<|user|>\nWrite a Python function to perform binary search on a sorted list.\n<|assistant|>\n",
    "<|user|>\nA triangle has sides of length 5, 12, and 13. Is it a right triangle? What is its area?\n<|assistant|>\n",
    "<|user|>\nWhat is the measurement problem in quantum mechanics?\n<|assistant|>\n"
]

@torch.no_grad()
def run_live_audit(step: int):
    model.eval()
    print(f"\n==================================================================", flush=True)
    print(f"   [STEP {step} LIVE GENERATION AUDIT]", flush=True)
    print(f"==================================================================", flush=True)
    for p in AUDIT_PROMPTS:
        tokens = enc.encode(p)
        gen = model.generate(tokens, max_tokens=50, temp=0.7, frequency_penalty=0.4, presence_penalty=0.3)
        res = enc.decode(gen)
        # Show prompt and response clearly
        p_clean = p.replace('\n', ' ').strip()
        resp_clean = res[len(p):].strip()
        print(f"\nPROMPT: {p_clean}", flush=True)
        print(f"RESPONSE:\n{resp_clean}\n", flush=True)
    print("==================================================================\n", flush=True)
    model.train()

print(f"[*] Running initial Step 0 baseline audit...", flush=True)
run_live_audit(0)

print(f"[*] Starting Master SFT Training Loop ({TOTAL_STEPS} Steps, Effective Batch = {BATCH_SIZE * GRAD_ACCUM})...", flush=True)

best_loss = 999.0
step_start = time.time()

# Shuffle indices
indices = torch.randperm(num_samples)
idx_cursor = 0

for step in range(1, TOTAL_STEPS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    
    for micro in range(GRAD_ACCUM):
        if idx_cursor + BATCH_SIZE > num_samples:
            indices = torch.randperm(num_samples)
            idx_cursor = 0
            
        batch_indices = indices[idx_cursor : idx_cursor + BATCH_SIZE]
        idx_cursor += BATCH_SIZE
        
        batch_x = input_ids_all[batch_indices]
        batch_y = labels_all[batch_indices]
        
        _, loss = model(batch_x, labels=batch_y)
        loss_scaled = loss / GRAD_ACCUM
        loss_scaled.backward()
        accum_loss += loss.item() / GRAD_ACCUM
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if step % 10 == 0:
        gc.collect()
        
    if step % 10 == 0 or step == 1:
        ram_gb = psutil.virtual_memory().available / (1024**3)
        ram_pct = psutil.virtual_memory().percent
        dt = (time.time() - step_start) / max(1, step)
        eta_min = (TOTAL_STEPS - step) * dt / 60
        print(f"Step [{step:03d}/{TOTAL_STEPS}] | Loss: {accum_loss:.4f} | RAM Free: {ram_gb:.2f}GB ({ram_pct}%) | Pace: {dt:.2f}s/step | ETA: {eta_min:.1f}m", flush=True)
        
    if step % 50 == 0:
        run_live_audit(step)
        
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'cfg': cfg,
            'step': step,
            'loss': accum_loss,
            'version': 'quillan-v5.3.1-unrolled-master'
        }, CHECKPOINT_PATH)
        print(f"[+] Saved Master Checkpoint (Step {step}, Loss: {accum_loss:.4f}) to {CHECKPOINT_PATH.name}!\n", flush=True)

print(f"\n==================================================================", flush=True)
print(f"   👑 MASTER SFT ANNEALING COMPLETED SUCCESSFULLY!", flush=True)
print(f"==================================================================", flush=True)
run_live_audit(TOTAL_STEPS)

torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': cfg,
    'step': TOTAL_STEPS,
    'loss': accum_loss,
    'version': 'quillan-v5.3.1-unrolled-master-final'
}, CHECKPOINT_PATH)

print(f"[+] Final Master Checkpoint Saved to {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size/(1024**2):.1f} MB)!", flush=True)
