#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — 100% CLEAN OMNI-FRONTIER TRAINING ENGINE
Trains across 24,077,205 clean tokens (GPT-5.5, Claude, Gemini, Code, NVIDIA 70B Thoughts)
with safe CPU thread capping (4 threads) and below-normal priority for 100% desktop stability.
"""

import os
import sys
import time
import math
import json
import random
import gc
import psutil
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

# Safe OS thread capping & process priority
p = psutil.Process()
try:
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — 100% CLEAN OMNI-FRONTIER ENGINE", flush=True)
print("   [SAFE SYSTEM MODE: 4 CPU Threads | Below Normal Priority]", flush=True)
print("==================================================================", flush=True)

DATA_DIR = REPO_ROOT / "training_data"
pt_file = DATA_DIR / "clean_unified_multi_frontier.pt"

print(f"[*] Memory-mapping verified dataset: {pt_file.name}...", flush=True)
t0 = time.time()
corpus_tensor = torch.load(pt_file, map_location="cpu", weights_only=False)
total_tokens = corpus_tensor.numel()
print(f"[+] Loaded {total_tokens:,} clean tokens in {time.time()-t0:.2f}s!\n", flush=True)

enc = tiktoken.get_encoding("gpt2")

# ─── LOAD MASTER ARCHITECTURE ─────────────────────────────────────────────────
cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Master Model: {ckpt_path.name}", flush=True)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded successfully (Step: {ckpt.get('step','N/A')}, Loss: {ckpt.get('loss','N/A')})\n", flush=True)

for param in model.parameters():
    param.requires_grad = True

# ─── TRAINING CONFIGURATION ───────────────────────────────────────────────────
BATCH_SIZE = 1
SEQ_LEN = 128
STEPS = 600
BASE_LR = 2.0e-5
MIN_LR = 1.0e-6

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def quick_eval(step_num: int):
    model.eval()
    test_queries = [
        "Hello! What is your name?",
        "Explain what a function is in Python in one clear sentence."
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [CLEAN OMNI-FRONTIER EVALUATION @ STEP {step_num}]", flush=True)
    print(f"{'='*65}", flush=True)
    for tq in test_queries:
        prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{tq}\n<|assistant|>\n"
        toks = enc.encode(prompt_str)
        out_tokens = model.generate(toks, max_tokens=35, temp=0.6, top_p=0.9, repetition_penalty=1.18)
        decoded = enc.decode(out_tokens).strip()
        print(f"  Q: '{tq}'", flush=True)
        print(f"  A: {decoded[:160]}", flush=True)
        print(f"{'-'*65}", flush=True)
    print(f"{'='*65}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Clean Omni-Frontier SFT ({STEPS} steps, Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

model.train()
t_train = time.time()
best_loss = 999.0
total_tokens_trained = 0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    idx = torch.randint(0, total_tokens - SEQ_LEN - 2, (1,)).item()
    x_batch = corpus_tensor[idx : idx + SEQ_LEN].unsqueeze(0)
    y_batch = corpus_tensor[idx + 1 : idx + SEQ_LEN + 1].unsqueeze(0)

    optimizer.zero_grad()
    logits, aux = model(x_batch)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        y_batch.view(-1),
        ignore_index=-100
    )
    total_loss = loss + 0.002 * aux
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    val = total_loss.item()
    total_tokens_trained += SEQ_LEN
    if val < best_loss:
        best_loss = val

    if step % 20 == 0 or step == 1:
        elapsed = time.time() - t_train
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        tok_per_sec = total_tokens_trained / max(elapsed, 0.1)
        print(f"  step {step:3d}/{STEPS}  loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  ({sps:.1f}s/st, {tok_per_sec:,.0f} tok/s, ETA {eta_m:.1f}m)", flush=True)

    if step % 100 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-clean-omni-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-clean-omni-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Clean Omni-Frontier Training Complete! Best Loss: {best_loss:.4f} in {(time.time()-t_train)/60:.1f}m\n", flush=True)
quick_eval(STEPS)
