#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — TARGET-MASKED TAG-SLICED SFT ENGINE
Trains on 11,009 intact (Prompt, <think>, <output>) pairs with target-loss masking:
- Loss = -100 on prompt (ignored)
- Loss calculated on reasoning thoughts and structured English output
- 4 CPU worker threads with BELOW_NORMAL_PRIORITY_CLASS for 100% PC stability
- Full 389.1M active parameter training across all 34 Council Experts and 9-Vector Prism
"""

import os
import sys
import time
import math
import random
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
print("   👑 QUILLAN-RONIN v5.3.1 — TAG-SLICED TARGET-MASKED SFT ENGINE", flush=True)
print("   [11,009 INTACT PAIRS | 4 CPU Threads | Below Normal Priority]", flush=True)
print("==================================================================", flush=True)

DATA_DIR = REPO_ROOT / "training_data"
pt_file = DATA_DIR / "intact_pair_dataset.pt"

print(f"[*] Loading intact pair dataset: {pt_file.name}...", flush=True)
t0 = time.time()
ds = torch.load(pt_file, map_location="cpu", weights_only=False)
input_ids_all = ds["input_ids"]
labels_all = ds["labels"]
num_samples = ds["num_samples"]
seq_len = ds["max_seq_len"]

print(f"[+] Loaded {num_samples:,} intact pairs (SeqLen={seq_len}) in {time.time()-t0:.2f}s!\n", flush=True)

enc = tiktoken.get_encoding("gpt2")

# ─── LOAD MASTER MODEL ────────────────────────────────────────────────────────
cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Master Weights: {ckpt_path.name}...", flush=True)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded successfully (Step: {ckpt.get('step','N/A')}, Previous Best Loss: {ckpt.get('loss','N/A')})\n", flush=True)

for param in model.parameters():
    param.requires_grad = True

# ─── TRAINING CONFIGURATION ───────────────────────────────────────────────────
BATCH_SIZE = 1
STEPS = 600
BASE_LR = 2.5e-5
MIN_LR = 1.0e-6

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def quick_eval(step_num: int):
    model.eval()
    test_queries = [
        "Hello! Who are you, and what are your primary capabilities?",
        "Write a Python function to check if a string is a palindrome.",
        "What is the key difference between SIGTERM and SIGKILL in Linux?"
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [TAG-SLICED EVALUATION @ STEP {step_num}]", flush=True)
    print(f"{'='*65}", flush=True)
    for tq in test_queries:
        prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign, highly intelligent AI assistant and expert coder.\n<|user|>\n{tq}\n<|assistant|>\n"
        toks = enc.encode(prompt_str)
        out_tokens = model.generate(toks, max_tokens=40, temp=0.6, top_p=0.9, repetition_penalty=1.18)
        decoded = enc.decode(out_tokens).strip()
        print(f"  Q: '{tq}'", flush=True)
        print(f"  A: {decoded[:180]}", flush=True)
        print(f"{'-'*65}", flush=True)
    print(f"{'='*65}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Tag-Sliced Target-Masked SFT ({STEPS} steps, Batch={BATCH_SIZE}, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

model.train()
t_train = time.time()
best_loss = 999.0
total_pairs_trained = 0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # Sample random intact pair
    idx = random.randint(0, num_samples - 1)
    x_batch = input_ids_all[idx].unsqueeze(0)
    y_batch = labels_all[idx].unsqueeze(0)

    optimizer.zero_grad()
    logits, aux = model(x_batch)
    
    # Calculate loss ONLY on assistant response (ignore -100 prompt tokens)
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        y_batch.view(-1),
        ignore_index=-100
    )
    total_loss = loss + 0.002 * aux
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    val = loss.item()
    total_pairs_trained += 1
    if val < best_loss:
        best_loss = val

    if step % 20 == 0 or step == 1:
        elapsed = time.time() - t_train
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        print(f"  step {step:3d}/{STEPS}  target_loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  ({sps:.1f}s/st, ETA {eta_m:.1f}m)", flush=True)

    if step % 100 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-tag-sliced-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-tag-sliced-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Tag-Sliced Target-Masked Training Complete! Best Target Loss: {best_loss:.4f} in {(time.time()-t_train)/60:.1f}m\n", flush=True)
quick_eval(STEPS)
