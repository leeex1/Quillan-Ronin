#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — GROUNDED SEMANTIC ADAPTER SFT
Preserves foundational English grammar by freezing base embeddings and training
the 34 Council MoE Experts, Nine-Vector Prism, and Flash Diffusion Core on intact
(<think>, <output>) tag-sliced instruction pairs.
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
print("   👑 QUILLAN-RONIN v5.3.1 — GROUNDED SEMANTIC SFT ENGINE", flush=True)
print("   [Base Embeddings Frozen | 34 MoE Adapters Trained | Intact Pairs]", flush=True)
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

# ─── LOAD SEMANTIC BASE FOUNDATION ───────────────────────────────────────────
cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

init_ckpt = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_1000000x_master_init.pt"
print(f"[*] Initializing from Pre-Trained English Foundation: {init_ckpt.name}...", flush=True)
ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print("[+] Pre-trained English syntax & vocabulary representations successfully loaded!\n", flush=True)

# ─── FREEZE BASE EMBEDDINGS TO PREVENT GRAMMAR DESTRUCTION ───────────────────
model.ingestion.txt_emb.weight.requires_grad = False
model.ingestion.pos_embed.requires_grad = False
model.txt_dec.weight.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"[+] Trainable Parameters (34 MoE Experts + Prism): {trainable_params/1e6:.1f}M", flush=True)
print(f"[+] Frozen Parameters (Base Vocabulary & Syntax):   {frozen_params/1e6:.1f}M\n", flush=True)

# ─── TRAINING CONFIGURATION ───────────────────────────────────────────────────
STEPS = 400
BASE_LR = 1.2e-5
MIN_LR = 5.0e-7

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=BASE_LR,
    weight_decay=0.01
)

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"

def quick_eval(step_num: int):
    model.eval()
    test_queries = [
        "Hello! Who are you, and what are your primary capabilities?",
        "Write a Python function to check if a string is a palindrome.",
        "What is the key difference between SIGTERM and SIGKILL in Linux?"
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [GROUNDED EVALUATION @ STEP {step_num}]", flush=True)
    print(f"{'='*65}", flush=True)
    for tq in test_queries:
        prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign, highly intelligent AI assistant and expert coder.\n<|user|>\n{tq}\n<|assistant|>\n"
        toks = enc.encode(prompt_str)
        gen = list(toks)
        with torch.no_grad():
            for _ in range(45):
                inp = torch.tensor([gen[-256:]], dtype=torch.long)
                logits = model(inp)
                if isinstance(logits, tuple):
                    logits = logits[0]
                curr = logits[:, -1, :].clone()
                if len(gen) > 0:
                    curr[0, gen[-1]] -= 50.0
                if (len(gen) - len(toks)) < 12:
                    curr[0, 50256] -= 100.0
                recent = gen[-30:]
                for tid in set(recent):
                    curr[0, tid] -= 2.5 * recent.count(tid)
                probs = F.softmax(curr / 0.7, dim=-1)
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=-1)
                remove = cum > 0.9
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                mask = remove.scatter(1, sorted_i, remove)
                probs[mask] = 0.0
                probs = probs / probs.sum()
                next_tok = torch.multinomial(probs, 1).item()
                gen.append(next_tok)
                if next_tok == 50256:
                    break
        decoded = enc.decode(gen[len(toks):]).strip()
        print(f"  Q: '{tq}'", flush=True)
        print(f"  A: {decoded[:200]}", flush=True)
        print(f"{'-'*65}", flush=True)
    print(f"{'='*65}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Grounded Semantic SFT ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)
quick_eval(0)

model.train()
t_train = time.time()
best_loss = 999.0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    idx = random.randint(0, num_samples - 1)
    x_batch = input_ids_all[idx].unsqueeze(0)
    y_batch = labels_all[idx].unsqueeze(0)

    optimizer.zero_grad()
    logits, aux = model(x_batch)
    
    loss = F.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        y_batch.view(-1),
        ignore_index=-100
    )
    total_loss = loss + 0.002 * aux
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
    optimizer.step()

    val = loss.item()
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
            'version': 'quillan-v5.3.1-grounded-semantic-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-grounded-semantic-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Grounded Semantic Training Complete! Best Target Loss: {best_loss:.4f} in {(time.time()-t_train)/60:.1f}m\n", flush=True)
quick_eval(STEPS)
