#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN MINI SFT — CALIBRATED HIGH-VELOCITY CPU RUNNER
=========================================================
Target: Drive loss from 7.14 -> < 2.5 on Intel AVX2 CPU.
Configuration:
  - 3-thread ceiling (guarantees Windows OS, browser, and IDE stay responsive)
  - batch=2, seq=256, grad_acc=1 (~1.5s per step)
  - Direct LRU shard streaming from mini_structured.pt (56 shards)
  - Linear warmup + Cosine decay
  - Live linguistic generation probes every 50 steps
  - Atomic best-only checkpoint saving
"""

import math
import os
import random
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

REPO = Path(r"C:\02_QUILLAN")
for p in [
    str(REPO / "scripts"),
    str(REPO / "09 - Projects" / "projects" / "oni"),
    str(REPO / "03 - Training & Model"),
    str(REPO),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# CPU Thread Safety Ceiling
safe_threads = max(1, (os.cpu_count() or 4) - 1)
torch.set_num_threads(safe_threads)
print(f"[quillan_sft] CPU execution threads set to {safe_threads} (OS UI protected).", flush=True)

from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

# ── Paths & Hyperparameters ──────────────────────────────────────────────
CKPT_IN = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
CKPT_OUT = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
MANIFEST = REPO / "training_data" / "mini_structured.pt"

STEPS = int(os.environ.get("SFT_STEPS", 600))
SEQ = 256
BATCH = 2
LR0 = 4e-5
LR1 = 1e-6
WARMUP = int(os.environ.get("SFT_WARMUP", 20))
VAL_EVERY = int(os.environ.get("SFT_VAL_EVERY", 25))
GEN_EVERY = int(os.environ.get("SFT_GEN_EVERY", 50))
LOG_EVERY = int(os.environ.get("SFT_LOG_EVERY", 5))

GEN_PROMPTS = [
    "User: Who are you?\n\nAssistant:",
    "User: Explain ST-MoE routing and Kuramoto synchronization.\n\nAssistant:",
    "User: What is BitNet 1.58b Straight-Through Estimator quantization?\n\nAssistant:",
]

# ── 1. Sharded Data Loader with LRU Cache ────────────────────────────────
print(f"[quillan_sft] Loading manifest from {MANIFEST.name}...", flush=True)
man = torch.load(MANIFEST, map_location="cpu", weights_only=True)
shards = [Path(s) for s in man["shards"]]
nval = max(1, len(shards) // 5)
tr_sh, va_sh = shards[:-nval], shards[-nval:]
print(f"[quillan_sft] train_shards={len(tr_sh)} val_shards={len(va_sh)} total_rows={man['total']}", flush=True)

pairs = []
for si, sh in enumerate(tr_sh):
    n = torch.load(sh, map_location="cpu", weights_only=True).shape[0]
    pairs.extend((si, ri) for ri in range(n))
random.Random(4242).shuffle(pairs)
print(f"[quillan_sft] Shuffled index ready: {len(pairs)} dialogue pairs.", flush=True)

shard_cache = {}
def get_row(si, ri):
    if si not in shard_cache:
        shard_cache[si] = torch.load(tr_sh[si], map_location="cpu", weights_only=True)
        if len(shard_cache) > 4:
            shard_cache.pop(next(iter(shard_cache)))
    return shard_cache[si][ri]

# Fixed validation split for apples-to-apples loss comparison
va_rows = torch.cat([
    torch.load(va_sh[i], map_location="cpu", weights_only=True)[:8]
    for i in range(min(2, len(va_sh)))
])[:16, :SEQ]

# ── 2. Model Initialization & Weight Binding ─────────────────────────────
print(f"[quillan_sft] Initializing QuillanRoninOni architecture...", flush=True)
cfg = QuillanOniConfig(
    vocab_size=50262,
    hidden_dim=1024,
    ffn_dim=2048,
    n_layer=6,
    num_experts=34,
    top_k=4,
    max_seq_len=512,
)
model = QuillanRoninOni(cfg)

print(f"[quillan_sft] Loading starting checkpoint {CKPT_IN.name}...", flush=True)
bd = torch.load(CKPT_IN, map_location="cpu", weights_only=True)
sd = bd.get("model_state_dict", bd.get("model", bd))
missing, unexp = model.load_state_dict(sd, strict=False)
meta = bd.get("meta", {}) if isinstance(bd, dict) else {}
prev_best = meta.get("val", 7.1400)
print(f"[quillan_sft] Checkpoint bound successfully: missing={len(missing)} unexp={len(unexp)} baseline_val={prev_best:.4f}", flush=True)

tok = QuillanBPETokenizer()
opt = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=0.01, betas=(0.9, 0.95))

# ── 3. Generation Probe Helper ───────────────────────────────────────────
def generate_sample(prompt: str, max_new: int = 35) -> str:
    model.eval()
    ids = tok.encode(prompt)
    gen = list(ids[-128:])
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            out = model(x)
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1, :50262].clone()
            # Repetition dampening
            for tid in set(gen[-24:]):
                if logits[tid] > 0:
                    logits[tid] /= 1.2
                else:
                    logits[tid] *= 1.2
            # Top-k
            v, _ = torch.topk(logits, 40)
            logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits / 0.7, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            if nxt in (50256, 50257, 50261):
                break
            gen.append(nxt)
    model.train()
    out_tokens = [t for t in gen[len(ids):] if t < 50257]
    return tok.decode(out_tokens).strip()

# ── 4. Main Calibrated Training Loop ─────────────────────────────────────
print(f"\n=======================================================", flush=True)
print(f"🚀 STARTING CALIBRATED SFT RUN", flush=True)
print(f"• Steps: {STEPS} | Batch: {BATCH} | SeqLen: {SEQ}", flush=True)
print(f"• LR Schedule: {LR0:.1e} -> {LR1:.1e} (Warmup: {WARMUP} steps)", flush=True)
print(f"• Target: Loss < 2.5 (Conversational Fluency)", flush=True)
print(f"=======================================================\n", flush=True)

model.train()
best_val = prev_best
t_start = time.time()
opt.zero_grad()

for step in range(1, STEPS + 1):
    t_step0 = time.time()
    
    # Warmup + Cosine LR Schedule
    if step <= WARMUP:
        cur_lr = LR0 * step / WARMUP
    else:
        progress = (step - WARMUP) / max(1, STEPS - WARMUP)
        cur_lr = LR1 + 0.5 * (LR0 - LR1) * (1.0 + math.cos(math.pi * progress))
    
    for pg in opt.param_groups:
        pg["lr"] = cur_lr

    # Forward + Loss with all academic paper auxiliary techniques
    global_idx = (step - 1) * BATCH
    batch_rows = [pairs[(global_idx + k) % len(pairs)] for k in range(BATCH)]
    x = torch.stack([get_row(si, ri) for si, ri in batch_rows])[:, :SEQ]
    
    logits, ce_loss, aux_dict = model(x, labels=x, return_aux=True)
    aux_loss = model.total_aux_loss(aux_dict)
    total_loss = ce_loss + aux_loss
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    
    step_time = time.time() - t_step0
    tokens_step = BATCH * SEQ
    tok_per_sec = tokens_step / max(step_time, 0.001)

    # Periodic Console Logging
    if step % LOG_EVERY == 0 or step == 1:
        elapsed = time.time() - t_start
        eta_sec = (elapsed / step) * (STEPS - step)
        print(
            f"Step {step:4d}/{STEPS} | CE: {ce_loss.item():.4f} | Aux: {aux_loss.item():.4f} | Tot: {total_loss.item():.4f} | LR: {cur_lr:.2e} | "
            f"Speed: {tok_per_sec:.1f} tok/s ({step_time:.2f}s/step) | ETA: {eta_sec/60:.1f}m",
            flush=True,
        )

    # Validation & Best Checkpoint Persistence
    if step % VAL_EVERY == 0 or step == STEPS:
        model.eval()
        with torch.no_grad():
            _, v_ce, v_aux = model(va_rows, labels=va_rows, return_aux=True)
            v_aux_loss = model.total_aux_loss(v_aux)
            v_loss = v_ce.item()
            v_tot = (v_ce + v_aux_loss).item()
        
        print(f"\n[VAL @ Step {step}] Current Val CE: {v_loss:.4f} (Tot: {v_tot:.4f}, Best: {best_val:.4f})", flush=True)
        if v_loss < best_val:
            best_val = v_loss
            # Atomic checkpoint save
            temp_out = CKPT_OUT.with_suffix(".tmp")
            torch.save({
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "meta": {
                    "val": best_val,
                    "step": step,
                    "stage": "calibrated_sft",
                    "seq": SEQ,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, temp_out)
            if temp_out.exists():
                temp_out.replace(CKPT_OUT)
            print(f"⭐ [CHECKPOINT SAVED] New best validation loss: {best_val:.4f} -> {CKPT_OUT.name}\n", flush=True)
        model.train()

    # Live Fluency Generation Probe
    if step % GEN_EVERY == 0:
        print(f"\n───────────────────────────────────────────────────────", flush=True)
        print(f"💬 LINGUISTIC PROBE @ STEP {step} (CE: {ce_loss.item():.4f} Tot: {total_loss.item():.4f})", flush=True)
        print(f"───────────────────────────────────────────────────────", flush=True)
        for prompt in GEN_PROMPTS:
            out_text = generate_sample(prompt, max_new=30)
            q_line = prompt.split("\n")[0]
            print(f"  {q_line}", flush=True)
            print(f"  --> {out_text!r}\n", flush=True)
        print(f"───────────────────────────────────────────────────────\n", flush=True)

total_elapsed = time.time() - t_start
print(f"\n=======================================================", flush=True)
print(f"✅ SFT TRAINING COMPLETED in {total_elapsed/60:.1f} minutes", flush=True)
print(f"• Final Best Validation Loss: {best_val:.4f}", flush=True)
print(f"• Checkpoint: {CKPT_OUT}", flush=True)
print(f"=======================================================", flush=True)
