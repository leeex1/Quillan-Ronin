#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — HIGH-THROUGHPUT 166M TOKEN TENSOR SFT ENGINE
Vectorized streaming training across 166.3M tokens (quillan_corpus_CLEAN_V7.pt)
blended with Gold Structured Thought Traces (Quillan_Refined_Thought_Corpus.jsonl).
Trains all 389.1M parameters across the 34 Council Experts, 9-Vector Prism,
and Sovereign Flash Diffusion Core end-to-end.
"""

import os
import sys
import time
import math
import json
import random
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — 166M TENSOR VECTORIZED SFT ENGINE", flush=True)
print("==================================================================", flush=True)

# ─── 1. LOAD 166.3M PRE-TOKENIZED DATASET TENSOR ─────────────────────────────
pt_file = REPO_ROOT / "training_data" / "quillan_corpus_CLEAN_V7.pt"
print(f"[*] Memory-mapping {pt_file.name}...", flush=True)
t_load = time.time()
corpus_tensor = torch.load(pt_file, map_location="cpu", weights_only=False)
total_tokens = corpus_tensor.shape[0]
print(f"[+] Successfully loaded {total_tokens:,} tokens in {time.time()-t_load:.2f}s!\n", flush=True)

# ─── 2. LOAD GOLD STRUCTURED THOUGHT REASONING SAMPLES ───────────────────────
enc = tiktoken.get_encoding("gpt2")
thought_file = REPO_ROOT / "training_data" / "Quillan_Refined_Thought_Corpus.jsonl"
thought_samples = []

if thought_file.exists():
    with open(thought_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            try:
                d = json.loads(line)
                q = d.get("prompt", "")
                ans = d.get("refined_reasoning", "")
                if q and ans:
                    full_txt = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{q}\n<|assistant|>\n{ans}\n<|end|>"
                    p_txt = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{q}\n<|assistant|>\n"
                    toks = enc.encode(full_txt)
                    p_toks = enc.encode(p_txt)
                    if len(toks) > len(p_toks) + 5:
                        thought_samples.append((toks, len(p_toks)))
            except Exception:
                continue

print(f"[+] Loaded {len(thought_samples)} Gold Structured Thought Samples for curriculum blending.\n", flush=True)

# ─── 3. LOAD QUILLAN-RONIN MASTER ARCHITECTURE ────────────────────────────────
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

# ─── 4. TRAINING HYPERPARAMETERS ──────────────────────────────────────────────
BATCH_SIZE = 4
SEQ_LEN = 128
STEPS = 600
BASE_LR = 2.5e-5
MIN_LR = 1.0e-6
GRAD_ACCUM = 2

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def quick_eval(step_num: int):
    model.eval()
    test_queries = [
        "Hello! Who are you?",
        "Explain what a function is in Python in one clear sentence."
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [166M TENSOR EVALUATION @ STEP {step_num}]", flush=True)
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

print(f"[TRAIN] Launching 166M-Token Vectorized SFT ({STEPS} steps, Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

model.train()
t0 = time.time()
best_loss = 999.0
total_tokens_trained = 0

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    optimizer.zero_grad()
    accum_loss = 0.0

    for _ in range(GRAD_ACCUM):
        # 80% 166M Corpus pre-tokenized tensor, 20% Gold Thought Reasoning
        use_thought = (random.random() < 0.20) and len(thought_samples) > 0

        if use_thought:
            raw_toks, p_len = random.choice(thought_samples)
            raw_toks = raw_toks[:SEQ_LEN]
            labs = [-100] * min(p_len, len(raw_toks)) + raw_toks[min(p_len, len(raw_toks)):]
            # Pad to SEQ_LEN
            if len(raw_toks) < SEQ_LEN:
                pad_len = SEQ_LEN - len(raw_toks)
                raw_toks = raw_toks + [50256] * pad_len
                labs = labs + [-100] * pad_len
            x_batch = torch.tensor([raw_toks] * BATCH_SIZE, dtype=torch.long)
            y_batch = torch.tensor([labs] * BATCH_SIZE, dtype=torch.long)
        else:
            # High-speed slicing across 166.3M tokens
            idx = torch.randint(0, total_tokens - SEQ_LEN - 2, (BATCH_SIZE,))
            x_list = [corpus_tensor[i : i + SEQ_LEN] for i in idx]
            y_list = [corpus_tensor[i + 1 : i + SEQ_LEN + 1] for i in idx]
            x_batch = torch.stack(x_list)
            y_batch = torch.stack(y_list)

        logits, aux = model(x_batch)
        loss = F.cross_entropy(
            logits.view(-1, cfg.vocab_size),
            y_batch.view(-1),
            ignore_index=-100
        )
        total_loss = (loss + 0.002 * aux) / GRAD_ACCUM
        total_loss.backward()
        accum_loss += loss.item() / GRAD_ACCUM

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    total_tokens_trained += BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    if accum_loss < best_loss:
        best_loss = accum_loss

    if step % 10 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        tok_per_sec = total_tokens_trained / max(elapsed, 0.1)
        print(f"  step {step:3d}/{STEPS}  loss={accum_loss:.4f}  best={best_loss:.4f}  lr={lr:.7f}  ({sps:.1f}s/st, {tok_per_sec:,.0f} tok/s, ETA {eta_m:.1f}m)", flush=True)

    if step % 100 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-166m-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-166m-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 166M Vectorized Training Complete! Best Loss: {best_loss:.4f} across {total_tokens_trained:,} tokens in {(time.time()-t0)/60:.1f}m\n", flush=True)
quick_eval(STEPS)
