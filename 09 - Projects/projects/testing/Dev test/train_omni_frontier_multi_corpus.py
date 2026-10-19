#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — OMNI-FRONTIER MULTI-CORPUS SFT ENGINE
Ingests and blends the ENTIRE multi-gigabyte dataset library across all frontier models:
- GPT-5.5 Distilled (GPT_5.5_Distilled.pt)
- Claude & Gemini Distilled (instruct_train.pt, full_train.pt, quillan_tokenized.pt)
- Code & Engineering (code_train.pt)
- Pure Mathematics & Science (quillan_science_absolute.pt, quillan_science_additional.pt)
- Master 166.3M Corpus (quillan_corpus_CLEAN_V7.pt)
- NVIDIA 70B Structured Thought Traces (Quillan_Refined_Thought_Corpus.jsonl)

Trains all 389.1M parameters across the 34 Council Experts, 9-Vector Prism,
and Sovereign Flash Diffusion Core using vectorized batch slicing.
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
from typing import List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — OMNI-FRONTIER MULTI-CORPUS ENGINE", flush=True)
print("==================================================================", flush=True)

DATA_DIR = REPO_ROOT / "training_data"

DATASET_FILES = [
    ("Master Corpus (166.3M)", DATA_DIR / "quillan_corpus_CLEAN_V7.pt", 0.35),
    ("GPT-5.5 Distilled", DATA_DIR / "GPT_5.5_Distilled.pt", 0.15),
    ("Claude/Gemini Instruct", DATA_DIR / "instruct_train.pt", 0.15),
    ("Full Multi-Domain Train", DATA_DIR / "full_train.pt", 0.10),
    ("Code & Engineering", DATA_DIR / "code_train.pt", 0.10),
    ("Quillan Tokenized Dialogue", DATA_DIR / "quillan_tokenized.pt", 0.05),
    ("Science & Mathematics", DATA_DIR / "quillan_science_absolute.pt", 0.05),
    ("Science Additional", DATA_DIR / "quillan_science_additional.pt", 0.05)
]

loaded_corpora = []
total_tokens_all = 0

for name, fpath, weight in DATASET_FILES:
    if fpath.exists():
        try:
            t0 = time.time()
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                # extract largest tensor
                largest_t = None
                for k, v in data.items():
                    if hasattr(v, 'shape') and (largest_t is None or v.numel() > largest_t.numel()):
                        largest_t = v
                data = largest_t if largest_t is not None else torch.tensor([], dtype=torch.long)
            elif isinstance(data, list):
                flat = []
                for item in data:
                    if isinstance(item, list): flat.extend(item)
                    elif hasattr(item, 'tolist'): flat.extend(item.tolist())
                data = torch.tensor(flat, dtype=torch.long)

            if hasattr(data, 'shape') and data.numel() > 1000:
                data = data.contiguous().view(-1)
                loaded_corpora.append((name, data, weight))
                total_tokens_all += data.numel()
                print(f"[+] Loaded {name:28s}: {data.numel():>11,} tokens ({fpath.stat().st_size/(1024**2):6.1f} MB in {time.time()-t0:.2f}s)", flush=True)
        except Exception as e:
            print(f"[-] Notice loading {fpath.name}: {e}", flush=True)

print(f"\n[+] TOTAL MULTI-FRONTIER CORPUS ACTIVE: {total_tokens_all:,} TOKENS across {len(loaded_corpora)} datasets!\n", flush=True)

# Normalize weights
w_sum = sum(w for _, _, w in loaded_corpora)
loaded_corpora = [(n, d, w / w_sum) for n, d, w in loaded_corpora]

# ─── LOAD NVIDIA 70B THOUGHT REASONING SAMPLES ───────────────────────────────
enc = tiktoken.get_encoding("gpt2")
thought_file = DATA_DIR / "Quillan_Refined_Thought_Corpus.jsonl"
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

print(f"[+] Loaded {len(thought_samples)} Gold 70B Structured Thought Samples for reasoning injection.\n", flush=True)

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
BATCH_SIZE = 2
SEQ_LEN = 128
STEPS = 500
BASE_LR = 2.0e-5
MIN_LR = 1.0e-6

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def quick_eval(step_num: int):
    model.eval()
    test_queries = [
        "Hello! Who are you?",
        "Explain what a function is in Python in one clear sentence."
    ]
    print(f"\n{'='*65}", flush=True)
    print(f"  [OMNI-FRONTIER MULTI-CORPUS EVALUATION @ STEP {step_num}]", flush=True)
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

print(f"[TRAIN] Launching Omni-Frontier Multi-Corpus SFT ({STEPS} steps, Batch={BATCH_SIZE}, SeqLen={SEQ_LEN}, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

model.train()
t0 = time.time()
best_loss = 999.0
total_tokens_trained = 0

weights = [w for _, _, w in loaded_corpora]

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # 15% 70B Structured Thought traces, 85% Multi-Frontier Tensor Corpora
    use_thought = (random.random() < 0.15) and len(thought_samples) > 0

    if use_thought:
        raw_toks, p_len = random.choice(thought_samples)
        raw_toks = raw_toks[:SEQ_LEN]
        labs = [-100] * min(p_len, len(raw_toks)) + raw_toks[min(p_len, len(raw_toks)):]
        if len(raw_toks) < SEQ_LEN:
            pad_len = SEQ_LEN - len(raw_toks)
            raw_toks = raw_toks + [50256] * pad_len
            labs = labs + [-100] * pad_len
        x_batch = torch.tensor([raw_toks] * BATCH_SIZE, dtype=torch.long)
        y_batch = torch.tensor([labs] * BATCH_SIZE, dtype=torch.long)
    else:
        # Sample proportionally across GPT-5.5, Claude, Gemini, Code, Math, Science, and Master 166M
        c_idx = random.choices(range(len(loaded_corpora)), weights=weights, k=1)[0]
        c_name, c_tensor, _ = loaded_corpora[c_idx]
        idx = torch.randint(0, c_tensor.numel() - SEQ_LEN - 2, (BATCH_SIZE,))
        x_list = [c_tensor[i : i + SEQ_LEN] for i in idx]
        y_list = [c_tensor[i + 1 : i + SEQ_LEN + 1] for i in idx]
        x_batch = torch.stack(x_list)
        y_batch = torch.stack(y_list)

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
    total_tokens_trained += BATCH_SIZE * SEQ_LEN
    if val < best_loss:
        best_loss = val

    if step % 10 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        tok_per_sec = total_tokens_trained / max(elapsed, 0.1)
        src_tag = "70B-Thought" if use_thought else c_name.split()[0]
        print(f"  step {step:3d}/{STEPS}  loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  [{src_tag:12s}]  ({sps:.1f}s/st, {tok_per_sec:,.0f} tok/s, ETA {eta_m:.1f}m)", flush=True)

    if step % 100 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-omni-frontier-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-omni-frontier-master-final'
}, ckpt_path)

print(f"\n[DONE] 🏆 Omni-Frontier Multi-Corpus SFT Complete! Best Loss: {best_loss:.4f} across {total_tokens_trained:,} tokens in {(time.time()-t0)/60:.1f}m\n", flush=True)
quick_eval(STEPS)
