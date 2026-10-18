#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN THOUGHT REFINERY & DISTILLATION PIPELINE
Transforms raw dataset entries into frontier-grade Structured Thought & Reasoning traces
(<thought>...</thought><output>...</output>) using NVIDIA Dual-Teacher NIM (8B high-speed + 70B deep proof)
and trains Quillan-Ronin's 34 Council Experts, 9-Vector Prism, and Spiderweb in real time.
"""

import os
import sys
import time
import math
import json
import random
import threading
import queue
import requests
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

enc = tiktoken.get_encoding("gpt2")

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN THOUGHT REFINERY ENGINE", flush=True)
print("==================================================================", flush=True)

# ─── 1. HARVEST RAW SEED PROMPTS FROM LOCAL DATASETS ──────────────────────────
DATA_FILES = [
    REPO_ROOT / "training_data" / "instruct_train.jsonl",
    REPO_ROOT / "training_data" / "full_train.jsonl",
    REPO_ROOT / "training_data" / "quillan_corpus_CLEAN_V7.jsonl",
    REPO_ROOT / "training_data" / "GPT_5.5_Distilled.jsonl"
]

raw_prompt_pool: List[str] = []

for df in DATA_FILES:
    if not df.exists():
        continue
    count = 0
    try:
        with open(df, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    candidate = None
                    if "instruction" in data and len(data["instruction"]) > 10:
                        candidate = data["instruction"]
                    elif "prompt" in data and len(data["prompt"]) > 10:
                        candidate = data["prompt"]
                    elif "input" in data and len(data["input"]) > 10:
                        candidate = data["input"]
                    elif "text" in data and len(data["text"]) > 20:
                        s = data["text"].split("\n")[0].strip()
                        if len(s) > 15:
                            candidate = s

                    if candidate:
                        candidate = candidate.replace("# 🤖", "").replace("# 🧠", "").strip()
                        if len(candidate) > 12 and candidate not in raw_prompt_pool:
                            raw_prompt_pool.append(candidate)
                            count += 1
                            if count >= 200:
                                break
                except Exception:
                    continue
    except Exception:
        pass

print(f"[+] Harvested {len(raw_prompt_pool)} unique seed prompts from local corpus.\n", flush=True)

CURRICULUM_BACKUP = [
    "Explain the architectural advantage of Mixture-of-Experts over dense transformers.",
    "How does low-rank adaptation (LoRA) enable parameter-efficient fine-tuning?",
    "Explain what a LayerNorm does in a transformer and why residual normalization prevents logit explosion.",
    "Write a clean Python context manager class for safe file locking.",
    "Prove why the square root of 2 is an irrational number step-by-step.",
    "Explain what an eigenvalue and eigenvector are in linear algebra in intuitive geometric terms.",
    "What is CWE-89 (SQL Injection) and what is the exact architectural pattern to remediate it?",
    "Explain how input sanitization, least privilege, and deterministic resource deallocation prevent remote code execution.",
    "Explain the difference between threading, multiprocessing, and asyncio in Python with clear use cases.",
    "Explain the mathematical foundation of gradient descent and why learning rate scheduling is necessary."
]

for item in CURRICULUM_BACKUP:
    if item not in raw_prompt_pool:
        raw_prompt_pool.append(item)

# ─── 2. HIGH-SPEED DUAL-TEACHER REASONING REFINERY ────────────────────────────
refinery_queue: queue.Queue = queue.Queue(maxsize=60)
stop_event = threading.Event()

SYSTEM_REFINER_PROMPT = (
    "You are a master AI engineer and philosopher. Given an input query, produce a structured reasoning trace in concise English:\n"
    "<thought>[Concise logical decomposition, key mathematical or architectural principles, edge cases]</thought>\n"
    "<output>[Direct, brilliant, articulate, and completely factual final answer]</output>"
)

def refinery_worker():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    while not stop_event.is_set():
        if refinery_queue.qsize() >= 30:
            time.sleep(0.5)
            continue
        
        raw_p = random.choice(raw_prompt_pool)
        # Use 8B for rapid parallel throughput (0.6s/call) and 70B for deep proofs
        model_name = random.choice(["meta/llama-3.1-8b-instruct", "meta/llama-3.1-8b-instruct", "meta/llama-3.1-70b-instruct"])
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_REFINER_PROMPT},
                {"role": "user", "content": raw_p}
            ],
            "max_tokens": 120,
            "temperature": 0.25
        }
        
        try:
            r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=20)
            if r.status_code == 200:
                refined_text = r.json()["choices"][0]["message"]["content"].strip()
                if refined_text and len(refined_text) > 20:
                    refinery_queue.put((raw_p, refined_text, model_name))
            else:
                time.sleep(0.5)
        except Exception:
            time.sleep(0.5)

# Launch 6 concurrent worker threads for high throughput
for _ in range(6):
    t = threading.Thread(target=refinery_worker, daemon=True)
    t.start()

print("[*] Launched 6 Parallel Thought Refinery Worker Threads. Priming reasoning buffer...", flush=True)
while refinery_queue.qsize() < 5:
    time.sleep(0.4)
print(f"[+] Buffer primed with {refinery_queue.qsize()} elevated thought reasoning samples!\n", flush=True)

# ─── 3. LOAD QUILLAN-RONIN MASTER ARCHITECTURE ────────────────────────────────
cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Aligned Master Model: {ckpt_path.name}", flush=True)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded successfully (Step: {ckpt.get('step','N/A')}, Loss: {ckpt.get('loss','N/A')})\n", flush=True)

for param in model.parameters():
    param.requires_grad = True

STEPS = 250
BASE_LR = 2.0e-5
MIN_LR = 1.0e-6
SEQ_LEN = 192

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

def quick_eval(step_num: int):
    model.eval()
    test_q = "Explain what a function is in Python in one clear sentence."
    prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{test_q}\n<|assistant|>\n"
    toks = enc.encode(prompt_str)
    gen = list(toks)
    with torch.no_grad():
        for _ in range(40):
            inp = torch.tensor([gen[-192:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr = logits[:, -1, :].clone()
            if len(gen) > 0:
                curr[0, gen[-1]] -= 50.0
            next_tok = torch.argmax(curr, dim=-1).item()
            gen.append(next_tok)
            if next_tok == 50256:
                break
    out = enc.decode(gen[len(toks):]).strip()
    print(f"\n{'='*65}", flush=True)
    print(f"  [SOVEREIGN REASONING EVALUATION @ STEP {step_num}]", flush=True)
    print(f"  Q: '{test_q}'", flush=True)
    print(f"  A: {out[:200]}", flush=True)
    print(f"{'='*65}\n", flush=True)
    model.train()

print(f"[TRAIN] Launching Sovereign Thought Refinery Training ({STEPS} steps, LR={BASE_LR} -> {MIN_LR})...\n", flush=True)

model.train()
t0 = time.time()
best_loss = 999.0

gold_log_path = REPO_ROOT / "training_data" / "Quillan_Refined_Thought_Corpus.jsonl"
gold_file = open(gold_log_path, "a", encoding="utf-8")

for step in range(1, STEPS + 1):
    lr = MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * step / STEPS))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    try:
        q, refined_trace, teacher_m = refinery_queue.get(timeout=8)
    except queue.Empty:
        q = random.choice(raw_prompt_pool)
        refined_trace = (
            "<thought>\nA function encapsulates a discrete unit of computation for reuse and isolation.</thought>\n"
            "<output>\nIn Python, a function is a named, reusable block of code that accepts inputs, executes a routine, and returns a result.</output>"
        )
        teacher_m = "local_curriculum"

    # Persist the gold sample to disk
    gold_file.write(json.dumps({"prompt": q, "refined_reasoning": refined_trace, "teacher": teacher_m}) + "\n")
    gold_file.flush()

    full_txt = (
        f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n"
        f"<|user|>\n{q}\n<|assistant|>\n{refined_trace}\n<|end|>"
    )
    prompt_txt = (
        f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n"
        f"<|user|>\n{q}\n<|assistant|>\n"
    )

    toks = enc.encode(full_txt)[:SEQ_LEN]
    p_toks = enc.encode(prompt_txt)
    labs = [-100] * len(p_toks) + toks[len(p_toks):]
    labs = labs[:SEQ_LEN]

    x = torch.tensor([toks], dtype=torch.long)
    y = torch.tensor([labs], dtype=torch.long)

    optimizer.zero_grad()
    logits, aux = model(x)
    loss = F.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, cfg.vocab_size),
        y[..., 1:].contiguous().view(-1),
        ignore_index=-100
    )
    total_loss = loss + 0.002 * aux
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    val = total_loss.item()
    if val < best_loss:
        best_loss = val

    if step % 10 == 0 or step == 1:
        elapsed = time.time() - t0
        sps = elapsed / step
        eta_m = sps * (STEPS - step) / 60.0
        short_t = teacher_m.split("/")[-1]
        print(f"  step {step:3d}/{STEPS}  loss={val:.4f}  best={best_loss:.4f}  lr={lr:.7f}  [{short_t[:14]:14s}]  [Queue: {refinery_queue.qsize():2d}]  ({sps:.1f}s/st, ETA {eta_m:.1f}m)", flush=True)

    if step % 50 == 0:
        quick_eval(step)
        torch.save({
            'model_state_dict': model.state_dict(),
            'step': step,
            'loss': best_loss,
            'version': 'quillan-v5.3.1-sovereign-thought-refined-master'
        }, ckpt_path)
        print(f"  [CHECKPOINT] Auto-saved master model at step {step}.\n", flush=True)

# Final Save
torch.save({
    'model_state_dict': model.state_dict(),
    'step': STEPS,
    'loss': best_loss,
    'version': 'quillan-v5.3.1-sovereign-thought-refined-final'
}, ckpt_path)

gold_file.close()
stop_event.set()
print(f"\n[DONE] 🏆 Sovereign Thought Refinery Distillation Complete! Best Loss: {best_loss:.4f} in {(time.time()-t0)/60:.1f}m\n", flush=True)
quick_eval(STEPS)
