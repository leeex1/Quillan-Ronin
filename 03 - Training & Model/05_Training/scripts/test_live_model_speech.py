#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — LIVE SPEECH & REASONING AUDITOR (PRECISION BEST)
Tests the landmark Loss ~1.00 checkpoint with stateful KV-caching.
"""

import sys
import time
import torch
from pathlib import Path

SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft")

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ckpt_path = CKPT_DIR / "quillan_gold_precision_best.pt"
if not ckpt_path.exists():
    ckpt_path = CKPT_DIR / "quillan_thinking_reasoning_master.pt"

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — LANDMARK LOSS ~1.00 SPEECH AUDIT", flush=True)
print(f"   Checkpoint: {ckpt_path.name}", flush=True)
print("==================================================================\n", flush=True)

device = torch.device("cpu")
tokenizer = SovereignTokenizer("gpt2")
cfg = QuillanUnrolledConfig()

model = QuillanUnrolledSovereign(cfg).to(device)
print(f"[*] Loading precision checkpoint weights from {ckpt_path}...", flush=True)
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"[+] Precision checkpoint loaded successfully (Recorded Loss: {ckpt.get('loss', 'N/A'):.4f})!\n", flush=True)

TEST_PROMPTS = [
    ("Identity & Council", "Question: Hello! Who are you and how do your 34 Council Experts work?\nAnswer:\n"),
    ("Python Palindrome", "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\n"),
    ("Photosynthesis", "Question: What is photosynthesis?\nAnswer:\n"),
    ("Linux Signals", "Question: What is the difference between SIGTERM (15) and SIGKILL (9) in Linux?\nAnswer:\n"),
    ("Formal Logic", "Question: If all A are B and all B are C, are all A necessarily C?\nAnswer:\n"),
    ("Security CWE-89", "Question: How do parameterized queries prevent SQL injection vulnerabilities?\nAnswer:\n"),
    ("Physics / Relativity", "Question: What is Einstein's mass-energy equivalence equation and what does it mean?\nAnswer:\n"),
]

for idx, (domain, prompt) in enumerate(TEST_PROMPTS, 1):
    toks = tokenizer.encode(prompt)
    t0 = time.time()
    out_toks = model.generate(
        toks,
        max_tokens=80,
        temp=0.45,
        top_k=40,
        top_p=0.85,
        frequency_penalty=0.40,
        presence_penalty=0.30,
    )
    elapsed = time.time() - t0
    
    gen_toks = out_toks[len(toks):]
    gen_text = tokenizer.decode(gen_toks).strip()
    gen_text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    
    tps = len(gen_toks) / max(0.001, elapsed)

    print(f"[{idx}/{len(TEST_PROMPTS)}] DOMAIN: {domain}", flush=True)
    print(f"PROMPT:\n{prompt.strip()}", flush=True)
    print(f"GENERATED ({elapsed:.2f}s, {tps:.1f} tok/s):\n{gen_text}", flush=True)
    print(f"{'-'*65}\n", flush=True)

print("==================================================================", flush=True)
print("   🏆 LANDMARK LOSS ~1.00 EVALUATION COMPLETE", flush=True)
print("==================================================================", flush=True)
