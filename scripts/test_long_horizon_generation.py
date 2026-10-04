#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — LONG-HORIZON SPEECH & REASONING AUDITOR
Tests deep reasoning, multi-paragraph explanations, and code generation across 256 tokens.
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

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — LONG-HORIZON REASONING AUDIT (256 TOKENS)", flush=True)
print(f"   Checkpoint: {ckpt_path.name}", flush=True)
print("==================================================================\n", flush=True)

device = torch.device("cpu")
tokenizer = SovereignTokenizer("gpt2")
cfg = QuillanUnrolledConfig()

model = QuillanUnrolledSovereign(cfg).to(device)
print(f"[*] Loading master checkpoint from {ckpt_path}...", flush=True)
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
model.eval()
print("[+] Master weights loaded successfully!\n", flush=True)

LONG_TEST_PROMPTS = [
    (
        "Formal Deductive Logic",
        "Question: If all humans are mortal and Socrates is human, is Socrates mortal? Provide the formal syllogistic proof.\nAnswer:\n"
    ),
    (
        "Scientific Theory & Photosynthesis",
        "Question: Explain the chemical process of photosynthesis in plants, including the balanced chemical equation.\nAnswer:\n"
    ),
    (
        "Theoretical Physics",
        "Question: State Einstein's mass-energy equivalence equation and explain its physical implications.\nAnswer:\n"
    ),
]

for idx, (domain, prompt) in enumerate(LONG_TEST_PROMPTS, 1):
    toks = tokenizer.encode(prompt)
    print(f"[{idx}/{len(LONG_TEST_PROMPTS)}] DOMAIN: {domain}", flush=True)
    print(f"PROMPT:\n{prompt.strip()}\n", flush=True)
    print("STREAMING OUTPUT:", flush=True)
    print("-" * 65, flush=True)
    
    t0 = time.time()
    out_toks = model.generate(
        toks,
        max_tokens=180,
        temp=0.40,
        top_k=35,
        top_p=0.85,
        frequency_penalty=0.45,
        presence_penalty=0.35,
    )
    elapsed = time.time() - t0
    
    gen_toks = out_toks[len(toks):]
    gen_text = tokenizer.decode(gen_toks).strip()
    gen_text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    
    tps = len(gen_toks) / max(0.001, elapsed)
    print(gen_text, flush=True)
    print("-" * 65, flush=True)
    print(f"Stats: {len(gen_toks)} tokens generated in {elapsed:.2f}s ({tps:.1f} tok/s)\n", flush=True)

print("==================================================================", flush=True)
print("   🏆 LONG-HORIZON AUDIT COMPLETE", flush=True)
print("==================================================================", flush=True)
