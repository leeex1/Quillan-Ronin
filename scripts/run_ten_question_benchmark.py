#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — 10-QUESTION MASTER BENCHMARK SUITE
Evaluates the 12-Layer Unrolled Sovereign Transformer with hardened decoding & repetition suppression.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig
from sovereign_inference_engine import SovereignInferenceEngine, SovereignTokenizer, SamplingParams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ckpt_path = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — HARDENED MASTER BENCHMARK SUITE", flush=True)
print(f"   Model: 12-Layer Unrolled Sovereign | Checkpoint: {ckpt_path.name}", flush=True)
print("==================================================================\n", flush=True)

tokenizer = SovereignTokenizer("gpt2")
engine = SovereignInferenceEngine.load_from_checkpoint(
    model_factory=lambda: QuillanUnrolledSovereign(QuillanUnrolledConfig()),
    checkpoint_path=ckpt_path,
    device="cpu",
)

BENCHMARK_QUESTIONS = [
    ("Identity", "User: Hello! What are your primary capabilities as an AI?\nAssistant: I am Quillan, an AI assistant capable of"),
    ("Python Palindrome", "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\ndef is_palindrome(s):\n    \"\"\"Check if s is palindrome.\"\"\"\n    return s =="),
    ("Science (Photosynthesis)", "Question: What is photosynthesis?\nAnswer:\nPhotosynthesis is the biological process where green plants convert"),
    ("Linux DevOps", "Question: What is the difference between SIGTERM (15) and SIGKILL (9) in Linux?\nAnswer:\nIn Linux, the key difference between SIGTERM and SIGKILL is that SIGTERM"),
    ("Logic / Deduction", "Question: If all roses are flowers and some flowers fade quickly, do all roses fade quickly?\nAnswer:\nLogical Analysis:"),
    ("Hardware Acceleration", "Question: How does BitNet 1.58-bit ternary quantization reduce memory bandwidth?\nAnswer:\nBitNet ternary quantization replaces"),
    ("Database / LanceDB", "Question: How does vector indexing in LanceDB achieve sub-millisecond retrieval?\nAnswer:\nLanceDB utilizes IVF-PQ vector indexing to"),
    ("API Design", "Question: What are the core architectural constraints of RESTful APIs?\nAnswer:\nRESTful API architecture enforces"),
    ("Security & CWE", "Question: How do parameterized queries prevent SQL injection vulnerabilities?\nAnswer:\nParameterized queries separate"),
    ("Creative Synthesis", "Question: Synthesize the relationship between entropy in physics and information theory.\nAnswer:\nBoth thermodynamic entropy and Shannon information entropy quantify")
]

params = SamplingParams(
    max_new_tokens=60,
    temperature=0.60,
    top_k=40,
    top_p=0.85,
    min_p=0.05,
    repetition_penalty=1.25,
    frequency_penalty=0.40,
    presence_penalty=0.30,
    no_repeat_ngram_size=3,
    stop_strings=("\nQuestion:", "\nUser:", "<|im_end|>"),
)

results = []
for idx, (cat, prompt) in enumerate(BENCHMARK_QUESTIONS, 1):
    t0 = time.time()
    gen_text = engine.generate(prompt, params=params)
    elapsed = time.time() - t0
    
    tokens_gen = len(tokenizer.encode(gen_text))
    tps = tokens_gen / max(0.001, elapsed)

    print(f"[{idx}/10] CATEGORY: {cat}", flush=True)
    print(f"PROMPT:\n{prompt}", flush=True)
    print(f"GENERATED ({elapsed:.2f}s, {tps:.1f} tok/s):\n{gen_text}", flush=True)
    print(f"{'-'*65}\n", flush=True)
    results.append((idx, cat, prompt, gen_text, elapsed, tps))

print("==================================================================", flush=True)
print("   🏆 HARDENED MASTER BENCHMARK EVALUATION COMPLETE", flush=True)
print("==================================================================", flush=True)
