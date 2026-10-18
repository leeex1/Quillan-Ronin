#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — 10-QUESTION MASTER BENCHMARK SUITE (UNROLLED SOVEREIGN)
Tests the 12-Layer Unrolled Sovereign Transformer with 34 Unique Council Expert Channels across:
1. Identity & Self-Awareness
2. Python Algorithms & Data Structures
3. Science & Photosynthesis
4. Linux Systems & Signals
5. Chain-of-Thought Logical Deduction
6. Hardware Acceleration & BitNet
7. Database & LanceDB Vector Search
8. Network Protocols & API Design
9. Security & Memory Hardening
10. Creative Reasoning & Synthesis
"""

import os, sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

enc = tiktoken.get_encoding("gpt2")
ckpt_path = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — 10-QUESTION MASTER BENCHMARK SUITE", flush=True)
print(f"   Model: 12-Layer Unrolled Sovereign (34 Unique Experts) | {ckpt_path.name}", flush=True)
print("==================================================================\n", flush=True)

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = QuillanUnrolledConfig()
model = QuillanUnrolledSovereign(cfg).to("cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

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

results = []
for idx, (cat, prompt) in enumerate(BENCHMARK_QUESTIONS, 1):
    toks = enc.encode(prompt)
    t0 = time.time()
    out = model.generate(toks, max_tokens=50, temp=0.55, top_p=0.85, repetition_penalty=1.12)
    elapsed = time.time() - t0
    gen_text = enc.decode(out).strip()
    
    print(f"[{idx}/10] CATEGORY: {cat}", flush=True)
    print(f"PROMPT:\n{prompt}", flush=True)
    print(f"GENERATED ({elapsed:.2f}s, {len(out)/max(0.001, elapsed):.1f} tok/s):\n{gen_text}", flush=True)
    print(f"{'-'*65}\n", flush=True)
    results.append((idx, cat, prompt, gen_text, elapsed))

print("==================================================================", flush=True)
print("   🏆 10-QUESTION MASTER BENCHMARK EVALUATION COMPLETE", flush=True)
print("==================================================================", flush=True)
