#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — IN-DEPTH PROMPT CONTINUATION AUDIT
Evaluates greedy and nucleus sampling across both prefix-completion and QA formats.
"""

import sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

REPO_ROOT = Path(r"C:\02_QUILLAN")
ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"

enc = tiktoken.get_encoding("gpt2")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = QuillanUnrolledSovereign(ckpt["cfg"]).to("cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

TESTS = [
    # Prefix continuation
    ("Photosynthesis Definition", "Photosynthesis is the biological process where green plants convert sunlight,"),
    ("Python Palindrome", "def is_palindrome(s):\n    \"\"\"Check if string s is a palindrome.\"\"\"\n    return"),
    ("Linux Signals", "In Linux, the key difference between SIGTERM (15) and SIGKILL (9) is that"),
    ("BitNet Ternary", "BitNet 1.58b quantizes model weights to ternary values {-1, 0, +1}, replacing"),
    ("LanceDB IVF-PQ", "LanceDB achieves sub-millisecond retrieval by using Inverted File with"),
    ("SQL Injection", "Parameterized queries prevent SQL injection vulnerabilities by separating"),
    ("Logic Rose", "If all roses are flowers and some flowers fade quickly, then"),
    ("AI Capabilities", "I am Quillan, a sovereign AI assistant capable of")
]

print("==================================================================", flush=True)
print("   👑 IN-DEPTH PROMPT CONTINUATION AUDIT", flush=True)
print("==================================================================\n", flush=True)

for name, prompt in TESTS:
    toks = enc.encode(prompt)
    t0 = time.time()
    out = model.generate(toks, max_tokens=40, temp=0.5, top_k=30, top_p=0.85, repetition_penalty=1.1, frequency_penalty=0.5, presence_penalty=0.3)
    elapsed = time.time() - t0
    gen_text = enc.decode(out).strip()
    print(f"[{name}]", flush=True)
    print(f"PROMPT: {prompt}", flush=True)
    print(f"GENERATED ({elapsed:.2f}s):\n{gen_text}", flush=True)
    print(f"{'-'*65}\n", flush=True)
