#!/usr/bin/env python3
import os, sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v9_dense_sovereign import QuillanDenseSovereign, QuillanDenseConfig

enc = tiktoken.get_encoding("gpt2")
ckpt_path = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = QuillanDenseSovereign(ckpt["cfg"]).to("cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompts = [
    "Question: What is photosynthesis?\nAnswer:",
    "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\ndef is_palindrome(s):",
    "User: Hello! What are your primary capabilities?\nAssistant: I am Quillan, an AI assistant capable of"
]

print("==================================================================", flush=True)
print("   👑 12-LAYER DENSE SOVEREIGN TRANSFORMER GENERATION TEST", flush=True)
print("==================================================================", flush=True)

for p in prompts:
    print(f"\nPROMPT:\n{p}", flush=True)
    toks = enc.encode(p)
    t0 = time.time()
    out = model.generate(toks, max_tokens=50, temp=0.7, top_p=0.85, repetition_penalty=1.15)
    ans = enc.decode(out).strip()
    print(f"GENERATED ({time.time()-t0:.2f}s):\n{ans}\n{'-'*65}", flush=True)
