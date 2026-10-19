#!/usr/bin/env python3
import os, sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

enc = tiktoken.get_encoding("gpt2")
ckpt_path = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = QuillanUnrolledSovereign(ckpt["cfg"]).to("cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("==================================================================", flush=True)
print("   👑 12-LAYER UNROLLED SOVEREIGN MASTER CHECKPOINT AUDIT", flush=True)
print(f"   Checkpoint: {ckpt_path.name} | Step: {ckpt.get('step', 'N/A')} | Loss: {ckpt.get('loss', 'N/A')}", flush=True)
print("==================================================================", flush=True)

prompts = [
    "User: Hello! What are your primary capabilities as an AI?\nAssistant: I am Quillan, an AI capable of",
    "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\ndef is_palindrome(s):\n    return",
    "Question: What is photosynthesis?\nAnswer:\nPhotosynthesis is the biological process where"
]

for p in prompts:
    print(f"\nPROMPT:\n{p}", flush=True)
    toks = enc.encode(p)
    t0 = time.time()
    out = model.generate(toks, max_tokens=45, temp=0.55, top_p=0.85, repetition_penalty=1.12)
    ans = enc.decode(out).strip()
    print(f"RESPONSE ({time.time()-t0:.2f}s):\n{ans}\n{'-'*65}", flush=True)
