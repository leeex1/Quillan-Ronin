#!/usr/bin/env python3
import os, sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from transformers import GPT2LMHeadModel

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v9_dense_sovereign import QuillanDenseSovereign, QuillanDenseConfig

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — EXACT DENSE WEIGHT INITIALIZATION", flush=True)
print("==================================================================", flush=True)

enc = tiktoken.get_encoding("gpt2")
cfg = QuillanDenseConfig()
model = QuillanDenseSovereign(cfg).to("cpu")

print("[*] Loading pre-trained GPT-2 Medium 12-layer backbone (1024-dim)...", flush=True)
gpt2_base = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# Copy base weights exactly
gpt2_sd = gpt2_base.state_dict()
model_sd = model.state_dict()

matched = 0
for k, v in gpt2_sd.items():
    # Map 'transformer.' prefix to model attributes
    target_k = k.replace("transformer.", "")
    if target_k in model_sd and model_sd[target_k].shape == v.shape:
        model_sd[target_k].copy_(v)
        matched += 1

model.load_state_dict(model_sd)
print(f"[+] Successfully transferred {matched} exact tensor layers into Quillan Sovereign model!\n", flush=True)

out_ckpt = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': cfg,
    'version': 'quillan-v5.3.1-dense-exact-master-init'
}, out_ckpt)

print(f"[+] Saved exact model to: {out_ckpt.name}\n", flush=True)

# Immediate generation test
print("==================================================================", flush=True)
print("   [EXACT ZERO-SHOT DENSE GENERATION TEST]", flush=True)
print("==================================================================", flush=True)

prompts = [
    "Question: What is photosynthesis?\nAnswer:",
    "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\ndef is_palindrome(s):",
    "User: Hello! What are your primary capabilities as an AI?\nAssistant: I am Quillan, an AI capable of"
]

for p in prompts:
    print(f"\nPROMPT:\n{p}", flush=True)
    toks = enc.encode(p)
    t0 = time.time()
    out = model.generate(toks, max_tokens=50, temp=0.7, top_p=0.85, repetition_penalty=1.12)
    ans = enc.decode(out).strip()
    print(f"RESPONSE ({time.time()-t0:.2f}s):\n{ans}\n{'-'*65}", flush=True)
