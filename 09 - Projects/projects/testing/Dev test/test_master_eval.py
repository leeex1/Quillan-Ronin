#!/usr/bin/env python3
"""
Diagnostic script to test conversational completion and inspect exact token output.
"""
import sys, time, torch, tiktoken
from pathlib import Path

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()

queries = [
    "Hello! What is your name?",
    "Explain what a function is in Python in one clear sentence."
]

print("=== LIVE GENERATION DIAGNOSTIC ===", flush=True)
for q in queries:
    prompt = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{q}\n<|assistant|>\n"
    tokens = enc.encode(prompt)
    out = model.generate(tokens, max_tokens=30, temp=0.7, top_p=0.9, repetition_penalty=1.2)
    decoded = enc.decode(out).strip()
    print(f"\n[QUERY]: {q}", flush=True)
    print(f"[OUTPUT]: {decoded}", flush=True)
