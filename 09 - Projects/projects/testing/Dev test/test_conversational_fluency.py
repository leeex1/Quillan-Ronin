#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — PRODUCTION CONVERSATIONAL FLUENCY TEST
Tests model responses across diverse natural prompts using normalized spiderweb architecture.
"""
import os, sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

enc = tiktoken.get_encoding("gpt2")

print("==================================================================", flush=True)
print("   👑 QUILLAN-RONIN v5.3.1 — CONVERSATIONAL INFERENCE TEST", flush=True)
print("==================================================================", flush=True)

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Aligned Master Checkpoint: {ckpt_path.name}", flush=True)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
print(f"[+] Loaded successfully (Step: {ckpt.get('step','N/A')}, Loss: {ckpt.get('loss','N/A')})\n", flush=True)

model.eval()

TEST_PROMPTS = [
    "Hello! What is your name and how can you help me today?",
    "Explain what a function is in Python in simple terms."
]

for i, user_q in enumerate(TEST_PROMPTS, 1):
    prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{user_q}\n<|assistant|>\n"
    prompt_tokens = enc.encode(prompt_str)
    
    print(f"==================================================================", flush=True)
    print(f"[{i}/{len(TEST_PROMPTS)}] USER: {user_q}", flush=True)
    print(f"------------------------------------------------------------------", flush=True)
    t0 = time.time()
    generated_tokens = model.generate(prompt_tokens, max_tokens=35, temp=0.5, top_p=0.9, repetition_penalty=1.15)
    response_text = enc.decode(generated_tokens).strip()
    latency = time.time() - t0
    print(f"ASSISTANT:\n{response_text}", flush=True)
    print(f"({len(generated_tokens)} tokens in {latency:.1f}s | {len(generated_tokens)/max(latency,0.01):.1f} tok/s)\n", flush=True)

print("==================================================================", flush=True)
print("   🏆 CONVERSATIONAL INFERENCE TEST COMPLETE", flush=True)
print("==================================================================", flush=True)
