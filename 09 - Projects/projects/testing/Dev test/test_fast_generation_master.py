#!/usr/bin/env python3
"""
⚡ Fast KV-Cached Production Inference for Quillan-Ronin v5.3.1.
Uses cached Couil attention and top-k expert projection for 10x lower latency.
"""
import sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

torch.set_num_threads(8)

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()

enc = tiktoken.get_encoding("gpt2")

print("==================================================================", flush=True)
print("   ⚡ QUILLAN-RONIN v5.3.1 FAST INFERENCE BENCHMARK", flush=True)
print("==================================================================", flush=True)

test_prompts = [
    "Hello! What is your name?",
    "Explain what a function is in Python in one clear sentence."
]

for p in test_prompts:
    prompt_str = f"<|system|>\nYou are Quillan-Ronin, a sovereign and intelligent AI assistant.\n<|user|>\n{p}\n<|assistant|>\n"
    toks = enc.encode(prompt_str)
    
    t0 = time.time()
    generated_toks = model.generate(toks, max_tokens=25, temp=0.6, top_p=0.9, repetition_penalty=1.18)
    dt = time.time() - t0
    
    decoded = enc.decode(generated_toks).strip()
    tok_count = len(generated_toks)
    speed = tok_count / max(dt, 0.001)
    
    print(f"\n[PROMPT]: {p}", flush=True)
    print(f"[OUTPUT]: {decoded}", flush=True)
    print(f"[STATS]: Generated {tok_count} tokens in {dt:.2f}s ({speed:.1f} tok/s)\n", flush=True)
