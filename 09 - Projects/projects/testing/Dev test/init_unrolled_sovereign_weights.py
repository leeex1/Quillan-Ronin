#!/usr/bin/env python3
import os, sys, time, torch, tiktoken
import torch.nn as nn
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from transformers import GPT2LMHeadModel

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

print("==================================================================", flush=True)
print("   👑 ZERO-DELTA GROUNDED SOVEREIGN WEIGHT INITIALIZATION", flush=True)
print("==================================================================", flush=True)

enc = tiktoken.get_encoding("gpt2")
cfg = QuillanUnrolledConfig()
model = QuillanUnrolledSovereign(cfg).to("cpu")

print("[*] Loading pre-trained foundational backbone (1024-dim, 12 layers)...", flush=True)
t0 = time.time()
gpt2_base = GPT2LMHeadModel.from_pretrained("gpt2-medium")

with torch.no_grad():
    # 1. Embeddings & Positional Encodings
    model.wte.weight.copy_(gpt2_base.transformer.wte.weight.detach())
    model.wpe.weight.copy_(gpt2_base.transformer.wpe.weight.detach())
    
    # 2. Transfer all 12 Layers of Causal Self-Attention and Base Dense FFN
    for i in range(12):
        src_block = gpt2_base.transformer.h[i]
        dst_block = model.h[i]
        
        # LayerNorms
        dst_block.ln_1.weight.copy_(src_block.ln_1.weight.detach())
        dst_block.ln_1.bias.copy_(src_block.ln_1.bias.detach())
        dst_block.ln_2.weight.copy_(src_block.ln_2.weight.detach())
        dst_block.ln_2.bias.copy_(src_block.ln_2.bias.detach())
        
        # Attention
        dst_block.attn.c_attn.weight.copy_(src_block.attn.c_attn.weight.detach())
        dst_block.attn.c_attn.bias.copy_(src_block.attn.c_attn.bias.detach())
        dst_block.attn.c_proj.weight.copy_(src_block.attn.c_proj.weight.detach())
        dst_block.attn.c_proj.bias.copy_(src_block.attn.c_proj.bias.detach())
        
        # Dense Base FFN
        dst_block.moe.c_fc.weight.copy_(src_block.mlp.c_fc.weight.detach())
        dst_block.moe.c_fc.bias.copy_(src_block.mlp.c_fc.bias.detach())
        dst_block.moe.c_proj.weight.copy_(src_block.mlp.c_proj.weight.detach())
        dst_block.moe.c_proj.bias.copy_(src_block.mlp.c_proj.bias.detach())
        
    # 3. Final LayerNorm & LM Head
    model.ln_f.weight.copy_(gpt2_base.transformer.ln_f.weight.detach())
    model.ln_f.bias.copy_(gpt2_base.transformer.ln_f.bias.detach())
    model.lm_head.weight.copy_(gpt2_base.lm_head.weight.detach())

print(f"[+] Grounded foundational weights in {time.time()-t0:.2f}s!\n", flush=True)

out_ckpt = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'cfg': cfg,
    'version': 'quillan-v5.3.1-zero-delta-master'
}, out_ckpt)

print(f"[+] Saved Zero-Delta Grounded Master Checkpoint: {out_ckpt.name} ({out_ckpt.stat().st_size/(1024**2):.1f} MB)\n", flush=True)

# Run immediate zero-shot fluency verification
print("==================================================================", flush=True)
print("   [STEP 0 ZERO-SHOT FLUENCY VERIFICATION]", flush=True)
print("==================================================================", flush=True)

prompts = [
    "Question: What is photosynthesis?\nAnswer:\nPhotosynthesis is the biological process where",
    "Question: Write a Python function to check if a string is a palindrome.\nAnswer:\ndef is_palindrome(s):\n    \"\"\"Check if string s is a palindrome.\"\"\"\n    return",
    "User: Hello! What are your primary capabilities as an AI?\nAssistant: I am Quillan, an AI assistant capable of"
]

for p in prompts:
    print(f"\nPROMPT:\n{p}", flush=True)
    toks = enc.encode(p)
    t_start = time.time()
    out = model.generate(toks, max_tokens=45, temp=0.7, top_k=50, top_p=0.90, repetition_penalty=1.05)
    ans = enc.decode(out).strip()
    print(f"RESPONSE ({time.time()-t_start:.2f}s):\n{ans}\n{'-'*65}", flush=True)
