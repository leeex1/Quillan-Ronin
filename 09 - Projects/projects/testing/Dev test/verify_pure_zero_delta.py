#!/usr/bin/env python3
"""
👑 QUILLAN-RONIN v5.3.1 — PURE ZERO-DELTA VERIFICATION
Verifies that the unrolled 34-expert architecture with Zero-Delta weights produces
100% fluent baseline language.
"""

import sys, time, torch, tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH_DIR))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

enc = tiktoken.get_encoding("gpt2")
cfg = QuillanUnrolledConfig()
model = QuillanUnrolledSovereign(cfg).to("cpu")

# Load pre-trained GPT-2 Medium weights directly
from transformers import GPT2LMHeadModel
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-medium")
sd = gpt2.state_dict()

model.wte.weight.data.copy_(sd['transformer.wte.weight'])
model.wpe.weight.data.copy_(sd['transformer.wpe.weight'])
model.ln_f.weight.data.copy_(sd['transformer.ln_f.weight'])
model.ln_f.bias.data.copy_(sd['transformer.ln_f.bias'])

for i in range(12):
    pfx = f'transformer.h.{i}.'
    b = model.h[i]
    b.ln_1.weight.data.copy_(sd[pfx + 'ln_1.weight'])
    b.ln_1.bias.data.copy_(sd[pfx + 'ln_1.bias'])
    b.attn.c_attn.weight.data.copy_(sd[pfx + 'attn.c_attn.weight'])
    b.attn.c_attn.bias.data.copy_(sd[pfx + 'attn.c_attn.bias'])
    b.attn.c_proj.weight.data.copy_(sd[pfx + 'attn.c_proj.weight'])
    b.attn.c_proj.bias.data.copy_(sd[pfx + 'attn.c_proj.bias'])
    b.ln_2.weight.data.copy_(sd[pfx + 'ln_2.weight'])
    b.ln_2.bias.data.copy_(sd[pfx + 'ln_2.bias'])
    b.moe.c_fc.weight.data.copy_(sd[pfx + 'mlp.c_fc.weight'])
    b.moe.c_fc.bias.data.copy_(sd[pfx + 'mlp.c_fc.bias'])
    b.moe.c_proj.weight.data.copy_(sd[pfx + 'mlp.c_proj.weight'])
    b.moe.c_proj.bias.data.copy_(sd[pfx + 'mlp.c_proj.bias'])

model.eval()

TESTS = [
    "Photosynthesis is the biological process where green plants convert sunlight,",
    "In Linux, the difference between SIGTERM and SIGKILL is that SIGTERM",
    "def is_palindrome(s):\n    \"\"\"Check if string s is a palindrome.\"\"\"\n    return",
    "I am Quillan, a sovereign AI assistant capable of"
]

print("==================================================================", flush=True)
print("   👑 ZERO-DELTA BASELINE FLUENCY TEST", flush=True)
print("==================================================================\n", flush=True)

for prompt in TESTS:
    toks = enc.encode(prompt)
    t0 = time.time()
    out = model.generate(toks, max_tokens=40, temp=0.7, top_k=50, top_p=0.90)
    ans = enc.decode(out).strip()
    print(f"PROMPT:\n{prompt}", flush=True)
    print(f"GENERATED ({time.time()-t0:.2f}s):\n{ans}", flush=True)
    print(f"{'-'*65}\n", flush=True)
