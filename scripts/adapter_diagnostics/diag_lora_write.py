#!/usr/bin/env python3
"""Diagnose whether a lora_B write actually moves the 4 letter logits."""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import format_problem
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import seed_lora_basis, enable_cache

tok = tok_for(151936)
lids = letter_ids(tok, 151936)
ids = tok.encode(format_problem(0), add_special_tokens=False)
x = torch.tensor([ids])
m = load_model(ROOT / "exports" / "quillan-mini-dense-council.safetensors")
seed_lora_basis(m, [23], 42)
enable_cache(m)
down = m.layers[-1].down_proj
with torch.inference_mode():
    lg0 = m(x)[0][0, -1, lids]
print("before:", [round(float(v), 3) for v in lg0], "argmax", int(torch.argmax(lg0)))
# write a big random lora_B
g = torch.Generator().manual_seed(7)
with torch.no_grad():
    down.lora_B.copy_(torch.randn(down.lora_B.shape, generator=g) * 1.0)
with torch.inference_mode():
    lg1 = m(x)[0][0, -1, lids]
print(
    "after big random B:",
    [round(float(v), 3) for v in lg1],
    "argmax",
    int(torch.argmax(lg1)),
)
# now zero B and write rank-1 aligned with E[target]
with torch.no_grad():
    down.lora_B.zero_()
    E = m.embed_tokens.unpack_float().float()
    A = down.lora_A.float()
    # target: boost letter C (index 2) using its embedding direction through down_proj
    tgt = E[lids[2]]
    # dB such that (A@x) @ dB.T ~ tgt  -> dB = outer(tgt, k)/||k||^2 * scale
    k = (
        A @ m.layers[-1].down_proj._lx[0, -1]
        if hasattr(down, "_lx")
        else A @ torch.randn(A.shape[1])
    )
    dB = torch.outer(tgt, k) / (k @ k + 1e-8) * 0.01
    down.lora_B.copy_(dB)
with torch.inference_mode():
    lg2 = m(x)[0][0, -1, lids]
print(
    "after targeted B:",
    [round(float(v), 3) for v in lg2],
    "argmax",
    int(torch.argmax(lg2)),
)
