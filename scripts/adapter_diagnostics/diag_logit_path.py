#!/usr/bin/env python3
"""Find which writable parameter most directly controls the 4 letter logits."""

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
enable_cache(m)


def lg():
    with torch.inference_mode():
        return m(x)[0][0, -1, lids].clone()


base = lg()
print("base:", [round(float(v), 2) for v in base])
# 1) Does final RMSNorm weight move logits?
w = m.norm.weight.detach().clone()
with torch.no_grad():
    m.norm.weight.copy_(w * 1.5)
print("norm*1.5:", [round(float(v), 2) for v in lg()])
with torch.no_grad():
    m.norm.weight.copy_(w)
# 2) Does a down_proj bias exist / can we add one effectively via lora_B constant?
# The lora path is F.linear(F.linear(x,A),B). If A@x is ~constant across the last
# token, a constant lora_B column acts like a bias. Check variance of k=A@x.
down = m.layers[-1].down_proj
if down.lora_A.abs().max().item() == 0:
    seed_lora_basis(m, [23], 42)
ks = []
for i in range(20):
    xi = tok.encode(format_problem(i), add_special_tokens=False)
    with torch.inference_mode():
        m(torch.tensor([xi]))
    ks.append(down._wc_x if hasattr(down, "_wc_x") else None)
print(
    "lora_A shape", tuple(down.lora_A.shape), "lora_B shape", tuple(down.lora_B.shape)
)
# 3) Direct: can editing the embedding row of a letter change its logit? (tied head)
emb = m.embed_tokens
print("embed type", type(emb).__name__, "has weight_int8", hasattr(emb, "weight_int8"))
print("unpack shape", tuple(emb.unpack_float().shape))
