#!/usr/bin/env python3
"""Does a down_proj lora_B write move Dragon's letter logits at all?"""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model
from wsp_math_tune_v3 import enable_cache, seed_lora_basis
import v23_common as vc

tok = vc.load_dragon_tokenizer()
lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in "ABCD"]
texts = ["The capital of France is", "Water freezes at 0 degrees", "2 + 2 equals"]
m = load_model(ROOT / "exports" / "quillan-dragon-packed.safetensors")
enable_cache(m)
down = m.layers[-1].down_proj
seed_lora_basis(m, [len(m.layers) - 1], 42)


def logs():
    o = []
    with torch.inference_mode():
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            o.append(m(torch.tensor([ids]))[0][0, -1, lids].clone())
    return torch.stack(o)


b = logs()
print("before:", [[round(float(v), 2) for v in row] for row in b])
g = torch.Generator().manual_seed(1)
with torch.no_grad():
    down.lora_B.copy_(torch.randn(down.lora_B.shape, generator=g) * 0.5)
a = logs()
print("after big B:", [[round(float(v), 2) for v in row] for row in a])
print("max abs change:", float((a - b).abs().max()))
