#!/usr/bin/env python3
"""Falsify the rep=0 artifact: print ACTUAL generated text at each council gate."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, load_model
from wsp_math_tune_v3 import enable_cache

m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
enable_cache(m)
moes = [l.council_moe for l in m.layers if l.council_moe is not None]
tok = tok_for(151936)
for alpha in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
    for moe in moes:
        with torch.no_grad():
            moe.gate_alpha.fill_(alpha)
    ids = tok.encode("The meaning of life is", add_special_tokens=False)
    with torch.inference_mode():
        out = m.generate(torch.tensor([ids]), max_new_tokens=40, temperature=0.0)
    g = out[0, len(ids) :].tolist()
    txt = tok.decode(g, skip_special_tokens=True)
    safe = txt.encode("ascii", "backslashreplace").decode("ascii")
    print(f"[gate={alpha}] ntok={len(g)} text={safe[:160]!r}", flush=True)
