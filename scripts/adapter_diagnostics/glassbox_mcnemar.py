#!/usr/bin/env python3
"""McNemar exact on the 80-item paired large bank, base vs frozen attention."""

from __future__ import annotations
import sys, math
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from heldout_domain_large import BANKS, prompts

torch.set_num_threads(1)


def score(m, tok, lids, dom):
    Q, K = BANKS[dom]
    out = []
    with torch.inference_mode():
        for seq, gold in zip(prompts(tok, dom), K):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            out.append(int(int(lg.argmax()) == gold))
    return out


def main():
    A = load_model(ROOT / "exports" / "quillan-mini-glassbox-base.safetensors")
    enable_cache(A)
    B = load_model(ROOT / "exports" / "quillan-mini-glassbox-attn.safetensors")
    enable_cache(B)
    tok = tok_for(A.cfg.vocab_size)
    lids = letter_ids(tok, A.cfg.vocab_size)
    a = []
    b = []
    for dom in BANKS:
        a += score(A, tok, lids, dom)
        b += score(B, tok, lids, dom)
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    k = min(b01, b10)
    p = min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / (2**n) * 2) if n else 1.0
    print(
        f"base={sum(a)}/80 attn={sum(b)}/80 | fixed={b01} broke={b10} n={n} | McNemar p={p:.5f} {'PASS<0.05' if p < 0.05 else 'FAIL'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
