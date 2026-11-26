#!/usr/bin/env python3
"""Paired McNemar: base vs r64 across all 120 held-out items.
Also computes exact binomial p on discordant pairs (no scipy dependency).
"""

from __future__ import annotations
import sys, math
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts as p1, KEY as K1
from heldout_fresh40b import prompts as p2, KEY as K2
from heldout_fresh40c import prompts as p3, KEY as K3

torch.set_num_threads(1)
BANKS = [("fresh40", p1, K1), ("fresh40b", p2, K2), ("fresh40c", p3, K3)]


def correctness(ck):
    m = load_model(Path(ck))
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    res = []
    with torch.inference_mode():
        for name, pf, KEY in BANKS:
            for seq, gold in zip(pf(tok), KEY):
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                res.append(int(int(lg.argmax()) == gold))
    return res


def main():
    base = correctness(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    r64 = correctness(ROOT / "exports" / "quillan-mini-hirank64.safetensors")
    b01 = sum(
        1 for a, b in zip(base, r64) if a == 0 and b == 1
    )  # base wrong, r64 right
    b10 = sum(
        1 for a, b in zip(base, r64) if a == 1 and b == 0
    )  # base right, r64 wrong
    n = b01 + b10
    # exact two-sided binomial p under H0 p=0.5
    if n > 0:
        k = min(b01, b10)
        p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n) * 2
        p = min(1.0, p)
    else:
        p = 1.0
    print(f"base_correct={sum(base)}/120 r64_correct={sum(r64)}/120", flush=True)
    print(
        f"discordant: base_wrong->r64_right={b01}, base_right->r64_wrong={b10}, n={n}",
        flush=True,
    )
    print(f"McNemar exact two-sided p = {p:.5f}", flush=True)


if __name__ == "__main__":
    main()
