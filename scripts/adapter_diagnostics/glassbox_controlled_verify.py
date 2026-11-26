#!/usr/bin/env python3
"""CONTROLLED A/B: base vs frozen glassbox-attn, pinned 1 thread, paired per-item.
Sets torch threads ONCE after all imports (bank modules no longer override).
"""

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
from domain_banks import BANKS, prompts

torch.set_num_threads(1)  # single pinned regime, after all imports


def score(m, tok, lids, domain):
    Q, K = BANKS[domain]
    bits = []
    with torch.inference_mode():
        for seq, gold in zip(prompts(tok, domain), K):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            bits.append(int(int(lg.argmax()) == gold))
    return bits


def main():
    base = load_model(ROOT / "exports" / "quillan-mini-glassbox-base.safetensors")
    enable_cache(base)
    attn = load_model(ROOT / "exports" / "quillan-mini-glassbox-attn.safetensors")
    enable_cache(attn)
    tok = tok_for(base.cfg.vocab_size)
    lids = letter_ids(tok, base.cfg.vocab_size)
    for dom in BANKS:
        b = score(base, tok, lids, dom)
        a = score(attn, tok, lids, dom)
        gain = sum(a) - sum(b)
        b01 = sum(1 for x, y in zip(b, a) if x == 0 and y == 1)
        b10 = sum(1 for x, y in zip(b, a) if x == 1 and y == 0)
        print(
            f"{dom:10s} base={sum(b)} attn={sum(a)} gain={gain:+d} "
            f"(fixed={b01}, broke={b10})",
            flush=True,
        )


if __name__ == "__main__":
    main()
