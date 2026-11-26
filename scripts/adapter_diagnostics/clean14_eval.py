#!/usr/bin/env python3
"""Score the two candidates on the 14 genuinely-clean questions of heldout_final40
(those with no overlap with any prior selection/confirmation bank), per Predator audit."""

from __future__ import annotations
import sys
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
from heldout_final40 import prompts, KEY

torch.set_num_threads(1)
CLEAN = [
    1,
    2,
    3,
    6,
    9,
    14,
    19,
    23,
    26,
    31,
    33,
    34,
    39,
    40,
]  # 1-based, Predator's clean subset
MODELS = {
    "mini": "quillan-mini-hifi-g4-councilfixed.safetensors",
    "flagship": "quillan-flagship-hifi-g4-exact.safetensors",
}


def main():
    for name, f in MODELS.items():
        m = load_model(ROOT / "exports" / f)
        tok = tok_for(m.cfg.vocab_size)
        lids = letter_ids(tok, m.cfg.vocab_size)
        ps = prompts(tok)
        ok = 0
        with torch.inference_mode():
            for i in CLEAN:
                lg = m(torch.tensor([ps[i - 1]]))[0][0, -1, lids].float()
                ok += int(int(lg.argmax()) == KEY[i - 1])
        print(f"{name} CLEAN14 = {ok}/14 ({100 * ok / 14:.1f}%)", flush=True)
        del m


if __name__ == "__main__":
    main()
