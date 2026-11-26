#!/usr/bin/env python3
"""What tokens does the model ACTUALLY put mass on at the Answer: position?

Full-vocab CE ~22 means the gold letter token has p ~ 3e-10. This probe shows
top-10 tokens and where A/B/C/D rank in the full vocabulary distribution.
That is the real object to repair, not gamma scales.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts, KEY, RAW

torch.set_num_threads(1)


def main():
    ck = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
    m = load_model(ck)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    ids = prompts(tok)

    print(
        "id->token map:", {t: tok.decode([i]) for i, t in zip(lids, "ABCD")}, flush=True
    )
    for qi in (0, 1, 2, 5, 16):
        seq = ids[qi]
        with torch.inference_mode():
            lg = m(torch.tensor([seq]))[0][0, -1, :].float()
        top = torch.topk(lg, 10)
        items = [
            (tok.decode([int(i)]), round(float(v), 3))
            for v, i in zip(top.values, top.indices)
        ]
        ranks = {}
        for L, tid in zip("ABCD", lids):
            rank = int((lg > lg[tid]).sum()) + 1
            ranks[L] = (rank, round(float(lg[tid]), 3))
        print(f"Q{qi + 1} gold={'ABCD'[KEY[qi]]}: top={items}", flush=True)
        print(f"      letter logits/ranks={ranks}", flush=True)


if __name__ == "__main__":
    main()
