#!/usr/bin/env python3
"""Save per-item correctness bits for one model to a .pt file."""

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
from heldout_fresh40 import prompts as p1, KEY as K1
from heldout_fresh40b import prompts as p2, KEY as K2
from heldout_fresh40c import prompts as p3, KEY as K3

torch.set_num_threads(1)
BANKS = [p1, p2, p3]
KEYS = [K1, K2, K3]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = load_model(Path(a.ck))
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    bits = []
    with torch.inference_mode():
        for pf, KEY in zip(BANKS, KEYS):
            for seq, gold in zip(pf(tok), KEY):
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                bits.append(int(int(lg.argmax()) == gold))
    t = torch.tensor(bits, dtype=torch.int8)
    torch.save(t, a.out)
    print(f"saved {a.out} correct={int(t.sum())}/{len(t)}", flush=True)


if __name__ == "__main__":
    main()
