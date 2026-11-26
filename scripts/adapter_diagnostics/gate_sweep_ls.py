#!/usr/bin/env python3
"""Sweep council gate_alpha on the LS+scale-norm bank checkpoint. Measure
balanced-bank accuracy + CE at each gate. Old orthogonal bank gave garbage;
test whether the improved alignment coexists with GK."""

from __future__ import annotations
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts as p1, KEY as K1
from heldout_fresh40c import prompts as p3, KEY as K3

torch.set_num_threads(1)
CK = ROOT / "exports" / "quillan-mini-council-ls.safetensors"


def score(m, tok, lids, pf, KEY):
    ok, ces, dist = 0, [], [0, 0, 0, 0]
    with torch.inference_mode():
        for seq, gold in zip(pf(tok), KEY):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            pi = int(lg.argmax())
            dist[pi] += 1
            ok += int(pi == gold)
            ces.append(-F.log_softmax(lg, -1)[gold].item())
    return ok, sum(ces) / len(ces), dist


def main():
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(CK)  # single load; mutate gate_alpha in place
    for alpha in [0.0, 0.02, 0.05, 0.1, 0.2]:
        with torch.no_grad():
            for li in (11, 22):
                m.layers[li].council_moe.gate_alpha.fill_(alpha)
        a1, ce1, d1 = score(m, tok, lids, p1, K1)
        a3, ce3, d3 = score(m, tok, lids, p3, K3)
        print(
            f"[gate={alpha}] fresh40={a1}/40 CE={ce1:.3f} dist={d1} | "
            f"fresh40c={a3}/40 CE={ce3:.3f} dist={d3}",
            flush=True,
        )


if __name__ == "__main__":
    main()
