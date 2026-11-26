#!/usr/bin/env python3
"""Verify rank-64 rebuild is output-neutral (same logits as rank-16 baseline)."""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math40 import PROBLEMS, KEY
from hard_math20 import LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model

tok = tok_for(151936)
lids = letter_ids(tok, 151936)


def fmt(i):
    q, o = PROBLEMS[i][0], PROBLEMS[i][1]
    return (
        f"Question {i + 1}: {q}\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, o))
        + "Answer:"
    )


xs = [torch.tensor([tok.encode(fmt(i), add_special_tokens=False)]) for i in range(40)]
logs = []
for name in [
    "quillan-mini-dense-council.safetensors",
    "quillan-mini-rank64.safetensors",
]:
    m = load_model(ROOT / "exports" / name)
    out = []
    with torch.inference_mode():
        for x in xs:
            out.append(m(x)[0][0, -1, lids].clone())
    logs.append(torch.stack(out))
    ok = sum(int(out[i].argmax() == KEY[i]) for i in range(40))
    print(f"{name}: {ok}/40")
drift = float((logs[0] - logs[1]).abs().max())
print(
    f"max logit drift rank16 vs rank64 = {drift:.8f} -> {'OUTPUT-NEUTRAL OK' if drift < 1e-3 else 'CHANGED'}"
)
