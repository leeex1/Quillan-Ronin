#!/usr/bin/env python3
"""Fresh-reload 40-problem check for Dragon (Falcon tokenizer)."""

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
from hard_math20 import LETTERS
from hard_math40 import PROBLEMS as P40, KEY as K40
from wsp_math_eval import load_model
import v23_common as vc


def fmt(i):
    q, o = P40[i][0], P40[i][1]
    return (
        f"Question {i + 1}: {q}\n"
        + "".join(f"{L}) {v}\n" for L, v in zip(LETTERS, o))
        + "Answer:"
    )


ck = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else ROOT / "exports" / "quillan-dragon-math40-headrow.safetensors"
)
tok = vc.load_dragon_tokenizer()
lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
m = load_model(ck)
ok = 0
miss = []
for i in range(40):
    ids = tok.encode(fmt(i), add_special_tokens=False)
    with torch.inference_mode():
        lg = m(torch.tensor([ids]))[0][0, -1, lids]
    p = int(lg.argmax())
    ok += int(p == K40[i])
    if p != K40[i]:
        miss.append(i + 1)
print(f"DRAGON FRESH {ok}/40 misses={miss}", flush=True)
