#!/usr/bin/env python3
"""Staged diagnostic: pinpoint where the council-ls gate test fails/stalls."""

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
print("[1] imports path set", flush=True)
from wsp_math_eval import tok_for, letter_ids, load_model

print("[2] wsp_math_eval imported", flush=True)
from heldout_fresh40 import prompts as p1, KEY as K1

print("[3] heldout imported", flush=True)

torch.set_num_threads(2)
CK = ROOT / "exports" / "quillan-mini-council-ls.safetensors"
print(f"[4] loading {CK.name} exists={CK.exists()}", flush=True)
m = load_model(CK)
print("[5] model loaded", flush=True)
tok = tok_for(m.cfg.vocab_size)
lids = letter_ids(tok, m.cfg.vocab_size)
print("[6] tokenizer ready", flush=True)


def score(pf, KEY):
    ok = 0
    with torch.inference_mode():
        for seq, gold in zip(pf(tok), KEY):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            ok += int(int(lg.argmax()) == gold)
    return ok


print(f"[7] gate=0 score fresh40={score(p1, K1)}/40", flush=True)
with torch.no_grad():
    for li in (11, 22):
        m.layers[li].council_moe.gate_alpha.fill_(0.05)
print("[8] gate set 0.05", flush=True)
print(f"[9] gate=0.05 score fresh40={score(p1, K1)}/40", flush=True)
print("DONE", flush=True)
