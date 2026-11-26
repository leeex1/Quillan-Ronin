#!/usr/bin/env python3
"""Verify left-pad batched logits at -1 match single-prompt logits exactly."""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from hard_math40 import PROBLEMS, KEY
from hard_math20 import LETTERS

tok = tok_for(151936)
lids = letter_ids(tok, 151936)
m = load_model(ROOT / "exports" / "quillan-mini-dense-council.safetensors")
seqs = [
    tok.encode(
        f"Question {i + 1}: {PROBLEMS[i][0]}\n"
        + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, PROBLEMS[i][1]))
        + "Answer:",
        add_special_tokens=False,
    )
    for i in range(40)
]
pad = tok.pad_token_id
L = max(len(s) for s in seqs)
xb = torch.full((40, L), pad, dtype=torch.long)
for i, s in enumerate(seqs):
    xb[i, L - len(s) :] = torch.tensor(s)
with torch.inference_mode():
    lb = m(xb)[0][:, -1, lids]
maxdrift = 0
for i in range(40):
    with torch.inference_mode():
        ls = m(torch.tensor([seqs[i]]))[0][0, -1, lids]
    maxdrift = max(maxdrift, float((ls - lb[i]).abs().max()))
print(f"max |single - leftpad_batch[-1]| over 40 = {maxdrift:.8f}")
print("VERDICT:", "MATCH" if maxdrift < 1e-4 else "MISMATCH - batch invalid")
