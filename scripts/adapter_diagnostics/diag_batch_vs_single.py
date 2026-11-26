#!/usr/bin/env python3
"""Compare batched-padded scoring vs single-prompt scoring on the SAME checkpoint.

If these disagree, the batched evaluator used for WSP tuning is invalid and all
reported fixed-set scores must be recomputed single-prompt.
"""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from hard_math20 import KEY, format_problem, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache, batch_inputs

ck = ROOT / "exports" / "quillan-mini-math20-wsp-v8.safetensors"
tok = tok_for(151936)
lids = letter_ids(tok, 151936)
m = load_model(ck)
enable_cache(m)

# single
single = []
for i in range(20):
    ids = tok.encode(format_problem(i), add_special_tokens=False)
    with torch.inference_mode():
        lg = m(torch.tensor([ids]))[0][0, -1, lids]
    single.append(max(range(4), key=lambda j: float(lg[j])))
sok = sum(int(single[i] == KEY[i]) for i in range(20))

# batched
x, lens = batch_inputs(tok, lids)
print("pad id:", tok.pad_token_id, "lids:", lids, "maxlen:", x.shape[1])
with torch.inference_mode():
    lg = m(x)[0]
batch = []
for i, ln in enumerate(lens):
    v = [float(lg[i, ln - 1, lids[j]]) for j in range(4)]
    batch.append(max(range(4), key=lambda j: v[j]))
bok = sum(int(batch[i] == KEY[i]) for i in range(20))

# batched reading LAST position (L-1) instead of ln-1
last = [max(range(4), key=lambda j: float(lg[i, -1, lids[j]])) for i in range(20)]
lok = sum(int(last[i] == KEY[i]) for i in range(20))

print(f"single={sok}/20 {single}")
print(f"batched(ln-1)={bok}/20 {batch}")
print(f"batched(last)={lok}/20 {last}")
print(
    "per-problem single vs batch:",
    [(i + 1, single[i], batch[i]) for i in range(20) if single[i] != batch[i]],
)
