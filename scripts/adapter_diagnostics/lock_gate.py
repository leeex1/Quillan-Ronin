#!/usr/bin/env python3
"""Lock the carved circuit at the coexistence gate (alpha=0.1) and verify fresh reload."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from hard_math20 import PROBLEMS as HP, KEY as HK, format_problem as hfmt
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from ptp_adder import DIG, SUMS

SRC = Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-circuit.safetensors")
OUT = Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-gate010.safetensors")


def gk(m, tok, lids):
    ok = 0
    for i, (q, opts, _) in enumerate(GQ):
        p = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:"
        )
        ids = tok.encode(p, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids]
        ok += int(int(lg.argmax()) == GK[i])
    return ok


def fixed(m, tok, lids):
    ok = 0
    for i in range(len(HP)):
        ids = tok.encode(hfmt(i), add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids]
        ok += int(int(lg.argmax()) == HK[i])
    return ok


def circuit(m):
    moe = m.layers[8].council_moe
    H = moe.hidden_size

    def out(a, b):
        x = torch.zeros(1, 1, H)
        x[0, 0, a] = 1.0
        x[0, 0, DIG + b] = 1.0
        with torch.inference_mode():
            y = moe(x)
        return int(y[0, 0, :SUMS].argmax())

    tr = sum(out(i, j) == i + j for i in range(6) for j in range(6))
    te = sum(
        out(i, j) == i + j for i in range(DIG) for j in range(DIG) if i >= 6 or j >= 6
    )
    return tr, te


tok = tok_for(151936)
lids = letter_ids(tok, 151936)
m = load_model(SRC)
enable_cache(m)
with torch.no_grad():
    m.layers[8].council_moe.gate_alpha.fill_(0.1)
print(
    f"[in-mem gate=0.1] GK={gk(m, tok, lids)}/25 fixed={fixed(m, tok, lids)}/20 circuit={circuit(m)}",
    flush=True,
)
m.save_packed_safetensors(OUT)
print(f"[saved] {OUT}", flush=True)
m2 = load_model(OUT)
enable_cache(m2)
print(
    f"[FRESH RELOAD] gate_alpha={float(m2.layers[8].council_moe.gate_alpha.detach()):.3f} GK={gk(m2, tok, lids)}/25 fixed={fixed(m2, tok, lids)}/20 circuit={circuit(m2)}",
    flush=True,
)
