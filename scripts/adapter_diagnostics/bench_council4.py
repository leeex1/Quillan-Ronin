#!/usr/bin/env python3
"""Decisive: does the ported 4-donor council fusion help when the gate opens?
Fresh-load both checkpoints; measure GK, fixed-set, verified generation across gates."""

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


def gen(m, tok):
    ids = tok.encode("The meaning of life is", add_special_tokens=False)
    with torch.inference_mode():
        out = m.generate(torch.tensor([ids]), max_new_tokens=30, temperature=0.0)
    g = out[0, len(ids) :].tolist()
    txt = (
        tok.decode(g, skip_special_tokens=True)
        .encode("ascii", "backslashreplace")
        .decode("ascii")
    )
    gr = [tuple(g[i : i + 5]) for i in range(max(0, len(g) - 4))]
    rep = 1 - len(set(gr)) / len(gr) if gr else 0.0
    return len(g), rep, txt[:70]


def run(ckpt, label):
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(ckpt)
    enable_cache(m)
    moes = [l.council_moe for l in m.layers if l.council_moe is not None]
    print(f"\n=== {label} ({ckpt.name}) ===", flush=True)
    for alpha in [0.0, 0.02, 0.05, 0.1]:
        for moe in moes:
            with torch.no_grad():
                moe.gate_alpha.fill_(alpha)
        g = gk(m, tok, lids)
        f = fixed(m, tok, lids)
        n, r, t = gen(m, tok)
        print(
            f"[gate={alpha}] GK={g}/25 fixed={f}/20 gen_ntok={n} rep={r:.2f} text={t!r}",
            flush=True,
        )


run(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors", "ORIGINAL")
run(ROOT / "exports" / "quillan-mini-council4.safetensors", "COUNCIL4-PORTED")
