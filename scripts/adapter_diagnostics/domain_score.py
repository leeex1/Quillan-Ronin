#!/usr/bin/env python3
"""Domain-scoped score for one checkpoint. Reports per-domain accuracy + CE."""

from __future__ import annotations
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from domain_banks import BANKS, prompts

# thread count owned by top-level script (Predator audit 2026-11-24);
# no set_num_threads here to avoid corrupting A/B regimes.


def score_all(m):
    """Return {domain: accuracy} for an already-loaded model."""
    enable_cache(m)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    out = {}
    for name, (Q, K) in BANKS.items():
        ok = 0
        with torch.inference_mode():
            for seq, gold in zip(prompts(tok, name), K):
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                ok += int(int(lg.argmax()) == gold)
        out[name] = ok
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    a = ap.parse_args()
    m = load_model(Path(a.ck))
    enable_cache(m)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    for name, (Q, K) in BANKS.items():
        ok, ces = 0, []
        with torch.inference_mode():
            for seq, gold in zip(prompts(tok, name), K):
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                ok += int(int(lg.argmax()) == gold)
                ces.append(-F.log_softmax(lg, -1)[gold].item())
        ce = sum(ces) / len(ces)
        print(
            f"[{Path(a.ck).name}] {name:10s} {ok}/{len(K)} CE={ce:.3f} PPL={math.exp(ce):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
