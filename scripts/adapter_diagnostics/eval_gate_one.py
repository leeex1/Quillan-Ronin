#!/usr/bin/env python3
"""Score ONE checkpoint at ONE gate value on ONE bank. Short single-shot job."""

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
from wsp_math_tune_v3 import enable_cache

torch.set_num_threads(2)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--bank", default="fresh40")
    ap.add_argument("--gate", type=float, default=0.0)
    a = ap.parse_args()
    m = load_model(Path(a.ck))
    enable_cache(m)  # cache unpacked 2-bit weights (required for reasonable speed)
    with torch.no_grad():
        for li in (11, 22):
            if m.layers[li].council_moe is not None:
                m.layers[li].council_moe.gate_alpha.fill_(a.gate)
    if a.bank == "fresh40":
        from heldout_fresh40 import prompts, KEY
    elif a.bank == "fresh40b":
        from heldout_fresh40b import prompts, KEY
    else:
        from heldout_fresh40c import prompts, KEY
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    ok, ces, dist = 0, [], [0, 0, 0, 0]
    with torch.inference_mode():
        for seq, gold in zip(prompts(tok), KEY):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            pi = int(lg.argmax())
            dist[pi] += 1
            ok += int(pi == gold)
            ces.append(-F.log_softmax(lg, -1)[gold].item())
    ce = sum(ces) / len(ces)
    print(f"[gate={a.gate}] {a.bank}: {ok}/40 CE={ce:.3f} dist={dist}", flush=True)


if __name__ == "__main__":
    main()
