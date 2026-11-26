#!/usr/bin/env python3
"""Causal trace: WHICH LAYER injects the A/B/C/D prior skew.

No gold answers used anywhere. Only the uniform-prior assumption: for a
multiple-choice prompt with balanced answer positions, A/B/C/D should be
roughly equiprobable a priori. We measure the letter logit distribution at
the output of each block (h @ E_letter.T) across the FRESH40 prompts and
report the per-layer skew (C minus A logit gap, and argmax distribution).

The layer where the skew first appears is the surgical target.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from heldout_fresh40 import prompts, RAW

torch.set_num_threads(1)


def main():
    ck = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
    m = load_model(ck)
    enable_cache(m)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    E = m.embed_tokens.unpack_float()[lids].float()  # 4 x hidden
    ids = prompts(tok)

    caps = {}
    hooks = []
    for li, layer in enumerate(m.layers):

        def mk(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                caps[idx] = h.detach().float()[:, -1, :].clone()

            return hook

        hooks.append(layer.register_forward_hook(mk(li)))

    with torch.inference_mode():
        for seq in ids:
            m(torch.tensor([seq]))

    # final norm output
    hfinal = m.norm(caps[len(m.layers) - 1].unsqueeze(0)).float()
    caps["final"] = hfinal.squeeze(0)

    print(
        f"layer | mean logits A/B/C/D        | C-A gap | argmax dist [A,B,C,D]",
        flush=True,
    )
    for key in list(range(len(m.layers))) + ["final"]:
        h = caps[key]
        lg = h @ E.T  # N x 4
        mean = lg.mean(0)
        gap = (mean[2] - mean[0]).item()
        am = lg.argmax(-1)
        dist = [int((am == j).sum()) for j in range(4)]
        print(
            f"{str(key):>5} | "
            + " ".join(f"{v:7.3f}" for v in mean.tolist())
            + f" | {gap:+7.3f} | {dist}",
            flush=True,
        )

    for hk in hooks:
        hk.remove()


if __name__ == "__main__":
    main()
