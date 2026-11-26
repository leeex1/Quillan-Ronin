#!/usr/bin/env python3
"""Surgical rank-1 weight edit: project the late-layer injected bias direction
OUT of the L22/L23 down_proj weights, preserving everything orthogonal.

Diagnosis (knowledge_flow_trace, no labels): L20 residual scores 41/80, final
scores 35/80; C-A letter bias jumps -0.075 -> +0.279 across L22-23. The last
two blocks inject a harmful residual direction.

Damping the whole matrix rotated the bias C->B (bias swap, falsified). This is
the correct surgical version:
  b = (mean h_L23 - mean h_L21) normalized   [measured from PROMPTS only]
  W' = W - b (b^T W)                          [project rows orthogonal to b]
Applied to unpacked down_proj, then repacked via set_weights_from_float.
No optimizer, no gradient, no lora, no answer labels.
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
from heldout_fresh40 import prompts

torch.set_num_threads(1)
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="22,23")
    ap.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="1.0 = remove full projected component",
    )
    a = ap.parse_args()

    m = load_model(BASE)
    tok = tok_for(m.cfg.vocab_size)
    ids = prompts(tok)

    # measure injected direction from prompts only (no answer keys)
    cap = {}
    hooks = []
    for li in (21, 23):

        def mk(idx):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cap[idx] = h.detach().float()[:, -1, :].clone()

            return hook

        hooks.append(m.layers[li].register_forward_hook(mk(li)))
    with torch.inference_mode():
        for seq in ids:
            m(torch.tensor([seq]))
    for hk in hooks:
        hk.remove()
    b = cap[23].mean(0) - cap[21].mean(0)
    b = b / b.norm().clamp_min(1e-12)
    print(
        f"injected direction b norm={b.norm().item():.4f} "
        f"top-5 dims={torch.topk(b.abs(), 5).indices.tolist()}",
        flush=True,
    )

    for li in [int(x) for x in a.layers.split(",")]:
        dp = m.layers[li].down_proj
        W = dp.unpack_effective_weight()  # (out=896, in=4864) fp32
        # residual direction b lives in output/row space (896). Remove each
        # row's component along b: W' = W - strength * b (b^T W)
        W2 = W - a.strength * torch.outer(b, b @ W)
        print(
            f"L{li}.down_proj delta_rel_frob="
            f"{(W2 - W).norm().item() / W.norm().item():.5f}",
            flush=True,
        )
        dp.set_weights_from_float(W2)

    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
