#!/usr/bin/env python3
"""Direct ternary-level surgery: damp the late blocks that inject answer bias.

Finding (knowledge_flow_trace, MINI): residual direction at L21 scores 25/40
correct-answer argmax, but L22-L23 rotate it toward the C letter (C-A bias
-0.075 -> +0.279) and accuracy drops to 20/40. The final read head is fixed;
the corruption is in the last two blocks' residual additions.

This scales the stored `gamma` scales of selected projections by a factor,
reducing how far those blocks rotate the residual. gamma is a stored float16
parameter; scaling it is a direct weight edit (no optimizer, no lora, no
gradient, no answer labels). The 2-bit ternary q_packed values are untouched.

Args select layers/components/factor so each variant is a clean single change.
"""

from __future__ import annotations
import sys, argparse
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
from wsp_math_eval import load_model

torch.set_num_threads(1)
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="22,23")
    ap.add_argument("--components", default="down_proj")
    ap.add_argument("--factor", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    layers = [int(x) for x in a.layers.split(",")]
    comps = a.components.split(",")
    m = load_model(BASE)
    edits = []
    for li in layers:
        layer = m.layers[li]
        for comp in comps:
            mod = getattr(layer, comp, None)
            if mod is None or not hasattr(mod, "gamma"):
                print(f"SKIP L{li}.{comp} (no gamma)", flush=True)
                continue
            before = mod.gamma.float().norm().item()
            with torch.no_grad():
                mod.gamma.mul_(a.factor)
            after = mod.gamma.float().norm().item()
            edits.append(
                f"L{li}.{comp} gamma_norm {before:.4f}->{after:.4f} x{a.factor}"
            )

    print(f"factor={a.factor} edits={len(edits)}", flush=True)
    for e in edits:
        print("  " + e, flush=True)
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
