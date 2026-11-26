#!/usr/bin/env python3
"""Direct weight surgery on the STORED ternary-model parameters.

NO optimizer, NO gradient, NO gold-answer direction, NO lora adapter writes.

Targets: outlier group scales (`gamma`) found by weight_pathology_scan.py —
a single group whose float16 scale is >5x its sibling median, which amplifies
one ternary pattern across the residual stream. Clip to a chosen multiple of
the per-tensor median. This edits q_packed/gamma storage directly; the 2-bit
ternary values are untouched.

Variants are saved as separate checkpoints so each can be scored with
eval_metrics.py on the untouched FRESH40 bank.
"""

from __future__ import annotations
import sys, argparse, copy
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
    ap.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="clip gamma entries above clip x per-tensor median",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="skip tensors with more than --max_entries outliers",
    )
    ap.add_argument("--max-entries", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = load_model(BASE)
    edits = []
    for li, layer in enumerate(m.layers):
        for sub in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            mod = getattr(layer, sub, None)
            if mod is None or not hasattr(mod, "gamma"):
                continue
            g = mod.gamma.float()
            med = g.median().clamp_min(1e-12)
            cap = a.clip * med
            mask = g > cap
            n = int(mask.sum())
            if a.strict and n > a.max_entries:
                # strict mode: only touch tensors with a handful of true outliers
                continue
            if n:
                before = g[mask].max().item()
                with torch.no_grad():
                    mod.gamma.copy_(g.clamp(max=cap).to(torch.float16))
                edits.append(
                    f"L{li}.{sub}: {n} entries max {before:.4g}->{cap.item():.4g}"
                )

    print(f"clip={a.clip}x edits={len(edits)}", flush=True)
    for e in edits:
        print("  " + e, flush=True)
    if not edits:
        print("  (no outliers above threshold)", flush=True)

    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out} size={out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
