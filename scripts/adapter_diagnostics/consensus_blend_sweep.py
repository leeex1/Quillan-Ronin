#!/usr/bin/env python3
"""Test user's mechanism faithfully: SCALE-NORMALIZED token-consensus blended
into the host embedding at controlled alpha. Prior fusion blended RAW donor
deltas (scales wildly different: BitNet ~7964 vs Qwen ~37) and failed. This
normalizes each donor to the host scale first (the 0.556/0.665->0.555 step),
then merges. Sweep alpha to find whether any normalized blend helps.
"""

from __future__ import annotations
import sys
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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", required=True)
    ap.add_argument("--alphas", default="0.02,0.05,0.1,0.2")
    ap.add_argument("--out-prefix", default="quillan-mini-nblend")
    a = ap.parse_args()

    # We build each alpha variant from BASE + (consensus - qwen_orig)*alpha
    base = load_model(BASE)
    qwen_E = base.embed_tokens.unpack_float().float().clone()
    cons = load_model(Path(a.consensus)).embed_tokens.unpack_float().float()
    delta = cons - qwen_E
    print(
        f"delta norm={delta.norm().item():.3f} qwen norm={qwen_E.norm().item():.3f} "
        f"rel={delta.norm().item() / qwen_E.norm().item():.4f}",
        flush=True,
    )
    for alpha in [float(x) for x in a.alphas.split(",")]:
        E = qwen_E + alpha * delta
        base.embed_tokens.set_weights_from_float(E)
        out = (
            ROOT
            / "exports"
            / f"{a.out_prefix}{str(alpha).replace('.', '')}.safetensors"
        )
        base.save_packed_safetensors(out)
        print(f"saved alpha={alpha} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
