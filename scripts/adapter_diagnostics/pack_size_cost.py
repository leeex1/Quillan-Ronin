#!/usr/bin/env python3
"""Storage cost of higher-fidelity packing configs for the mini model.
Packed ternary bits depend on planes (uint8 4-per-byte); gamma = planes*out*(in/group) fp16.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model

torch = None


def main():
    m = load_model(ROOT / "exports" / "quillan-mini-glassbox-base.safetensors")
    # enumerate packed linear layers: (out, in)
    layers = []
    for mod in m.modules():
        if (
            hasattr(mod, "q_packed")
            and hasattr(mod, "gamma")
            and hasattr(mod, "in_features")
        ):
            layers.append((mod.out_features, mod.in_features))
    # embedding is int8 separately
    emb = m.embed_tokens.weight_int8.numel() + m.embed_tokens.weight_scale.numel() * 2
    print(f"packed linears={len(layers)} embed_bytes={emb / 1e6:.1f}MB", flush=True)
    for group, planes in [(32, 3), (16, 6), (8, 3), (8, 4), (8, 6)]:
        packed = gamma = 0
        for out, inp in layers:
            packed += planes * ((out * inp + 3) // 4)  # uint8 bytes
            ng = inp // group if inp % group == 0 else 1
            gamma += planes * out * ng * 2  # fp16
        total = (packed + gamma + emb) / 1e9
        print(
            f"group={group:3d} planes={planes}: q_packed={packed / 1e6:.0f}MB gamma={gamma / 1e6:.0f}MB total~{total:.2f}GB",
            flush=True,
        )


if __name__ == "__main__":
    main()
