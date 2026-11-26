#!/usr/bin/env python3
"""Build a clean rank-64 base: high-rank architecture, faithful packed donor
weights, ZERO adapters. This is the starting substrate for surgical section
edits (identity / attention / math_ffn / council) one at a time.
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import asdict
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import QuillanSovereignConfig, QuillanSovereignBitNetLM
from wsp_math_eval import load_model

torch.set_num_threads(2)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    cfg = QuillanSovereignConfig(**{**asdict(base.cfg), "eggroll_rank": a.rank})
    m = QuillanSovereignBitNetLM(cfg).eval()
    bsd, msd = base.state_dict(), m.state_dict()
    # copy every non-lora tensor (packed weights, gamma, norms, embeddings)
    copy = {
        k: v
        for k, v in bsd.items()
        if k in msd and ".lora_" not in k and msd[k].shape == v.shape
    }
    m.load_state_dict(copy, strict=False)
    # lora stays zero (fresh init) = no repair applied yet
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    nz = sum(
        1 for k, v in m.state_dict().items() if ".lora_B" in k and v.abs().max() > 0
    )
    print(f"SAVED {out} copied={len(copy)} lora_nonzero={nz} rank={a.rank}", flush=True)


if __name__ == "__main__":
    main()
