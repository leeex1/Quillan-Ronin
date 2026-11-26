#!/usr/bin/env python3
"""Artifact verification: load rank96 checkpoint, check param count, and prove
the 481/821 copied tensors are exactly the non-lora parameters (i.e. no packed
weight/gamma/norm was left uninitialized)."""

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
from wsp_math_eval import load_model

torch.set_num_threads(1)


def main():
    base = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    hi = load_model(ROOT / "exports" / "quillan-mini-hirank96.safetensors")
    bsd, hsd = base.state_dict(), hi.state_dict()
    lora = [k for k in hsd if ".lora_A" in k or ".lora_B" in k]
    nonlora = [k for k in hsd if k not in lora]
    print(
        f"hi state tensors: total={len(hsd)} lora={len(lora)} nonlora={len(nonlora)}",
        flush=True,
    )
    # verify every non-lora tensor matches base exactly
    mism = []
    for k in nonlora:
        if k not in bsd or bsd[k].shape != hsd[k].shape:
            mism.append(f"{k}: missing/shape")
        elif not torch.equal(bsd[k], hsd[k]):
            d = (bsd[k].float() - hsd[k].float()).abs().max().item()
            mism.append(f"{k}: maxdiff={d}")
    print(f"non-lora mismatches vs base: {len(mism)}", flush=True)
    for m in mism[:10]:
        print("  " + m, flush=True)

    # param count (logical = sum of all stored float elements for lora + packed weights)
    total = sum(v.numel() for v in hsd.values())
    lora_n = sum(v.numel() for k, v in hsd.items() if ".lora_A" in k or ".lora_B" in k)
    print(
        f"stored elements total={total:,} lora={lora_n:,} non-lora={total - lora_n:,}",
        flush=True,
    )
    print(f"cfg eggroll_rank={hi.cfg.eggroll_rank} layers={len(hi.layers)}", flush=True)
    # forward smoke test
    ids = torch.randint(0, 1000, (1, 16))
    with torch.inference_mode():
        out = hi(ids)
    print(
        f"forward smoke OK logits shape={tuple(out[0].shape) if isinstance(out, tuple) else tuple(out.shape)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
