#!/usr/bin/env python3
"""SHAM for the attention glassbox edit: random residual, matched rank+norm,
on the SAME attention modules, same seed. If sham reproduces the +9, the
attention edit is generic perturbation not donor-precision recovery."""

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

torch.manual_seed(20261124)
torch.set_num_threads(1)
BASE = ROOT / "exports" / "quillan-mini-glassbox-base.safetensors"
OUT = ROOT / "exports" / "quillan-mini-glassbox-attn-sham.safetensors"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def main():
    base = load_model(BASE)
    cfg = QuillanSovereignConfig(**{**asdict(base.cfg), "eggroll_rank": 64})
    m = QuillanSovereignBitNetLM(cfg).eval()
    bsd, msd = base.state_dict(), m.state_dict()
    m.load_state_dict(
        {
            k: v
            for k, v in bsd.items()
            if k in msd and ".lora_" not in k and msd[k].shape == v.shape
        },
        strict=False,
    )
    n = 0
    for li, layer in enumerate(m.layers):
        for sub in ATTN:
            mm = getattr(layer, sub, None)
            if mm is None or not hasattr(mm, "unpack_effective_weight"):
                continue
            Wq = mm.unpack_effective_weight().float()
            out, inp = Wq.shape
            r = min(64, mm.lora_A.shape[0], out, inp)
            # random residual with same total norm as a realistic quant residual (~8% of W)
            target = 0.08 * Wq.norm().item() / max(r, 1) ** 0.5
            A = torch.randn(r, inp) * target / (inp**0.5)
            B = torch.randn(out, r) * target / (r**0.5)
            with torch.no_grad():
                mm.lora_A.copy_(A.to(mm.lora_A.dtype))
                mm.lora_B.copy_(B.to(mm.lora_B.dtype))
                mm._wc = None
            n += 1
    m.save_packed_safetensors(OUT)
    print(f"SAVED sham attention {OUT} modules={n}", flush=True)


if __name__ == "__main__":
    main()
