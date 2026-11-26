#!/usr/bin/env python3
"""Nullion's decisive sham control: replace the true donor residual with a
spectrum-matched RANDOM residual (permuted donor residual, same singular values,
random directions). If sham repair reproduces the CE/accuracy gain, the effect
is generic regularizing perturbation, not knowledge restoration.
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

torch.set_num_threads(1)
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
G = torch.Generator().manual_seed(20261124)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = load_model(BASE)
    cfg = QuillanSovereignConfig(**{**asdict(base.cfg), "eggroll_rank": a.rank})
    m = QuillanSovereignBitNetLM(cfg).eval()
    bsd, msd = base.state_dict(), m.state_dict()
    m.load_state_dict(
        {k: v for k, v in bsd.items() if k in msd and msd[k].shape == v.shape},
        strict=False,
    )
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    n = 0
    for li in range(len(m.layers)):
        ql = q.model.layers[li]
        for sub in ATTN + ("gate_proj", "up_proj", "down_proj"):
            mm = getattr(m.layers[li], sub, None)
            parent = ql.self_attn if sub in ATTN else ql.mlp
            qq = getattr(parent, sub, None)
            if mm is None or qq is None or not hasattr(mm, "unpack_effective_weight"):
                continue
            We = qq.weight.detach().float()
            Wq = mm.unpack_effective_weight().float()
            if tuple(We.shape) != tuple(Wq.shape):
                continue
            R = We - Wq
            r = min(a.rank, min(R.shape))
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            # SHAM: keep singular values, randomize the bases (same energy/spectrum)
            Ur = torch.linalg.qr(torch.randn(U[:, :r].shape, generator=G)).Q
            Vr = torch.linalg.qr(torch.randn(Vh[:r].T.shape, generator=G)).Q.T
            A = Vr.contiguous()
            B = (Ur * S[:r]).contiguous()
            with torch.no_grad():
                mm.lora_A.copy_(A.to(mm.lora_A.dtype))
                mm.lora_B.copy_(B.to(mm.lora_B.dtype))
            n += 1
    print(f"sham residual repair on {n} modules rank={a.rank}", flush=True)
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
