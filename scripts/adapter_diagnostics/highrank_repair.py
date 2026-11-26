#!/usr/bin/env python3
"""Higher-rank closed-form repair of quantization residual.

Rank-16 repaired only 4.6% of the residual (error is high-rank). Build a model
with eggroll_rank=64, copy the base packed weights, then SVD-repair each matrix
with a rank-64 correction from (W_exact - W_quantized). Still canonical BitNet
ternary + EGGROLL lora; no backprop, no optimizer, no labels.
"""

from __future__ import annotations
import sys, argparse, json
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


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = load_model(BASE)
    cfg = QuillanSovereignConfig(**{**asdict(base.cfg), "eggroll_rank": a.rank})
    m = QuillanSovereignBitNetLM(cfg).eval()
    # copy all base state that exists in new model (lora shapes differ -> skip those)
    bsd = base.state_dict()
    msd = m.state_dict()
    copy = {k: v for k, v in bsd.items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(copy, strict=False)
    print(
        f"copied {len(copy)}/{len(msd)} tensors (lora shapes differ by design)",
        flush=True,
    )

    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    tot_b = tot_a = 0.0
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
            A = Vh[:r].contiguous()
            B = (U[:, :r] * S[:r]).contiguous()
            with torch.no_grad():
                mm.lora_A.copy_(A.to(mm.lora_A.dtype))
                mm.lora_B.copy_(B.to(mm.lora_B.dtype))
                mm._wc = None
            tot_b += R.norm().item()
            tot_a += (R - B @ A).norm().item()
            n += 1
    print(
        f"modules={n} rank={a.rank} residual {tot_b:.2f}->{tot_a:.2f} "
        f"({100 * (1 - tot_a / max(tot_b, 1e-9)):.1f}% repaired)",
        flush=True,
    )
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
