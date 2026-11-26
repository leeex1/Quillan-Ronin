#!/usr/bin/env python3
"""Closed-form repair of quantization loss via SVD of the conversion residual.

The 3-plane 2-bit packing loses ~14% relative Frobenius per matrix. The exact
donor weights are available (stock Qwen). For each packed projection:

    R = W_exact - W_quantized

R is a known, fixed numerical error matrix -- NOT a training gradient. Compute
a truncated SVD R ~= U_r S_r V_r^T and store it in the layer's existing
lora_B (out,r) / lora_A (r,in) parameters, which the forward pass already adds:

    W_effective = W_quantized + lora_B @ lora_A ~= W_exact

No backprop, no optimizer, no gradient, no answer labels. Pure algebra on the
difference between the donor weights and their quantized copy.
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
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = load_model(BASE)
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    layers = (
        list(range(24)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    )

    tot_before = tot_after = 0.0
    n = 0
    for li in layers:
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
            if mm.lora_A is None or mm.lora_B is None:
                continue
            R = We - Wq
            # the deployed lora rank is fixed at 16; never exceed it
            r = min(a.rank, mm.lora_A.shape[0], min(R.shape))
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            A = Vh[:r].contiguous()  # (r, in)
            B = (U[:, :r] * S[:r]).contiguous()  # (out, r)
            with torch.no_grad():
                mm.lora_A.copy_(A.to(mm.lora_A.dtype))
                mm.lora_B.copy_(B.to(mm.lora_B.dtype))
                mm._wc = None
            before = R.norm().item()
            after = (R - B @ A).norm().item()
            tot_before += before
            tot_after += after
            n += 1
    print(
        f"modules={n} rank={a.rank} residual_norm {tot_before:.2f} -> {tot_after:.2f} "
        f"({100 * (1 - tot_after / max(tot_before, 1e-9)):.1f}% repaired)",
        flush=True,
    )
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
