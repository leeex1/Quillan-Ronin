#!/usr/bin/env python3
"""How many ternary planes / what config reconstructs Qwen weights faithfully?

Round-trip exact Qwen matrices through PackedBitLinear at planes=1,2,3,4,6
(with and without RHT) and report relative Frobenius error. Determines whether
a higher-plane repack can recover the precision lost in the deployed 3-plane
model -- a closed-form weight repair.
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
from quillan_sovereign_bitnet import PackedBitLinear

torch.set_num_threads(1)
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def rel_err(a, b):
    return (a - b).norm().item() / a.norm().clamp_min(1e-12).item()


def main():
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    W = q.model.layers[0].self_attn.q_proj.weight.detach().float()  # 896x896
    print(f"matrix shape={tuple(W.shape)} norm={W.norm().item():.3f}", flush=True)
    for rht in (True, False):
        for planes in (1, 2, 3, 4, 6, 8):
            m = PackedBitLinear(
                W.shape[1],
                W.shape[0],
                group_size=32,
                num_planes=planes,
                eggroll_rank=0,
                use_rht=rht,
            )
            m.set_weights_from_float(W)
            W2 = m.unpack_effective_weight().float()
            print(
                f"  rht={rht!s:5s} planes={planes}: rel_err={rel_err(W, W2):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
