#!/usr/bin/env python3
"""Closed-form two-donor merge for flagship: Llama (host) + Falcon3-1B.

Both are 2048-hidden / 8192-FFN. Falcon 18 layers -> select 16 to match host.
Vocab differs (128256 vs 131072) -> map embedding by token string where shared,
keep host for the rest. Matrices merged by aligned average in the HOST frame:
  W_merged = (1-a)*W_llama + a*W_falcon
where Falcon and Llama are already in comparable 2048/8192 geometry. Alpha is
a tunable closed-form scalar (no gradient). If a>0 improves held-out over
a=0 (single Llama), the "merge similar, keep unique" thesis has support.

Memory-efficient: stream one layer at a time from safetensors.
"""

from __future__ import annotations
import sys, re
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
from safetensors import safe_open

torch.set_num_threads(2)
LLAMA = r"E:\source_models\llama_1b"
FALCON = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model_bf16.safetensors"
HOST = (
    ROOT / "exports" / "quillan-flagship-hifi-g4-exact.safetensors"
)  # faithful Llama g4
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    host = load_model(HOST)
    # select 16 falcon layers evenly spaced from 18
    falcon_layers = [round(i * 17 / 15) for i in range(16)]  # 0..17 -> 16 layers
    print(f"falcon layer map: {falcon_layers}", flush=True)
    with safe_open(FALCON, framework="pt", device="cpu") as ff:
        # merge each matrix: host packed currently holds llama; unpack, blend with falcon, repack
        n = 0
        for li, layer in enumerate(host.layers):
            fl = falcon_layers[li]
            for sub in ATTN + FFN:
                mm = getattr(layer, sub, None)
                if mm is None or not hasattr(mm, "unpack_effective_weight"):
                    continue
                W_host = mm.unpack_effective_weight().float()
                fk = (
                    f"model.layers.{fl}."
                    + ("self_attn." if sub in ATTN else "mlp.")
                    + sub
                    + ".weight"
                )
                if fk not in ff.keys():
                    continue
                W_fal = ff.get_tensor(fk).float()
                if tuple(W_fal.shape) != tuple(W_host.shape):
                    # GQA mismatch: Falcon k/v output is 1024, host is 512.
                    # Mean-pool head axis 1024 -> 512 (closed-form), no skip.
                    if (
                        sub in ("k_proj", "v_proj")
                        and W_fal.shape[0] == 1024
                        and W_host.shape[0] == 512
                        and W_fal.shape[1] == W_host.shape[1]
                    ):
                        W_fal = W_fal.view(512, 2, W_fal.shape[1]).mean(dim=1)
                    else:
                        print(
                            f"skip L{li}.{sub} shape {tuple(W_host.shape)} vs {tuple(W_fal.shape)}",
                            flush=True,
                        )
                        continue
                W = a.alpha * W_fal + (1 - a.alpha) * W_host
                mm.set_weights_from_float(W)
                n += 1
            if li % 4 == 0:
                print(f"  merged layer {li}", flush=True)
    out = ROOT / "exports" / a.out
    host.save_packed_safetensors(out)
    print(
        f"SAVED {out} merged={n} alpha={a.alpha} size={out.stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
