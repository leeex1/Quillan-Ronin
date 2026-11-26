#!/usr/bin/env python3
"""High-fidelity reconversion: rebuild the mini from faithful stock-Qwen weights
with group_size=8 (3 planes, config-only change; planes capped at 3 in class).
Keeps rank64 lora slots for later surgical edits. No backprop.
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
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    cfg = QuillanSovereignConfig(
        **{**asdict(base.cfg), "group_size": a.group, "eggroll_rank": 64}
    )
    m = QuillanSovereignBitNetLM(cfg).eval()
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    # embedding: exact qwen (set_weights_from_float handles int8 quantization)
    m.embed_tokens.set_weights_from_float(
        q.get_input_embeddings().weight.detach().float()
    )
    # norms/bias: copy from base (they're already faithful qwen values)
    bsd = base.state_dict()
    for k, v in bsd.items():
        if k in ("embed_tokens.weight_int8", "embed_tokens.weight_scale"):
            continue
        if ".q_packed" in k or ".gamma" in k or ".lora_" in k:
            continue
        if k in m.state_dict() and m.state_dict()[k].shape == v.shape:
            m.state_dict()[k].copy_(v)
    # repack each linear from stock qwen weights
    n = 0
    ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
    for li, layer in enumerate(m.layers):
        ql = q.model.layers[li]
        for sub in ATTN + ("gate_proj", "up_proj", "down_proj"):
            mm = getattr(layer, sub, None)
            parent = ql.self_attn if sub in ATTN else ql.mlp
            qq = getattr(parent, sub, None)
            if mm is None or qq is None or not hasattr(mm, "set_weights_from_float"):
                continue
            W = qq.weight.detach().float()
            if (mm.out_features, mm.in_features) != (W.shape[0], W.shape[1]):
                print(
                    f"shape mismatch L{li}.{sub} {mm.out_features}x{mm.in_features} vs {tuple(W.shape)}",
                    flush=True,
                )
                continue
            mm.set_weights_from_float(W)
            n += 1
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(
        f"SAVED {out} repacked={n} group={a.group} size={out.stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
