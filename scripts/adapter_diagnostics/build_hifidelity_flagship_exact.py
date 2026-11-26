#!/usr/bin/env python3
"""Faithful flagship reconversion from Llama-3.2-1B with matched FFN width.

Root cause of prior failure: flagship FFN=5120 but Llama FFN=8192; 8192 is not
a multiple of 5120 so pool_ffn TRUNCATED (kept 5120 of 8192), destroying
gate/up/down alignment -> predictions collapsed to 100% A.

Fix (owner authorized FFN as technical detail): set intermediate_size=8192 to
match the donor exactly. Then every matrix is a direct faithful repack, no
pooling, exactly like the successful mini group4 conversion. This also serves
as the non-Qwen donor replication: if group4 helps a Llama-based model, the
packing-fidelity mechanism is general.
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
LLAMA = r"E:\source_models\llama_1b"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = load_model(
        ROOT / "exports" / "quillan-flagship-sovereign-bitnet.safetensors"
    )
    cfg = QuillanSovereignConfig(
        **{
            **asdict(base.cfg),
            "group_size": a.group,
            "eggroll_rank": 64,
            "intermediate_size": 8192,
        }
    )
    m = QuillanSovereignBitNetLM(cfg).eval()
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        LLAMA, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    m.embed_tokens.set_weights_from_float(
        q.get_input_embeddings().weight.detach().float()
    )
    # copy non-packed tensors with matching shape (norms, biases, router, gate)
    bsd = base.state_dict()
    msd = m.state_dict()
    cp = {}
    for k, v in bsd.items():
        if k.startswith("embed_tokens."):
            continue
        if ".q_packed" in k or ".gamma" in k or "_q" in k or ".lora_" in k:
            continue
        if k in msd and msd[k].shape == v.shape:
            cp[k] = v
    m.load_state_dict(cp, strict=False)
    ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
    FFN = ("gate_proj", "up_proj", "down_proj")
    n = 0
    skip = 0
    for li, layer in enumerate(m.layers):
        ql = q.model.layers[li]
        for sub in ATTN + FFN:
            mm = getattr(layer, sub, None)
            parent = ql.self_attn if sub in ATTN else ql.mlp
            qq = getattr(parent, sub, None)
            if mm is None or qq is None or not hasattr(mm, "set_weights_from_float"):
                continue
            W = qq.weight.detach().float()
            if (mm.out_features, mm.in_features) != (W.shape[0], W.shape[1]):
                skip += 1
                print(
                    f"skip L{li}.{sub} {mm.out_features}x{mm.in_features} vs {tuple(W.shape)}",
                    flush=True,
                )
                continue
            mm.set_weights_from_float(W)
            n += 1
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(
        f"SAVED {out} repacked={n} skipped={skip} group={a.group} ffn=8192 size={out.stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
