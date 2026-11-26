#!/usr/bin/env python3
"""High-fidelity reconversion of FLAGSHIP from stock Llama-3.2-1B.

Flagship: hidden 2048, 16 layers, vocab 128256, FFN 5120 (rank16, group32).
Llama:    hidden 2048, 16 layers, vocab 128256, FFN 8192.
Only FFN width differs -> deterministic mean-pool 8192 -> 5120 for the three
FFN matrices (gate/up/down). This IS the non-Qwen donor replication Predator
demanded: if group4 fidelity improves a Llama-based model too, the packing
mechanism is general, not Qwen self-similarity.

No backprop, no optimizer. group_size configurable (32 control vs 4/2 hifi).
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


def pool_ffn(W, target, axis):
    """Mean-pool a matrix's FFN axis from W.shape[axis] down to target."""
    cur = W.shape[axis]
    if cur == target:
        return W
    if cur % target != 0:
        # truncate to nearest multiple
        keep = (cur // target) * target
        W = W.narrow(axis, 0, keep)
        cur = keep
    k = cur // target
    sh = list(W.shape)
    sh[axis] = target
    sh.insert(axis + 1, k)
    return W.reshape(sh).mean(dim=axis + 1)


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
        **{**asdict(base.cfg), "group_size": a.group, "eggroll_rank": 64}
    )
    m = QuillanSovereignBitNetLM(cfg).eval()
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        LLAMA, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    # embedding from llama
    m.embed_tokens.set_weights_from_float(
        q.get_input_embeddings().weight.detach().float()
    )
    # copy non-packed tensors with matching shape
    bsd = base.state_dict()
    msd = m.state_dict()
    cp = {}
    for k, v in bsd.items():
        if (
            k.startswith("embed_tokens.")
            or ".q_packed" in k
            or "_q" in k
            and "lora" not in k
            or ".gamma" in k
            or ".lora_" in k
        ):
            continue
        if k in msd and msd[k].shape == v.shape:
            cp[k] = v
    m.load_state_dict(cp, strict=False)
    ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
    FFN = ("gate_proj", "up_proj", "down_proj")
    n = 0
    pooled = 0
    for li, layer in enumerate(m.layers):
        ql = q.model.layers[li]
        for sub in ATTN + FFN:
            mm = getattr(layer, sub, None)
            parent = ql.self_attn if sub in ATTN else ql.mlp
            qq = getattr(parent, sub, None)
            if mm is None or qq is None or not hasattr(mm, "set_weights_from_float"):
                continue
            W = qq.weight.detach().float()
            if sub in ("gate_proj", "up_proj"):
                # out dim is the FFN width (8192 -> 5120)
                W = pool_ffn(W, mm.out_features, 0)
                pooled += 1
            elif sub == "down_proj":
                # in dim is the FFN width (8192 -> 5120)
                W = pool_ffn(W, mm.in_features, 1)
                pooled += 1
            if (mm.out_features, mm.in_features) != (W.shape[0], W.shape[1]):
                print(
                    f"skip L{li}.{sub} shape {mm.out_features}x{mm.in_features} vs {tuple(W.shape)}",
                    flush=True,
                )
                continue
            mm.set_weights_from_float(W)
            n += 1
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(
        f"SAVED {out} repacked={n} pooled_ffn={pooled} group={a.group} size={out.stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
