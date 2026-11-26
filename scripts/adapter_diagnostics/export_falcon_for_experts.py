#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-export the Falcon3-1B GGUF extract into the layout build_donor_experts.py expects:
  E:\source_models\falcon3_1b\model.safetensors   (HF-style layer weights)
  E:\source_models\falcon3_1b\tokenizer.json       (token strings from GGUF)

Reads the already-extracted falcon3_1b_donors.pt (full 165-tensor GGUF dump) and
the GGUF blob for the tokenizer vocabulary. No model training, no backprop.
Writes NEW files only; never touches canonical/gates/existing donors.
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file

GGUF = Path(
    r"C:\Users\Admin\.ollama_models_c\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"
)
PT = Path(r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_donors.pt")
OUT = Path(r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf")
OUT.mkdir(parents=True, exist_ok=True)

# Map GGUF tensor names -> HF-style names the builder reads:
#   token_embd.weight        -> model.embed_tokens.weight
#   blk.N.ffn_down.weight    -> model.layers.N.mlp.down_proj.weight
#   blk.N.ffn_gate.weight    -> model.layers.N.mlp.gate_proj.weight
#   blk.N.ffn_up.weight      -> model.layers.N.mlp.up_proj.weight
#   blk.N.attn_q.weight      -> model.layers.N.self_attn.q_proj.weight
#   blk.N.attn_k.weight      -> model.layers.N.self_attn.k_proj.weight
#   blk.N.attn_v.weight      -> model.layers.N.self_attn.v_proj.weight
#   blk.N.attn_output.weight -> model.layers.N.self_attn.o_proj.weight
#   blk.N.attn_norm.weight   -> model.layers.N.input_layernorm.weight
#   blk.N.ffn_norm.weight    -> model.layers.N.post_attention_layernorm.weight
#   output_norm.weight       -> model.norm.weight
NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}


def hf_name(k: str):
    if k in NAME_MAP:
        return NAME_MAP[k]
    import re

    m = re.match(r"blk\.(\d+)\.(.+)\.weight$", k)
    if not m:
        return None
    n, part = m.group(1), m.group(2)
    m2 = {
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_output": "self_attn.o_proj",
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
    }
    if part not in m2:
        return None
    return f"model.layers.{n}.{m2[part]}.weight"


def main():
    print("Loading Falcon donor pt...", flush=True)
    d = torch.load(PT, map_location="cpu", weights_only=True)
    out = {}
    for k, v in d.items():
        if not hasattr(v, "shape"):
            continue
        hn = hf_name(k)
        if not hn:
            continue
        # Falcon3 GGUF pads dims to multiples of 128: embed 2176 (=2048+128),
        # ffn_down 8704, ffn_gate/up 8192x2176. Crop trailing pad to true dims
        # so the builder's Procrustes gets [V,2048] as its spec declares.
        if hn == "model.embed_tokens.weight" and v.shape[1] > 2048:
            v = v[:, :2048]
        if "mlp.down_proj" in hn and v.shape[1] > 8192:
            v = v[:, :8192]  # down_proj [d, ffn]
        if ("mlp.gate_proj" in hn or "mlp.up_proj" in hn) and v.shape[0] > 8192:
            v = v[:8192, :]  # gate/up [ffn, d]
        if (
            "mlp.gate_proj" in hn or "mlp.up_proj" in hn or "self_attn" in hn
        ) and v.shape[1] > 2048:
            v = v[:, :2048]  # input dim d
        out[hn] = v.contiguous()
    print(f"mapped {len(out)} tensors -> safetensors", flush=True)
    # drop lm_head (tied) to save space; builder only needs embed + layers + norm
    out.pop("lm_head.weight", None)
    save_file(out, str(OUT / "model.safetensors"), metadata={"format": "pt"})
    print(
        f"wrote {OUT / 'model.safetensors'} ({(OUT / 'model.safetensors').stat().st_size / 1e6:.0f} MB)",
        flush=True,
    )

    # Tokenizer: Ollama GGUF stores weights only (no tokenizer.ggml.tokens), so
    # we use the official Falcon3 tokenizer already downloaded to OUT/tokenizer.json.
    tok_path = OUT / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit(
            f"FATAL: {tok_path} missing — download official Falcon3 tokenizer first"
        )
    print(
        f"using official Falcon3 tokenizer: {tok_path} ({tok_path.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )

    print(
        "[done] Falcon donor now in builder-expected layout (true dims + official tokenizer).",
        flush=True,
    )


if __name__ == "__main__":
    main()
