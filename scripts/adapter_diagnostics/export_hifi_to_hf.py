#!/usr/bin/env python3
"""Export a QuillanSovereignBitNetLM checkpoint (packed 2-bit) to a standard
HF llama checkpoint (fp16) so llama.cpp convert_hf_to_gguf.py can build a GGUF.
No training. Pure unpack: W = sum_p (ternary_p * gamma_p) + lora_B@lora_A.
"""

from __future__ import annotations
import sys, json, shutil
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

torch.set_num_threads(2)
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", default=r"E:\source_models\llama_1b")
    ap.add_argument("--model-type", default="llama")
    a = ap.parse_args()
    m = load_model(ROOT / "exports" / a.ck)
    c = m.cfg
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {}
    # embeddings (tied)
    Ew = m.embed_tokens.unpack_float().half()
    sd["model.embed_tokens.weight"] = Ew
    # each layer
    for li, layer in enumerate(m.layers):
        pre = f"model.layers.{li}."
        sd[pre + "input_layernorm.weight"] = (
            layer.input_layernorm.weight.detach().half()
        )
        sd[pre + "post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm.weight.detach().half()
        )
        for sub in ATTN + FFN:
            mod = getattr(layer, sub, None)
            if mod is None:
                continue
            W = mod.unpack_effective_weight().float()
            # add lora correction if present
            if (
                mod.lora_A is not None
                and mod.lora_B is not None
                and mod.lora_B.abs().max() > 0
            ):
                W = W + mod.lora_B.float() @ mod.lora_A.float()
            key = pre + ("self_attn." if sub in ATTN else "mlp.") + sub + ".weight"
            sd[key] = W.half()
            if mod.bias is not None:
                sd[pre + ("self_attn." if sub in ATTN else "mlp.") + sub + ".bias"] = (
                    mod.bias.detach().half()
                )
    sd["model.norm.weight"] = m.norm.weight.detach().half()
    # tied lm_head
    sd["lm_head.weight"] = Ew.clone()
    from safetensors.torch import save_file

    save_file(sd, str(out / "model.safetensors"))
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": a.model_type,
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "num_hidden_layers": len(m.layers),
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "vocab_size": c.vocab_size,
        "max_position_embeddings": c.max_position_embeddings,
        "rope_theta": c.rope_theta,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "bos_token_id": 128000 if c.vocab_size == 128256 else 151643,
        "eos_token_id": 128001 if c.vocab_size == 128256 else 151645,
        "torch_dtype": "float16",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    # tokenizer
    tp = Path(a.tok)
    for f in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src = tp / f
        if src.exists():
            shutil.copy(src, out / f)
    print(
        f"EXPORTED {out} tensors={len(sd)} size={(out / 'model.safetensors').stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
