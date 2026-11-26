#!/usr/bin/env python3
"""ISOLATION: exact stock-Qwen weights + Quillan forward pass.

Monkeypatch each PackedBitLinear.unpack_effective_weight to return the exact
Qwen matrix for that layer/projection. Activation quant left ON (as deployed).
- If score jumps to ~72% => quantized WEIGHTS are the culprit.
- If score stays ~45% => the FORWARD PASS (SwiGLU/RoPE/RMSNorm/attn) is buggy.
Diagnostic only, no file changes.
"""

from __future__ import annotations
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts as p1, KEY as K1

torch.set_num_threads(1)
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def score(m, tok, lids, pf, KEY, name):
    ok, ces = 0, []
    with torch.inference_mode():
        for seq, gold in zip(pf(tok), KEY):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            pi = int(lg.argmax())
            ok += int(pi == gold)
            ces.append(-F.log_softmax(lg, -1)[gold].item())
    ce = sum(ces) / len(ces)
    print(f"  {name}: {ok}/{len(KEY)} CE={ce:.4f} PPL={math.exp(ce):.3f}", flush=True)


def main():
    m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)

    print("=== BASELINE (quantized weights, Quillan forward) ===", flush=True)
    score(m, tok, lids, p1, K1, "fresh40")

    # install exact-weight patches
    patched = 0
    for li, layer in enumerate(m.layers):
        ql = q.model.layers[li]
        for sub in ATTN + ("gate_proj", "up_proj", "down_proj"):
            mm = getattr(layer, sub, None)
            parent = ql.self_attn if sub in ATTN else ql.mlp
            qq = getattr(parent, sub, None)
            if mm is None or qq is None or not hasattr(mm, "unpack_effective_weight"):
                continue
            Wq = qq.weight.detach().float().clone()
            if tuple(Wq.shape) != (mm.out_features, mm.in_features):
                print(
                    f"  SHAPE MISMATCH L{li}.{sub}: {tuple(Wq.shape)} vs "
                    f"{(mm.out_features, mm.in_features)}",
                    flush=True,
                )
                continue
            mm.unpack_effective_weight = lambda W=Wq: W
            patched += 1
    # exact embedding read head
    Eq = q.get_input_embeddings().weight.detach().float().clone()
    m.embed_tokens.unpack_float = lambda: Eq
    print(f"  patched {patched} projections + embedding", flush=True)

    print(
        "=== ISOLATION (exact Qwen weights, Quillan forward, act-quant ON) ===",
        flush=True,
    )
    score(m, tok, lids, p1, K1, "fresh40")


if __name__ == "__main__":
    main()
