#!/usr/bin/env python3
"""Which layers' quantized weights cause the most damage?

Progressively replace exact-Qwen weights into the Quillan model, one layer at
a time (all 7 projections), and score. A monotone climb toward ~29/40 confirms
quantization error accumulation. Identifies which layers matter most.
Diagnostic only.
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


def score(m, tok, lids):
    ok, ces = 0, []
    with torch.inference_mode():
        for seq, gold in zip(p1(tok), K1):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            ok += int(int(lg.argmax()) == gold)
            ces.append(-F.log_softmax(lg, -1)[gold].item())
    return ok, sum(ces) / len(ces)


def swap_layer(m, q, li):
    ql = q.model.layers[li]
    n = 0
    for sub in ATTN + ("gate_proj", "up_proj", "down_proj"):
        mm = getattr(m.layers[li], sub, None)
        parent = ql.self_attn if sub in ATTN else ql.mlp
        qq = getattr(parent, sub, None)
        if mm is None or qq is None or not hasattr(mm, "unpack_effective_weight"):
            continue
        Wq = qq.weight.detach().float().clone()
        mm.unpack_effective_weight = lambda W=Wq: W
        n += 1
    return n


def main():
    m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    Eq = q.get_input_embeddings().weight.detach().float().clone()
    m.embed_tokens.unpack_float = lambda: Eq  # exact embedding throughout

    a, ce = score(m, tok, lids)
    print(f"exact embedding only: {a}/40 CE={ce:.4f}", flush=True)
    # swap layers in order, front to back
    for li in range(24):
        swap_layer(m, q, li)
        if li in (0, 3, 7, 11, 15, 19, 21, 22, 23):
            a, ce = score(m, tok, lids)
            print(f"  +exact layers 0..{li}: {a}/40 CE={ce:.4f}", flush=True)


if __name__ == "__main__":
    main()
