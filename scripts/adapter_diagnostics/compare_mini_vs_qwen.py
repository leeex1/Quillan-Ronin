#!/usr/bin/env python3
"""Directly compare Quillan mini's stored weights against stock Qwen 0.5B.

If the mini is a faithful conversion, embedding rows and layer weights should
have high cosine similarity to Qwen. If they are ~orthogonal/random, the mini
was never a real conversion -- it is a Qwen-shaped random/scratch model, which
fully explains 45% vs stock Qwen's 72%.
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
from wsp_math_eval import load_model, tok_for, letter_ids

torch.set_num_threads(1)
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()

    # embedding comparison (tied): mini E vs qwen embed
    Em = m.embed_tokens.unpack_float().float()
    Eq = q.get_input_embeddings().weight.detach().float()
    print(f"embed shapes mini={tuple(Em.shape)} qwen={tuple(Eq.shape)}", flush=True)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    # compare full-table global cosine and a sample of rows
    print(f"embedding global cos = {cos(Em, Eq):.4f}", flush=True)
    idxs = lids + [0, 1, 100, 1000, 5000, 20000, 100000]
    print("per-row cos (mini vs qwen):", flush=True)
    for i in idxs:
        d = tok.decode([i]).encode("ascii", "replace").decode("ascii")
        print(
            f"  tok {i} ({d!r}): {cos(Em[i], Eq[i]):+.4f} "
            f"| mini_norm={Em[i].norm().item():.3f} qwen_norm={Eq[i].norm().item():.3f}",
            flush=True,
        )

    # layer weight comparison: find mapping (mini 24 layers vs qwen 24)
    n_q = len(q.model.layers)
    n_m = len(m.layers)
    print(f"layers mini={n_m} qwen={n_q}", flush=True)
    for li in [0, 6, 11, 22, 23]:
        if li >= n_q:
            continue
        for sub in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            mm = getattr(m.layers[li], sub, None)
            qq = getattr(
                q.model.layers[li].self_attn
                if sub in ("q_proj", "k_proj", "v_proj", "o_proj")
                else q.model.layers[li].mlp,
                sub,
                None,
            )
            if mm is None or qq is None or not hasattr(mm, "unpack_effective_weight"):
                continue
            Wm = mm.unpack_effective_weight().float()
            Wq = qq.weight.detach().float()
            if Wm.shape != Wq.shape:
                print(
                    f"  L{li}.{sub} SHAPE MISMATCH {tuple(Wm.shape)} vs {tuple(Wq.shape)}",
                    flush=True,
                )
            else:
                print(f"  L{li}.{sub} cos={cos(Wm, Wq):+.4f}", flush=True)


if __name__ == "__main__":
    main()
