#!/usr/bin/env python3
"""Energy audit: compare TRUE SVD residual norm vs SHAM residual norm on the
attention modules. Answers Predator's 'fair energy match' attack with numbers."""

from __future__ import annotations
import sys, math
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

torch.set_num_threads(1)
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def main():
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    m = load_model(ROOT / "exports" / "quillan-mini-glassbox-base.safetensors")
    tot_true = tot_w = tot_sham = 0.0
    n = 0
    for li, layer in enumerate(m.layers):
        ql = q.model.layers[li]
        for sub in ATTN:
            mm = getattr(layer, sub, None)
            qq = getattr(ql.self_attn, sub, None)
            if mm is None or qq is None:
                continue
            We = qq.weight.detach().float()
            Wq = mm.unpack_effective_weight().float()
            if tuple(We.shape) != tuple(Wq.shape):
                continue
            R = We - Wq
            tot_true += R.norm().item() ** 2
            tot_w += Wq.norm().item() ** 2
            # sham formula from glassbox_sham_attn.py
            out, inp = Wq.shape
            r = min(64, inp, out)
            target = 0.08 * Wq.norm().item() / math.sqrt(r)
            A = torch.randn(r, inp) * target / math.sqrt(inp)
            B = torch.randn(out, r) * target / math.sqrt(r)
            tot_sham += (B @ A).norm().item() ** 2
            n += 1
    print(f"modules={n}", flush=True)
    print(
        f"TRUE residual total norm={math.sqrt(tot_true):.3f}  relative={math.sqrt(tot_true / tot_w):.4f}",
        flush=True,
    )
    print(
        f"SHAM residual total norm={math.sqrt(tot_sham):.3f}  relative={math.sqrt(tot_sham / tot_w):.4f}",
        flush=True,
    )
    print(f"sham/true energy ratio={math.sqrt(tot_sham / tot_true):.3f}", flush=True)


if __name__ == "__main__":
    main()
