#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FULL DONOR ATTENTION IMPORT — READ-ONLY, IN-MEMORY, REAL FORWARD.

User decision #1122: full-donor fusion = our own weights. Load REAL trained
Qwen2.5-0.5B attention q/k/v/o (single layer, shape-exact) into the custom
graph's diffusion_core.couil_attn, clear the quant cache, PE rescale, then
run the REAL model forward and compute gate A/C + semantic decode.

This is the first test that uses the actual model(x), not frozen caches.
No files written, no WSP, no backprop.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
QWEN = Path(r"E:\source_models\qwen_0.5b\model.safetensors")
PE_RESCALE = 0.01924
PROSE_A, PROSE_B = "The capital of France is", "Hello, my name is"
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
]


def load_qwen_attn(layer: int) -> Dict[str, torch.Tensor]:
    out = {}
    with safe_open(str(QWEN), framework="pt", device="cpu") as f:
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            k = f"model.layers.{layer}.self_attn.{proj}.weight"
            out[proj] = f.get_tensor(k).float()
    return out


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoTokenizer

    tok, _ = plq.load_tokenizer()
    print("=" * 78)
    print("FULL DONOR ATTENTION IMPORT — REAL MODEL FORWARD (read-only)")
    print("=" * 78)

    # baseline model
    plq.CKPT = CKPT
    model, _ = plq.build_real_model()
    for name, param in model.named_parameters():
        if name.endswith("pos_embed"):
            with torch.no_grad():
                param.data.mul_(PE_RESCALE)
            break
    attn = model.diffusion_core.couil_attn
    print(
        f"attn q {tuple(attn.q_proj.weight.shape)} k {tuple(attn.k_proj.weight.shape)} "
        f"v {tuple(attn.v_proj.weight.shape)} o {tuple(attn.o_proj.weight.shape)}"
    )

    donor_acts = torch.load(
        str(DIAG / "_acts" / "donor_acts.pt"), map_location="cpu", weights_only=False
    )

    def last_logits(p: str) -> torch.Tensor:
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        with torch.no_grad():
            out = model(torch.tensor([ids], dtype=torch.long))
        lg = out[0] if isinstance(out, tuple) else out
        return lg[0, -1, :].float()

    def gate_report(label: str):
        la, lb = last_logits(PROSE_A), last_logits(PROSE_B)
        cos = float(F.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0)).item())
        t1a, t1b = int(la.argmax()), int(lb.argmax())
        a_pass = cos < 0.99 and t1a != t1b
        c1 = c5 = 0
        for p in PROMPTS:
            cl = last_logits(p)
            dl = donor_acts[p]["logits"][0, -1, :].float()
            c1 += int(
                bool(
                    set(torch.topk(cl, 1).indices.tolist())
                    & set(torch.topk(dl, 1).indices.tolist())
                )
            )
            c5 += len(
                set(torch.topk(cl, 5).indices.tolist())
                & set(torch.topk(dl, 5).indices.tolist())
            )
        print(
            f"\n[{label}] A: cos={cos:.4f} top1={tok.decode([t1a])!r} vs {tok.decode([t1b])!r} "
            f"differ={t1a != t1b} -> {'PASS' if a_pass else 'FAIL'}"
        )
        print(
            f"[{label}] C: mean top1={c1 / len(PROMPTS):.4f} top5={c5 / len(PROMPTS):.4f}"
        )
        # decode
        for p in [
            PROSE_A,
            PROSE_B,
            "The quick brown fox jumps over the lazy dog and then",
        ]:
            cl = last_logits(p)
            tops = [tok.decode([i]) for i in torch.topk(cl, 5).indices.tolist()]
            print(f"    {p[:40]!r:42} -> {tops}")

    gate_report("BASELINE (sign-vote QKV)")

    # import real Qwen attention per layer, test each
    for layer in [0, 11, 23]:
        qw = load_qwen_attn(layer)
        with torch.no_grad():
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                mod = getattr(attn, proj)
                target = mod.weight
                src = qw[proj]
                if tuple(src.shape) != tuple(target.shape):
                    print(
                        f"layer {layer} {proj} shape mismatch {tuple(src.shape)} vs {tuple(target.shape)} — skip"
                    )
                    continue
                target.data.copy_(src.to(target.dtype))
                mod._w_quant_cache = None  # clear stale quantization cache
        gate_report(f"QWEN LAYER {layer} ATTENTION")

    print("\n" + "=" * 78)
    print("DONE — no files written, WSP untouched")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
