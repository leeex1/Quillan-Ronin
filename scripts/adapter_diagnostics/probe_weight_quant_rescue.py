#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decisive control: lowering _weight_quant eps should restore prompt sensitivity."""

from __future__ import annotations
import sys, gc
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
import torch
import torch.nn.functional as F
from probe_longctx_quality import build_real_model, load_tokenizer
import quillan_v8_saturated as arch

model, meta = build_real_model()
tok, _ = load_tokenizer()
A = [int(x) for x in tok.encode("The capital of France is", add_special_tokens=False)]
B = [int(x) for x in tok.encode("Hello, my name is", add_special_tokens=False)]


def last_logits(ids):
    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long))
    lg = out[0] if isinstance(out, tuple) else out
    return lg[0, -1, :].clone()


def diff(x, y):
    d = (x - y).abs()
    return f"max={d.max().item():.6e} mean={d.mean().item():.6e} pct<1e-4={(d < 1e-4).float().mean().item() * 100:.2f}% cos={F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item():.6f}"


la, lb = last_logits(A), last_logits(B)
print("BASELINE (eps=0.01):", diff(la, lb))
print("  top1 A/B:", int(la.argmax()), int(lb.argmax()))

# Rescue: set every BitLinear eps to 1e-5 so scale is NOT clamped away.
n = 0
for m in model.modules():
    if isinstance(m, arch.BitLinear):
        m.eps = 1e-5
        m._w_quant_cache = None
        n += 1
print(f"Set eps=1e-5 on {n} BitLinear modules; cleared quant caches.")

wq = arch._weight_quant(model.diffusion_core.couil_attn.q_proj.weight, 1e-5)
print(
    f"q_proj _weight_quant(eps=1e-5): mean_abs={wq.abs().mean().item():.6e} all_zero={bool((wq == 0).all())}"
)

la2, lb2 = last_logits(A), last_logits(B)
print("RESCUE  (eps=1e-5):", diff(la2, lb2))
print("  top1 A/B:", int(la2.argmax()), int(lb2.argmax()))
for tag, lg in (("A", la2), ("B", lb2)):
    v, i = torch.topk(lg, 5)
    print(
        f"  top5({tag}):",
        [
            (tok.decode([int(t)], skip_special_tokens=True), round(float(p), 2))
            for t, p in zip(i.tolist(), v.tolist())
        ],
    )
