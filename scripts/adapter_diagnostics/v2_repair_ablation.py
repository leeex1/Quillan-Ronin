#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 REPAIR ablations -- in-memory only, no writes, no source edits.

Loads the real checkpoint ONCE, then for each A/B/C combination restores the
original tensors/eps and measures last-position prompt sensitivity + donor
top-k overlap. Answers: which defect is actually load-bearing, and does
reviving the (structurally mismatched) attention towers re-kill the prompt?
"""

from __future__ import annotations

import gc
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import quillan_v8_saturated as arch  # noqa: E402
from probe_longctx_quality import build_real_model, load_tokenizer  # noqa: E402

POS_SCALE = 0.01924392
EPS_FIXED = 1e-12
EPS_ORIG = 0.01
PROMPTS = ("The capital of France is", "Hello, my name is")


def last_logits(model, ids):
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    lg = out[0] if isinstance(out, tuple) else out
    return lg[0, -1, :].float().clone()


def main() -> None:
    t0 = time.perf_counter()
    tok, _ = load_tokenizer()
    model, meta = build_real_model()
    bits = [m for m in model.modules() if isinstance(m, arch.BitLinear)]
    pe0 = model.ingestion.pos_embed.detach().clone()
    nw0 = model.ingestion.norm.weight.detach().clone()
    donor = torch.load(
        str(DIAG / "_acts" / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    A = [int(x) for x in tok.encode(PROMPTS[0], add_special_tokens=False)]
    B = [int(x) for x in tok.encode(PROMPTS[1], add_special_tokens=False)]
    dl = donor[PROMPTS[0]]["logits"][0, -1, :].float()

    print(
        f"MODEL params={meta['param_count']:,} bitlinear_modules={len(bits)}",
        flush=True,
    )
    print(
        f"{'variant':<10} {'maxdiff':>11} {'meandiff':>11} {'cos':>9} "
        f"{'top1A':>7} {'top1B':>7} {'dtop1':>6} {'dtop5':>6} {'qout':>11}",
        flush=True,
    )

    for flags in (
        (),
        ("A",),
        ("B",),
        ("C",),
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("A", "B", "C"),
    ):
        # restore pristine state
        with torch.no_grad():
            model.ingestion.pos_embed.copy_(pe0)
            model.ingestion.norm.weight.copy_(nw0)
        for m in bits:
            m.eps = EPS_ORIG
            m._w_quant_cache = None
        # apply selected fixes
        if "B" in flags:
            with torch.no_grad():
                model.ingestion.pos_embed.mul_(POS_SCALE)
        if "C" in flags:
            with torch.no_grad():
                model.ingestion.norm.weight.copy_(model.final_norm.weight)
        if "A" in flags:
            for m in bits:
                m.eps = EPS_FIXED
                m._w_quant_cache = None
        la, lb = last_logits(model, A), last_logits(model, B)
        d = (la - lb).abs()
        cos = F.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0)).item()
        c1 = set(torch.topk(la, 1).indices.tolist())
        d1 = set(torch.topk(dl, 1).indices.tolist())
        c5 = set(torch.topk(la, 5).indices.tolist())
        d5 = set(torch.topk(dl, 5).indices.tolist())
        with torch.no_grad():
            qo = (
                model.diffusion_core.couil_attn.q_proj(
                    torch.randn(
                        4, model.diffusion_core.couil_attn.q_proj.weight.shape[1]
                    )
                )
                .float()
                .abs()
                .mean()
                .item()
            )
        print(
            f"{'+'.join(flags) if flags else 'NONE':<10} "
            f"{d.max().item():>11.4e} {d.mean().item():>11.4e} {cos:>9.5f} "
            f"{int(la.argmax()):>7} {int(lb.argmax()):>7} "
            f"{len(c1 & d1):>6} {len(c5 & d5):>6} {qo:>11.4e}",
            flush=True,
        )
        del la, lb, d, c1, d1, c5, d5
        gc.collect()

    print(f"elapsed={time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
