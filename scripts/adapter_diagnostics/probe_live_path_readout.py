#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LIVE-PATH READOUT PROBE (board #986/#988/#996) — READ-ONLY, IN-MEMORY.

Tests the last mechanism with evidence behind it:
  1. PE rescale x0.01924 on ingestion.pos_embed (kills 37x position dominance)
  2. Untie readout: txt_dec.weight <- donor qwen embed_tokens (= tied lm_head)
  3. Zero txt_dec.lora_B (kill rank-256 residual term)
  4. Pooled readout: average final_norm(x_final) over positions instead of last only

All mutations are IN MEMORY on the loaded module. No checkpoint, config, gate,
adapter or source file is written. Donor tensor is read from the frozen cache.

Gate-A style measurement: two prose prompts -> logit cos + top-1.
Diagnostic: does top-1 become a sensible English token at any position/ pooling?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DONOR = ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"
PROSE_A = "The capital of France is"
PROSE_B = "Hello, my name is"
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
]
PE_RESCALE = 0.01924


def capture_hidden(model: Any, ids: List[int]) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: Tuple[Any, ...], out: torch.Tensor) -> None:
        captured["h"] = out.detach().clone()

    h = model.final_norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(torch.tensor([ids], dtype=torch.long))
    finally:
        h.remove()
    return captured["h"][0]  # [T, hidden]


def top_tok(tok: Any, lg: torch.Tensor, k: int = 5) -> str:
    ids = torch.topk(lg, k).indices.tolist()
    return " | ".join(
        f"{tok.decode([int(i)], skip_special_tokens=True)!r}" for i in ids
    )


def measure(model: Any, tok: Any, label: str) -> None:
    print(f"\n=== {label} ===")
    a_ids = [int(x) for x in tok.encode(PROSE_A, add_special_tokens=False)]
    b_ids = [int(x) for x in tok.encode(PROSE_B, add_special_tokens=False)]
    ha = capture_hidden(model, a_ids)  # [Ta, 896]
    hb = capture_hidden(model, b_ids)
    with torch.no_grad():
        dec = model.txt_dec
        # position-wise logits through the (possibly untied) head
        la = dec(ha).float()
        lb = dec(hb).float()
        # pooled readout: mean over all positions
        hpa = ha.mean(dim=0, keepdim=True)
        hpb = hb.mean(dim=0, keepdim=True)
        pla = dec(hpa)[0].float()
        plb = dec(hpb)[0].float()

    # last-token gate-A style
    cos_last = float(
        F.cosine_similarity(la[-1].unsqueeze(0), lb[-1].unsqueeze(0)).item()
    )
    t1a, t1b = int(la[-1].argmax()), int(lb[-1].argmax())
    # pooled gate-A style
    cos_pool = float(F.cosine_similarity(pla.unsqueeze(0), plb.unsqueeze(0)).item())
    p1a, p1b = int(pla.argmax()), int(plb.argmax())
    print(
        f"LAST  : cos={cos_last:.6f} top1_A={tok.decode([t1a])!r} top1_B={tok.decode([t1b])!r} differ={t1a != t1b}"
    )
    print(
        f"POOLED: cos={cos_pool:.6f} top1_A={tok.decode([p1a])!r} top1_B={tok.decode([p1b])!r} differ={p1a != p1b}"
    )
    print(f"  POOLED top5 A: {top_tok(tok, pla)}")
    print(f"  POOLED top5 B: {top_tok(tok, plb)}")

    # all 5 frozen gate prompts: pooled top-1
    print("  all-prompt pooled top1:")
    for p in PROMPTS:
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        h = capture_hidden(model, ids)
        with torch.no_grad():
            pl = dec(h.mean(dim=0, keepdim=True))[0].float()
        print(f"    {p[:38]!r:42} -> {top_tok(tok, pl, 3)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    ckpt = args.checkpoint.resolve()

    import probe_longctx_quality as plq

    tok, _ = plq.load_tokenizer()
    print("=" * 78)
    print("LIVE-PATH READOUT PROBE (all mutations in-memory, read-only)")
    print(f"checkpoint: {ckpt}")
    print(f"donor     : {DONOR}")
    print("=" * 78)

    plq.CKPT = ckpt
    model, meta = plq.build_real_model()
    print(f"model params={meta['param_count']:,}")

    measure(model, tok, "CONDITION 0: RAW (baseline)")

    # 1) PE rescale
    pe = None
    for name, param in model.named_parameters():
        if name.endswith("pos_embed"):
            pe = param
            print(f"PE rescale on {name} shape={tuple(pe.shape)}")
            break
    if pe is not None:
        with torch.no_grad():
            pe.data.mul_(PE_RESCALE)
    measure(model, tok, "CONDITION 1: PE RESCALED")

    # 2) untie to donor head + 3) zero lora_B
    donor = torch.load(str(DONOR), map_location="cpu", weights_only=False)
    Wu = donor["embed_tokens"].float()  # [151936, 896]
    dec = model.txt_dec
    print(f"\ntxt_dec.weight before: {tuple(dec.weight.shape)} {dec.weight.dtype}")
    print(f"donor embed_tokens   : {tuple(Wu.shape)}")
    if tuple(Wu.shape) != tuple(dec.weight.shape):
        print("SHAPE MISMATCH — aborting untie")
        return 2
    with torch.no_grad():
        dec.weight.data.copy_(Wu.to(dec.weight.dtype))
        if hasattr(dec, "lora_B"):
            dec.lora_B.data.zero_()
            print("zeroed txt_dec.lora_B")
    measure(
        model,
        tok,
        "CONDITION 2: PE RESCALED + DONOR HEAD UNTIED + POOLED (head only; pooling in measure)",
    )

    # GATE-C STYLE: pooled custom logits vs frozen donor last-position logits
    donor_acts = DIAG / "_acts" / "donor_acts.pt"
    if donor_acts.exists():
        da = torch.load(str(donor_acts), map_location="cpu", weights_only=False)
        print("\n=== GATE-C STYLE OVERLAP (pooled custom vs donor last-position) ===")
        tot1 = tot5 = 0.0
        for p in PROMPTS:
            ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
            h = capture_hidden(model, ids)
            with torch.no_grad():
                pl = model.txt_dec(h.mean(dim=0, keepdim=True))[0].float()
            dl = da[p]["logits"][0, -1, :].float()
            c1 = set(torch.topk(pl, 1).indices.tolist())
            d1 = set(torch.topk(dl, 1).indices.tolist())
            c5 = set(torch.topk(pl, 5).indices.tolist())
            d5 = set(torch.topk(dl, 5).indices.tolist())
            o1, o5 = len(c1 & d1), len(c5 & d5)
            tot1 += o1
            tot5 += o5
            print(f"  {p[:38]!r:42} top1={o1} top5={o5} custom={top_tok(tok, pl, 3)}")
        print(
            f"  MEAN top1={tot1 / len(PROMPTS):.4f} top5={tot5 / len(PROMPTS):.4f} "
            f"(immutable gate needs both >0)"
        )
    else:
        print(f"\nWARN donor_acts missing: {donor_acts}")

    print("\n" + "=" * 78)
    print("DONE — no files written")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
