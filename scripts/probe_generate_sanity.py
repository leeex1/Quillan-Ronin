#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FORGE v2 — isolate whether degenerate output is the checkpoint or generate().

NEW FILE. Loads the real trained mini checkpoint and, for a few trivial prompts,
compares (a) raw direct-forward top-5 next tokens with (b) generate() output.
Both paths use torch.no_grad(). Read-only diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402

from probe_longctx_quality import build_real_model, load_tokenizer  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "2 + 2 =",
    "The magic word is zyzzyva. Remember it. Question: The magic word is",
]


def main() -> None:
    model, meta = build_real_model()
    tok, _ = load_tokenizer()
    print(f"loaded params={meta['param_count']:,}")
    with torch.no_grad():
        emb = model.ingestion.txt_emb.weight.float()
        pe = model.ingestion.pos_embed.float()
        print(
            f"EMBED_NORMS txt_emb_mean_abs={emb.abs().mean().item():.5f} "
            f"pos_embed_mean_abs={pe.abs().mean().item():.5f} "
            f"pos_embed_max_abs={pe.abs().max().item():.4f}"
        )
        del emb, pe
    first_last = None
    for text in PROMPTS:
        ids = [int(x) for x in tok.encode(text, add_special_tokens=False)]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        last = logits[0, -1, :].float()
        if first_last is None:
            first_last = last.clone()
        else:
            diff = (last - first_last).abs()
            print(
                f"  logit_delta_vs_prompt1 max={diff.max().item():.4f} "
                f"mean={diff.mean().item():.4f} "
                f"pct_identical={float((diff < 1e-4).float().mean()) * 100:.2f}%"
            )
            del diff
        top = torch.topk(last, 5)
        raw_top = [
            (tok.decode([int(i)], skip_special_tokens=True), round(float(p), 4))
            for i, p in zip(top.indices.tolist(), top.values.tolist())
        ]
        with torch.no_grad():
            cont = model.generate(ids, max_tokens=8, temp=0.0, repetition_penalty=1.0)
        gen = tok.decode([int(t) for t in cont], skip_special_tokens=True)
        print(f"\nPROMPT: {text!r}")
        print(f"  raw_forward_top5={raw_top}")
        print(f"  generate_ids={[int(t) for t in cont]}")
        print(f"  generate_text={gen!r}")
        del ids, x, out, logits, last, top, cont
    import gc

    gc.collect()

    # Controlled in-memory experiment: match pos_embed scale to txt_emb scale.
    # No files are edited; this only tests the embedding-scale hypothesis.
    with torch.no_grad():
        model.ingestion.pos_embed.mul_(0.02)
    print("\n=== AFTER in-memory pos_embed *= 0.02 (scale match test) ===")
    first_last = None
    for text in PROMPTS[:3]:
        ids = [int(x) for x in tok.encode(text, add_special_tokens=False)]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        last = logits[0, -1, :].float()
        top = torch.topk(last, 5)
        raw_top = [
            (tok.decode([int(i)], skip_special_tokens=True), round(float(p), 4))
            for i, p in zip(top.indices.tolist(), top.values.tolist())
        ]
        print(f"  {text!r} -> top5={raw_top}")
        del ids, x, out, logits, last, top
    gc.collect()


if __name__ == "__main__":
    main()
