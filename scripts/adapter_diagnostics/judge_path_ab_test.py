#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B test to isolate WHY the independent judge scores 0/10 while build-time
generation is coherent. NO weight edits, NO WSP, NO checkpoint modification.

Tests, on identical token IDs:
  A) raw prompt, add_special_tokens=False  (what judge_sovereign_bitnet.py does)
  B) tok(prompt) with add_special_tokens=True (what build script generation does)
For each: packed Quillan model top-5 for the 10 FACTUAL prompts.
Also loads donor Qwen float (if available) and scores A/B for comparison.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
FACTUAL = [
    ("The capital of France is", [" Paris"]),
    ("The capital of Japan is", [" Tokyo"]),
    ("The largest planet in our solar system is", [" Jupiter"]),
    ("Water freezes at zero degrees", [" Celsius", " celsius"]),
    ("The sun rises in the", [" east"]),
    ("2 + 2 equals", [" 4", "4"]),
    ("Hello, my name is", [" John", " Alex", " Anna"]),
    ("The opposite of hot is", [" cold"]),
    ("A dog is a type of", [" animal"]),
    ("The Earth orbits around the", [" sun"]),
]


def topk_texts(logits, tok, k=5):
    return [tok.decode([i]) for i in torch.topk(logits, k).indices.tolist()]


def packed_logits(model, ids):
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        o = model(x)
    lg = o[0] if isinstance(o, (tuple, list)) else o
    return lg[0, -1, :].float()


def main():
    from transformers import AutoTokenizer

    snaps = glob.glob(
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
        )
    )
    tok = AutoTokenizer.from_pretrained(snaps[0])
    print(
        f"[tok] {type(tok).__name__} vocab={tok.vocab_size} bos={tok.bos_token!r} eos={tok.eos_token!r}"
    )
    print(
        f"[tok] add_special_tokens sample A={tok.encode('The capital of France is', add_special_tokens=False)}"
    )
    print(
        f"[tok] add_special_tokens sample B={tok('The capital of France is', add_special_tokens=True)['input_ids']}"
    )

    from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

    packed = QuillanSovereignBitNetLM.load_packed_safetensors(CKPT).eval()
    print(f"[model] packed loaded from {CKPT.name}")

    for mode in ("A_no_special", "B_with_special"):
        hits = 0
        print(f"\n=== PACKED mode {mode} ===")
        for text, expected in FACTUAL:
            if mode == "A_no_special":
                ids = tok.encode(text, add_special_tokens=False)
            else:
                ids = tok(text, add_special_tokens=True)["input_ids"]
            top5 = topk_texts(packed_logits(packed, ids), tok)
            hit = any(any(e in t for t in top5) for e in expected)
            hits += int(hit)
            print(f"  {'HIT ' if hit else 'miss'} {text[:38]:38} -> {top5[:3]}")
        print(f"  MODE {mode}: {hits}/10 hits")
    print(
        "\n[differentiator] If B >> A, judge harness (stripped special tokens) is the cause."
    )


if __name__ == "__main__":
    main()
