#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Judge a checkpoint with its OWN tokenizer (matches embedding vocab)."""

import argparse, glob, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
from judge_sovereign_bitnet import FACTUAL
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokdir", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(a.tokdir), local_files_only=True)
    model = QuillanSovereignBitNetLM.load_packed_safetensors(a.checkpoint).eval()
    print(f"[model] vocab={model.cfg.vocab_size}  [tok] vocab={len(tok)}")
    hits = 0
    for text, expected in FACTUAL:
        ids = tok.encode(text, add_special_tokens=False)
        if max(ids) >= model.cfg.vocab_size:
            print(
                f"  SKIP {text[:35]!r}: token id {max(ids)} >= vocab {model.cfg.vocab_size}"
            )
            continue
        x = torch.tensor([ids])
        with torch.inference_mode():
            o = model(x)
        lg = o[0] if isinstance(o, (tuple, list)) else o
        top5 = [tok.decode([i]) for i in torch.topk(lg[0, -1, :], 5).indices.tolist()]
        hit = any(any(e in t for t in top5) for e in expected)
        hits += int(hit)
        print(f"  {'HIT ' if hit else 'miss'} {text[:38]:38} -> {top5[:3]}")
    print(f"J4 matched-tokenizer: {hits}/10")


from contextlib import contextmanager


@contextmanager
def safe_embed_check(model, tok):
    print(f"[model] vocab={model.cfg.vocab_size}  [tok] vocab={len(tok)}")


if __name__ == "__main__":
    main()
