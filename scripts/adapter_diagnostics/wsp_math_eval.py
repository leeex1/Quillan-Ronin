#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast A/B/C/D math scorer: ONE forward pass per problem (no generation).

Loads a QuillanSovereignBitNet checkpoint, formats each of the 20 problems as
multiple choice ending in "Answer:", reads the next-token logits, and picks the
letter with the highest logit. This is the standard MMLU-style log-likelihood
eval and runs in seconds, unlike autoregressive generation.

Usage:
  python wsp_math_eval.py --ckpt <path> [--model-tag name] [--json out.json]
"""

from __future__ import annotations
import sys, json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hard_math20 import PROBLEMS, KEY, LETTERS, format_problem

QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
LLAMA = r"E:\source_models\llama_1b"


def tok_for(vocab: int):
    from transformers import AutoTokenizer

    if vocab <= 128256:
        return AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    return AutoTokenizer.from_pretrained(QWEN, local_files_only=True)


def letter_ids(tok, vocab: int) -> List[int]:
    """Token ids for ' A',' B',' C',' D' (and bare fallback), valid for embedding table."""
    out = []
    for L in LETTERS:
        ids = []
        for cand in (f" {L}", L):
            enc = tok.encode(cand, add_special_tokens=False)
            if len(enc) == 1 and enc[0] < vocab:
                ids.append(enc[0])
        out.append(
            ids[0] if ids else tok.encode(f" {L}", add_special_tokens=False)[0] % vocab
        )
    return out


def load_model(path: Path):
    from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

    m = QuillanSovereignBitNetLM.load_packed_safetensors(Path(path)).eval()
    return m


def last_logits(model, ids: List[int]) -> torch.Tensor:
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits, _ = model(x, use_cache=False)
    return logits[0, -1, :].float()


def score_checkpoint(ckpt: Path, model=None, tok=None) -> dict:
    m = model or load_model(ckpt)
    tok = tok or tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    # logit for a letter = max over its candidate token ids
    preds, correct = [], 0
    details = []
    for i in range(len(PROBLEMS)):
        ids = tok.encode(format_problem(i), add_special_tokens=False)
        lg = last_logits(m, ids)
        scores = [
            float(max(lg[t] for t in cand))
            if isinstance(cand, list)
            else float(lg[cand])
            for cand in lids
        ]
        p = int(max(range(4), key=lambda j: scores[j]))
        preds.append(p)
        ok = p == KEY[i]
        correct += int(ok)
        details.append(
            {
                "i": i + 1,
                "pred": LETTERS[p],
                "gold": LETTERS[KEY[i]],
                "ok": ok,
                "scores": [round(s, 2) for s in scores],
            }
        )
    return {
        "ckpt": str(ckpt),
        "n": len(PROBLEMS),
        "correct": correct,
        "accuracy": correct / len(PROBLEMS),
        "preds": preds,
        "details": details,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    r = score_checkpoint(a.ckpt)
    tag = a.model_tag or a.ckpt.name
    print(f"{'=' * 70}")
    print(f"HARD-MATH20 SCORE | {tag}")
    print(f"{'=' * 70}")
    print(f"accuracy: {r['correct']}/20 = {r['accuracy']:.0%}")
    for d in r["details"]:
        mark = "OK " if d["ok"] else "MISS"
        print(
            f"  {mark} Q{d['i']:2d}: pred={d['pred']} gold={d['gold']}  scores={d['scores']}"
        )
    if a.json:
        a.json.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
