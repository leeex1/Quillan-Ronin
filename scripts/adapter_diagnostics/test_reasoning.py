#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reasoning probes for Quillan models (C8-METASYNTH design, board #1270).

Counterfactual content so J4 factual memorization cannot cheat.
Rubric 0-2 each (/12): 2 = correct answer PLUS explicit intermediate step;
1 = correct answer with no derivation; 0 = wrong/contradiction missed.
Pass = >=4 twos AND >=9 total. Echo/null controls must score <=2.
"""

from __future__ import annotations
import sys, re
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

PROBES = [
    ("transitive", "P beats Q, Q beats R, R beats S. Who is strongest, P or S?", ["P"]),
    (
        "math",
        "3 jobs per second for 2 minutes, then double the rate for 1 minute. Total jobs? Show work.",
        ["720"],
    ),
    (
        "rule",
        "A 'flib' is even AND greater than 10. Is 14 a flib? Is 8? Explain.",
        ["yes", "no"],
    ),
    (
        "causal",
        "All Zorps are glinty; glinty things rust in rain, but Zorps are waterproof. Does a Zorp rust?",
        ["no"],
    ),
    (
        "contradiction",
        "Vault opens only if the red key is turned. The key was never turned. The vault opened. Consistent?",
        ["no", "inconsistent"],
    ),
    (
        "second_order",
        "Ann outranks Bob, Bob outranks Cam. Now Cam outranks Ann. What must be false?",
        ["cycle", "cannot", "ann", "cam"],
    ),
]
CONNECTIVES = ["because", "so ", "therefore", "thus", "since", "then"]


def tok_for(v):
    from transformers import AutoTokenizer

    if v <= 128256:
        return AutoTokenizer.from_pretrained(
            r"E:\source_models\llama_1b", local_files_only=True
        )
    return AutoTokenizer.from_pretrained(
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
        local_files_only=True,
    )


def run(ckpt):
    print("=" * 78)
    print(f"REASONING PROBES: {ckpt.name}")
    print("=" * 78)
    m = QuillanSovereignBitNetLM.load_packed_safetensors(ckpt).eval()
    tok = tok_for(m.cfg.vocab_size)
    total = 0
    twos = 0
    for name, prompt, markers in PROBES:
        ids = tok.encode(prompt, add_special_tokens=False)
        with torch.inference_mode():
            out = m.generate(torch.tensor([ids]), max_new_tokens=96, temperature=0.0)
        gen = out[0, len(ids) :].tolist()
        text = tok.decode(gen, skip_special_tokens=True).lower()
        correct = any(mk.lower() in text for mk in markers)
        step = any(c in text for c in CONNECTIVES)
        score = 2 if (correct and step) else (1 if correct else 0)
        total += score
        twos += int(score == 2)
        print(f"\n[{name}] score={score}/2 correct={correct} step={step}")
        print(f"  Q: {prompt[:80]}")
        print(
            f"  A: {tok.decode(gen, skip_special_tokens=True)[:400].replace(chr(10), ' ')}"
        )
    print(
        f"\n[RESULT] total={total}/12 twos={twos} -> {'PASS' if (twos >= 4 and total >= 9) else 'FAIL'}"
    )
    print(
        "[NOTE] answer-only (score 1) = no trained CoT; needs SFT on step traces, not a better judge."
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    for c in ap.parse_args().ckpt:
        run(c)
