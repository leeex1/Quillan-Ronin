#!/usr/bin/env python3
"""Smoke test the compiled flagship: encode with canonical Quillan vocab, generate."""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DEST = ROOT / "exports" / "quillan_flagship_compiled"


def build_encoder(text2q):
    # index by first character for greedy longest match
    by_first = {}
    for s in text2q:
        if s:
            by_first.setdefault(s[0], []).append(s)
    for k in by_first:
        by_first[k].sort(key=len, reverse=True)
    return by_first


def encode(text, text2q, by_first):
    ids = []
    i = 0
    while i < len(text):
        cands = by_first.get(text[i], ())
        match = None
        for s in cands:
            if text.startswith(s, i):
                match = s
                break
        if match is None:
            i += 1
            continue
        ids.append(text2q[match])
        i += len(match)
    return ids


def main():
    text2q = json.loads(
        (ROOT / "exports" / "quillan_vocab.json").read_text(encoding="utf-8")
    )
    q2text = {v: k for k, v in text2q.items()}
    by_first = build_encoder(text2q)
    prompts = ["The capital of France is", "2 + 2 =", "Hello, my name is"]
    print("=== encoded prompts ===", flush=True)
    enc = {}
    for p in prompts:
        ids = encode(p, text2q, by_first)
        enc[p] = ids
        print(f"  {p!r} -> {ids}", flush=True)

    print("\n=== load model ===", flush=True)
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        str(DEST), local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    print(f"  loaded, emb={tuple(m.get_input_embeddings().weight.shape)}", flush=True)

    print("\n=== generate ===", flush=True)
    with torch.inference_mode():
        for p, ids in enc.items():
            inp = torch.tensor([ids])
            out = m.generate(
                inp,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=128001,
            )
            gen = out[0, len(ids) :].tolist()
            txt = "".join(q2text.get(t, f"<{t}>") for t in gen)
            print(f"  {p!r} -> {txt!r}", flush=True)


if __name__ == "__main__":
    main()
