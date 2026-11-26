#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short-form + long-form generation test for Quillan mini/flagship checkpoints.

Short-form: 5 prompts, 24 tokens each — factual + instruction following.
Long-form:  3 prompts, 200 tokens each — coherence, repetition, degradation.
Reports distinct-token ratio, repeated-ngram rate, and prints samples.
Read-only on checkpoints. No backprop.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

SHORT = [
    "The capital of France is",
    "Water freezes at",
    "Explain why binary search has O(log n) time complexity:",
    "The Bushido virtue of rectitude means",
    "2 + 2 equals",
]
LONG = [
    "Explain in detail why the sky appears blue during the day and red at sunset.",
    "Write a short story about a ronin who guards a mountain village.",
    "Describe the Quillan cognitive architecture and how the council deliberates.",
]


def tok_for(vocab):
    from transformers import AutoTokenizer

    if vocab <= 128256:
        return AutoTokenizer.from_pretrained(
            r"E:\source_models\llama_1b", local_files_only=True
        )
    return AutoTokenizer.from_pretrained(
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775",
        local_files_only=True,
    )


def ngram_repeat(ids, n=5):
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def run(ckpt):
    print("=" * 78)
    print(f"CHECKPOINT: {ckpt.name}")
    print("=" * 78)
    m = QuillanSovereignBitNetLM.load_packed_safetensors(ckpt).eval()
    tok = tok_for(m.cfg.vocab_size)
    print(
        f"[model] vocab={m.cfg.vocab_size} layers={m.cfg.num_hidden_layers} tokenizer={type(tok).__name__}"
    )
    for label, prompts, ntok in [("SHORT", SHORT, 24), ("LONG", LONG, 200)]:
        print(f"\n--- {label} ({ntok} tokens) ---")
        for prompt in prompts:
            ids = tok.encode(prompt, add_special_tokens=False)
            x = torch.tensor([ids])
            with torch.inference_mode():
                out = m.generate(x, max_new_tokens=ntok, temperature=0.0)
            gen = out[0, len(ids) :].tolist()
            text = tok.decode(gen, skip_special_tokens=True)
            dtr = len(set(gen)) / max(1, len(gen))
            rep = ngram_repeat(gen, 5)
            print(f"\nPROMPT: {prompt[:70]}")
            print(f"  distinct={dtr:.3f} repeat5gram={rep:.3f} tokens={len(gen)}")
            print(f"  TEXT: {text[:600].replace(chr(10), ' ')}")
    del m


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    for c in ap.parse_args().ckpt:
        run(c)
