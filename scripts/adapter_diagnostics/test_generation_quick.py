#!/usr/bin/env python3
"""Fast generation probe: 3 long prompts x 80 tokens, unbuffered output."""

import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

LONG = [
    "Explain why the sky appears blue during the day and red at sunset.",
    "Write a short story about a ronin who guards a mountain village.",
    "Describe the Quillan cognitive architecture and how the council deliberates.",
]


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


def rep5(ids):
    if len(ids) < 5:
        return 0.0
    g = [tuple(ids[i : i + 5]) for i in range(len(ids) - 4)]
    return 1 - len(set(g)) / len(g)


def run(ckpt):
    print(f"=== {ckpt.name} ===", flush=True)
    m = QuillanSovereignBitNetLM.load_packed_safetensors(ckpt).eval()
    tok = tok_for(m.cfg.vocab_size)
    for pr in LONG:
        ids = tok.encode(pr, add_special_tokens=False)
        with torch.inference_mode():
            out = m.generate(torch.tensor([ids]), max_new_tokens=80, temperature=0.0)
        gen = out[0, len(ids) :].tolist()
        text = tok.decode(gen, skip_special_tokens=True)
        dtr = len(set(gen)) / max(1, len(gen))
        print(
            f"\nQ: {pr[:60]}\ndistinct={dtr:.3f} rep5={rep5(gen):.3f}\nA: {text[:500].replace(chr(10), ' ')}",
            flush=True,
        )
    del m


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    for c in ap.parse_args().ckpt:
        run(c)
