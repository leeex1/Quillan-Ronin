#!/usr/bin/env python3
"""Smoke test: prove both faithful models actually GENERATE text (not just MCQ argmax).
Greedy decode a short completion. No training, no backprop.
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
from wsp_math_eval import tok_for, load_model

torch.set_num_threads(2)
MODELS = {
    "mini": "quillan-mini-hifi-g4-councilfixed.safetensors",
    "flagship": "quillan-flagship-hifi-g4-exact.safetensors",
}
PROMPTS = ["The capital of Japan is", "Water is composed of", "2 + 2 equals"]


def main():
    for name, f in MODELS.items():
        m = load_model(ROOT / "exports" / f)
        tok = tok_for(m.cfg.vocab_size)
        print(f"\n=== {name} ===", flush=True)
        for text in PROMPTS:
            ids = tok.encode(text, add_special_tokens=False)
            gen = list(ids)
            with torch.inference_mode():
                for _ in range(8):
                    lg = m(torch.tensor([gen]))[0][0, -1, :].float()
                    nxt = int(lg.argmax())
                    gen.append(nxt)
            out = (
                tok.decode(gen[len(ids) :], errors="replace")
                if hasattr(tok, "decode")
                else str(gen[len(ids) :])
            )
            print(f"  '{text}' -> '{out}'", flush=True)
        del m


if __name__ == "__main__":
    main()
