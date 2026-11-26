#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LAYER-BY-LAYER SEMANTIC PROBE (user's discipline: build/check each layer).

Runs Gemini's mini BitNet model forward ONE LAYER AT A TIME, and after each layer
decodes the hidden state through the tied embedding head to see whether the
factual answer (Paris, Tokyo, east...) is present at that depth. This localizes
EXACTLY which layer destroys the donor's semantic knowledge.

Read-only. No edits to model/checkpoint/gate. No WSP.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(r"C:\02_QUILLAN")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from quillan_sovereign_bitnet import QuillanSovereignBitNetLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CKPT = Path(r"C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet.safetensors")
PROBES = [
    ("The capital of France is", [" Paris"]),
    ("The sun rises in the", [" east"]),
    ("2 + 2 equals", [" 4", "4"]),
    ("The capital of Japan is", [" Tokyo"]),
]


def main() -> int:
    model = QuillanSovereignBitNetLM.load_packed_safetensors(CKPT)
    model.eval()
    snaps = glob.glob(
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
        )
    )
    tok = AutoTokenizer.from_pretrained(snaps[0])
    E = model.embed_tokens.unpack_float()  # [vocab, hidden] tied head

    print("=" * 82)
    print("LAYER-BY-LAYER SEMANTIC PROBE — where does the answer vanish?")
    print(f"model layers: {len(model.layers)}, hidden: {model.cfg.hidden_size}")
    print("=" * 82)

    # precompute token ids for each probe
    probe_data = []
    for text, expected in PROBES:
        ids = tok.encode(text, add_special_tokens=False)
        probe_data.append((text, expected, torch.tensor([ids], dtype=torch.long)))

    # embed once
    hs = [model.embed_tokens(ids) for _, _, ids in probe_data]
    # checkpoint 0 = right after embedding
    checkpoints = [("embed", [h.clone() for h in hs])]

    with torch.inference_mode():
        for idx, layer in enumerate(model.layers):
            hs = [layer(h)[0] for h in hs]
            checkpoints.append((f"L{idx:02d}", [h.clone() for h in hs]))

    # decode at each checkpoint
    for name, states in checkpoints:
        results = []
        for (text, expected, _), h in zip(probe_data, states):
            last = h[0, -1, :]  # [hidden]
            if name != "embed":
                last = model.norm(last.unsqueeze(0))[
                    0
                ]  # apply final norm for non-embed layers
            lg = F.linear(last.unsqueeze(0), E)[0]  # [vocab]
            top5 = [tok.decode([i]) for i in torch.topk(lg, 5).indices.tolist()]
            hit = any(any(e in t for t in top5) for e in expected)
            results.append("HIT" if hit else ".")
        print(f"  {name:6}  {' '.join(f'{r:>3}' for r in results)}")

    print("\nlegend: columns = " + " | ".join(t[:14] for t, _, _ in PROBES))
    print("=" * 82)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
