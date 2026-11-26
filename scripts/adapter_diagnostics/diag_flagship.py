#!/usr/bin/env python3
"""Why is flagship GK 2/25? Check config + raw logit distribution + tokenizer fit."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache


def run(ckpt):
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(ckpt)
    enable_cache(m)
    cfg = m.cfg
    print(f"\n=== {ckpt.name} ===")
    print(
        f" vocab={getattr(cfg, 'vocab_size', None)} hidden={getattr(cfg, 'hidden_size', None)} layers={len(m.layers)} "
        f"lm_head={'tied' if m.lm_head is None else 'separate rank' + str(getattr(m.lm_head, 'eggroll_rank', 0))}"
    )
    print(f" letter_ids={lids}")
    L = []
    for i, (q, opts, _) in enumerate(GQ):
        p = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{Lx}) {o}\n" for Lx, o in zip(LETTERS, opts))
            + "Answer:"
        )
        ids = tok.encode(p, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids].float()
        L.append(lg)
    L = torch.stack(L)
    pred = L.argmax(-1)
    K = torch.tensor(GK)
    print(
        f" GK={(pred == K).sum().item()}/25 pred_dist={[int((pred == j).sum()) for j in range(4)]} true_dist={[int((K == j).sum()) for j in range(4)]}"
    )
    print(
        f" logit mean={[round(float(x), 2) for x in L.mean(0)]} std={[round(float(x), 2) for x in L.std(0)]}"
    )
    print(
        f" argmax-margin mean={float((L.topk(2, dim=-1).values[:, 0] - L.topk(2, dim=-1).values[:, 1]).mean()):.3f}"
    )


for n in [
    "quillan-mini-sovereign-bitnet.safetensors",
    "quillan-flagship-sovereign-bitnet.safetensors",
]:
    run(ROOT / "exports" / n)
