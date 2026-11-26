#!/usr/bin/env python3
"""Runtime ablation: disable W1.58A8 activation quantization and re-score.

Diagnostic only, no file changes. If accuracy jumps from ~45% toward stock
Qwen's ~72%, the INT8 activation quantization is what destroys knowledge in
the forward pass (weights themselves round-trip at 0.99 cos).
"""

from __future__ import annotations
import sys, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
import quillan_sovereign_bitnet as QSB
from wsp_math_eval import tok_for, letter_ids, load_model
from heldout_fresh40 import prompts as p1, KEY as K1
from heldout_fresh40b import prompts as p2, KEY as K2
from heldout_fresh40c import prompts as p3, KEY as K3

torch.set_num_threads(1)


def score(m, tok, lids, pf, KEY, name):
    ok, ces, dist = 0, [], [0, 0, 0, 0]
    with torch.inference_mode():
        for seq, gold in zip(pf(tok), KEY):
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            pi = int(lg.argmax())
            dist[pi] += 1
            ok += int(pi == gold)
            ces.append(-F.log_softmax(lg, -1)[gold].item())
    ce = sum(ces) / len(ces)
    print(
        f"  {name}: {ok}/{len(KEY)} CE={ce:.4f} PPL={math.exp(ce):.3f} dist={dist}",
        flush=True,
    )


def run(label):
    m = load_model(ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors")
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    print(f"\n=== {label} ===", flush=True)
    for name, pf, KEY in (
        ("fresh40", p1, K1),
        ("fresh40b", p2, K2),
        ("fresh40c", p3, K3),
    ):
        score(m, tok, lids, pf, KEY, name)


def main():
    run("WITH activation quant (baseline)")
    orig = QSB.quantize_activations_w158a8
    QSB.quantize_activations_w158a8 = lambda x, eps=1e-5: x  # identity
    run("WITHOUT activation quant (ablation)")
    QSB.quantize_activations_w158a8 = orig


if __name__ == "__main__":
    main()
