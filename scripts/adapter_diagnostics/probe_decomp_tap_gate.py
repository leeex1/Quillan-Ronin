#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DECOMPOSITION-TAP GATE VALIDATION — READ-ONLY, IN-MEMORY.

Monkeypatch the model forward so logits are read from DECOMPOSITION hidden
(the last stage carrying donor signal) through the correct-fold head, instead
of from final_norm. Then compute gate A and C logic EXACTLY (same prompts,
same thresholds) to see if the plumbing fix would pass.

No source file edited. No checkpoint written. No WSP.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
DONOR_CACHE = ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"
PE_RESCALE = 0.01924
LAM = 0.01

# gate prompts (exact from conversion_gate.py)
GATE_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
]
PROSE_A, PROSE_B = GATE_PROMPTS[0], GATE_PROMPTS[2]

from probe_construct_readout_map import TRAIN, HOLDOUT  # noqa: E402


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("DECOMPOSITION-TAP GATE VALIDATION (in-memory forward patch)")
    print("=" * 78)

    plq.CKPT = CKPT
    model, _ = plq.build_real_model()
    for name, param in model.named_parameters():
        if name.endswith("pos_embed"):
            with torch.no_grad():
                param.data.mul_(PE_RESCALE)
            break

    snaps = glob.glob(
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
        )
    )
    tok = AutoTokenizer.from_pretrained(snaps[0])
    donor = AutoModelForCausalLM.from_pretrained(snaps[0]).eval()
    Wu = torch.load(str(DONOR_CACHE), map_location="cpu", weights_only=False)[
        "embed_tokens"
    ].float()

    # capture decomposition hidden for TRAIN prompts, fit R from decomposition -> donor hidden
    Xs, Ys = [], []
    out: Dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: tuple, o: torch.Tensor) -> None:
        out["h"] = o.detach().float().clone()

    h = model.decomposition.register_forward_hook(hook)
    try:
        for p in TRAIN:
            ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
            with torch.no_grad():
                _ = model(torch.tensor([ids], dtype=torch.long))
            hc = out["h"][0]
            with torch.no_grad():
                dm = donor(
                    torch.tensor([ids], dtype=torch.long), output_hidden_states=True
                )
            hd = dm.hidden_states[-1][0].detach().float()
            T = min(hc.shape[0], hd.shape[0])
            Xs.append(hc[:T])
            Ys.append(hd[:T])
    finally:
        h.remove()
    X = torch.cat(Xs, 0)
    Y = torch.cat(Ys, 0)
    A = X.T @ X + LAM * torch.eye(X.shape[1], dtype=X.dtype)
    R = torch.linalg.solve(A, X.T @ Y)
    W_eff = Wu @ R.T
    print(f"fit R from decomposition on {tuple(X.shape)} TRAIN pairs; LAM={LAM}")

    # helper: decomposition hidden -> logits via correct fold
    # BUG FIX: the fit-loop hook was removed at line 100, so out["h"] was stale.
    # Register a FRESH hook per call so each prompt captures its own hidden.
    def decomp_logits(p: str) -> torch.Tensor:
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        fresh: Dict[str, torch.Tensor] = {}

        def fresh_hook(_m: Any, _i: tuple, o: torch.Tensor) -> None:
            t = o[0] if isinstance(o, (tuple, list)) else o
            if isinstance(t, torch.Tensor):
                fresh["h"] = t.detach().float().clone()

        hh = model.decomposition.register_forward_hook(fresh_hook)
        try:
            with torch.no_grad():
                _ = model(torch.tensor([ids], dtype=torch.long))
        finally:
            hh.remove()
        hc = fresh["h"][0]  # [T,896] — fresh for THIS prompt
        with torch.no_grad():
            return (hc @ W_eff.T).float()

    # frozen donor logits
    donor_acts = torch.load(
        str(DIAG / "_acts" / "donor_acts.pt"), map_location="cpu", weights_only=False
    )

    # ---- GATE A: prompt sensitivity (two prose prompts, last position) ----
    la, lb = decomp_logits(PROSE_A), decomp_logits(PROSE_B)
    A_cos = float(F.cosine_similarity(la[-1].unsqueeze(0), lb[-1].unsqueeze(0)).item())
    A_t1a, A_t1b = int(la[-1].argmax()), int(lb[-1].argmax())
    A_pass = A_cos < 0.99 and A_t1a != A_t1b
    print(
        f"\n[A] PROMPT SENSITIVITY: cos={A_cos:.6f} top1_A={tok.decode([A_t1a])!r} "
        f"top1_B={tok.decode([A_t1b])!r} differ={A_t1a != A_t1b} -> {'PASS' if A_pass else 'FAIL'}"
    )

    # ---- GATE C: donor top-k overlap on 5 frozen prompts ----
    print("\n[C] DONOR TOP-K OVERLAP (decomp tap, frozen prompts):")
    tot1 = tot5 = 0.0
    for p in GATE_PROMPTS:
        cl = decomp_logits(p)[-1]
        dl = donor_acts[p]["logits"][0, -1, :].float()
        c1 = set(torch.topk(cl, 1).indices.tolist())
        d1 = set(torch.topk(dl, 1).indices.tolist())
        c5 = set(torch.topk(cl, 5).indices.tolist())
        d5 = set(torch.topk(dl, 5).indices.tolist())
        o1, o5 = len(c1 & d1), len(c5 & d5)
        tot1 += o1
        tot5 += o5
        print(f"  {p[:40]!r:43} top1={o1} top5={o5}")
    m1, m5 = tot1 / len(GATE_PROMPTS), tot5 / len(GATE_PROMPTS)
    C_pass = m1 > 0 and m5 > 0
    print(f"  mean top1={m1:.4f} top5={m5:.4f} -> {'PASS' if C_pass else 'FAIL'}")

    print("\n" + "=" * 78)
    print(
        f"DECOMP-TAP GATE: A={'PASS' if A_pass else 'FAIL'} C={'PASS' if C_pass else 'FAIL'}"
    )
    print("No files written, no WSP.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
