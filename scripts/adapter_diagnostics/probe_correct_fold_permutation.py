#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PERMUTATION CONTROL IN CORRECT-FOLD LOGIT SPACE — READ-ONLY.

The decisive kill test before any WSP authorization. Fit ridge R from custom
decomposition hidden -> donor hidden on TRAIN prompts under:
  (a) TRUE pairing     custom[p] -> donor[p]
  (b) DERANGED pairing custom[p] -> donor[perm[p]]  (no fixed points)
Evaluate BOTH on HELD-OUT prompts with TRUE pairing, using the CORRECT fold
W_eff = Wu @ R.T fed raw h_c (logits = h_c @ R @ Wu.T).

If TRUE doesn't beat DERANGED on held-out top1/top5/logit-cos, the signal is
base-rate geometry, not donor transfer. No files written.
"""

from __future__ import annotations

import glob
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DONOR_CACHE = ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"
CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
PE_RESCALE = 0.01924
STAGE = "ingestion"  # best gate-C config from #1051
LAMS = [0.01, 0.1]

from probe_construct_readout_map import TRAIN, HOLDOUT  # noqa: E402


def capture(model: Any, ids: List[int]) -> torch.Tensor:
    out: Dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: Tuple[Any, ...], o: torch.Tensor) -> None:
        t = o[0] if isinstance(o, (tuple, list)) else o
        if isinstance(t, torch.Tensor):
            out["h"] = t.detach().float().clone()

    h = model.ingestion.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(torch.tensor([ids], dtype=torch.long))
    finally:
        h.remove()
    return out["h"][0]


def donor_hl(donor: Any, ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = donor(torch.tensor([ids], dtype=torch.long), output_hidden_states=True)
    return out.hidden_states[-1][0].detach().float(), out.logits[0].detach().float()


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("CORRECT-FOLD PERMUTATION CONTROL (stage=ingestion, strict holdout)")
    print("=" * 78)

    plq.CKPT = CKPT
    custom, _ = plq.build_real_model()
    for name, param in custom.named_parameters():
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

    # capture all activations ONCE (expensive), then cheap fits
    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    t0 = time.perf_counter()
    for i, p in enumerate(TRAIN + HOLDOUT):
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        hc = capture(custom, ids)
        hd, dl = donor_hl(donor, ids)
        T = min(hc.shape[0], hd.shape[0], dl.shape[0])
        cache[p] = {"hc": hc[:T], "hd": hd[:T], "dl": dl[:T]}
        if (i + 1) % 15 == 0:
            print(
                f"  captured {i + 1}/{len(TRAIN) + len(HOLDOUT)} ({time.perf_counter() - t0:.1f}s)"
            )

    Xt = torch.cat([cache[p]["hc"] for p in TRAIN], 0)
    Ht = {p: cache[p]["hd"] for p in TRAIN}

    def solve(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
        A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
        return torch.linalg.solve(A, X.T @ Y)

    def evaluate(R: torch.Tensor, lam: float) -> Dict[str, float]:
        W = Wu @ R.T  # correct fold
        o1s, o5s, cs = [], [], []
        for p in HOLDOUT:
            hc, dl = cache[p]["hc"], cache[p]["dl"]
            t = hc.shape[0] - 1
            with torch.no_grad():
                cl = (hc[t : t + 1] @ W.T)[0]
            d = dl[t]
            o1s.append(
                int(
                    bool(
                        set(torch.topk(cl, 1).indices.tolist())
                        & set(torch.topk(d, 1).indices.tolist())
                    )
                )
            )
            o5s.append(
                len(
                    set(torch.topk(cl, 5).indices.tolist())
                    & set(torch.topk(d, 5).indices.tolist())
                )
            )
            cs.append(
                float(F.cosine_similarity(cl.unsqueeze(0), d.unsqueeze(0)).item())
            )
        return {
            "t1": sum(o1s) / len(o1s),
            "t5": sum(o5s) / len(o5s),
            "cos": sum(cs) / len(cs),
            "t1n": sum(o1s),
            "t5n": sum(1 for x in o5s if x > 0),
        }

    rng = random.Random(20261121)
    for lam in LAMS:
        # TRUE pairing
        R_true = solve(Xt, torch.cat([Ht[p] for p in TRAIN], 0), lam)
        # DERANGED pairing at prompt level
        idx = list(range(len(TRAIN)))
        while True:
            rng.shuffle(idx)
            if all(idx[j] != j for j in range(len(idx))):
                break
        Y_perm = torch.cat([Ht[TRAIN[idx[j]]] for j in range(len(TRAIN))], 0)
        R_perm = solve(Xt, Y_perm, lam)
        # random-map control (equal norm)
        scale = (
            torch.cat([Ht[p] for p in TRAIN], 0).norm(dim=1).mean()
            / Xt.norm(dim=1).mean()
        ).item()
        R_rand = torch.randn(896, 896) * (scale / (896**0.5))

        rt = evaluate(R_true, lam)
        rp = evaluate(R_perm, lam)
        rr = evaluate(R_rand, lam)
        print(f"\nlam={lam}")
        print(
            f"  TRUE     t1={rt['t1']:.4f}({rt['t1n']}/20) t5={rt['t5']:.4f}({rt['t5n']}/20) cos={rt['cos']:+.4f}"
        )
        print(
            f"  DERANGED t1={rp['t1']:.4f}({rp['t1n']}/20) t5={rp['t5']:.4f}({rp['t5n']}/20) cos={rp['cos']:+.4f}"
        )
        print(
            f"  RANDOM   t1={rr['t1']:.4f}({rr['t1n']}/20) t5={rr['t5']:.4f}({rr['t5n']}/20) cos={rr['cos']:+.4f}"
        )
        verdict = (
            "SURVIVES"
            if (rt["t1"] > rp["t1"] or rt["t5"] > rp["t5"] + 0.1)
            and rt["t1"] > rr["t1"]
            else "COLLAPSES"
        )
        print(f"  PERMUTATION VERDICT: {verdict}")

    print("\n" + "=" * 78)
    print("DONE — no files written, WSP untouched")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
