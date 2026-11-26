#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FINAL_NORM CORRECT-FOLD + PERMUTATION — READ-ONLY, STRICT HOLDOUT.

The gate reads logits = txt_dec(final_norm(x_final)) at v8:1359.
So R must be fit from FINAL_NORM hidden -> donor hidden, not ingestion.

One activation capture (40 train + 20 holdout), then:
  1. correct fold W_eff = Wu @ R.T fed raw h_final, on strict holdout
  2. permutation control (true vs deranged vs random) in same logit space
If holdout top1/top5 > 0 AND true beats deranged -> rebuild candidate.
"""

from __future__ import annotations

import glob
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

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
LAMS = [1e-3, 1e-2, 1e-1]

from probe_construct_readout_map import TRAIN, HOLDOUT  # noqa: E402


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("FINAL_NORM CORRECT-FOLD + PERMUTATION (40 train / 20 strict holdout)")
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

    out: Dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: tuple, o: torch.Tensor) -> None:
        out["h"] = o.detach().float().clone()

    h = custom.final_norm.register_forward_hook(hook)

    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    t0 = time.perf_counter()
    try:
        for i, p in enumerate(TRAIN + HOLDOUT):
            ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
            with torch.no_grad():
                _ = custom(torch.tensor([ids], dtype=torch.long))
            hc = out["h"][0]
            with torch.no_grad():
                dm = donor(
                    torch.tensor([ids], dtype=torch.long), output_hidden_states=True
                )
            hd, dl = (
                dm.hidden_states[-1][0].detach().float(),
                dm.logits[0].detach().float(),
            )
            T = min(hc.shape[0], hd.shape[0], dl.shape[0])
            cache[p] = {"hc": hc[:T], "hd": hd[:T], "dl": dl[:T]}
            if (i + 1) % 15 == 0:
                print(
                    f"  captured {i + 1}/{len(TRAIN) + len(HOLDOUT)} ({time.perf_counter() - t0:.1f}s)"
                )
    finally:
        h.remove()

    Xt = torch.cat([cache[p]["hc"] for p in TRAIN], 0)
    Yt_all = torch.cat([cache[p]["hd"] for p in TRAIN], 0)
    scale = (Yt_all.norm(dim=1).mean() / Xt.norm(dim=1).mean()).item()

    def solve(Y: torch.Tensor, lam: float) -> torch.Tensor:
        A = Xt.T @ Xt + lam * torch.eye(Xt.shape[1], dtype=Xt.dtype)
        return torch.linalg.solve(A, Xt.T @ Y)

    def evaluate(R: torch.Tensor) -> Dict[str, float]:
        W = Wu @ R.T
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
        R_true = solve(Yt_all, lam)
        idx = list(range(len(TRAIN)))
        while True:
            rng.shuffle(idx)
            if all(idx[j] != j for j in range(len(idx))):
                break
        Y_perm = torch.cat([cache[TRAIN[idx[j]]]["hd"] for j in range(len(TRAIN))], 0)
        R_perm = solve(Y_perm, lam)
        R_rand = torch.randn(896, 896) * (scale / (896**0.5))

        rt, rp, rr = evaluate(R_true), evaluate(R_perm), evaluate(R_rand)
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
        gate_c = rt["t1"] > 0 and rt["t5"] > 0
        perm = (rt["t1"] > rp["t1"] or rt["t5"] > rp["t5"] + 0.1) and rt["t1"] > rr[
            "t1"
        ]
        print(
            f"  gate-C condition: {'PASS' if gate_c else 'FAIL'} | permutation: {'SURVIVES' if perm else 'COLLAPSES'}"
        )

    print("\n" + "=" * 78)
    print("DONE — no files written")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
