#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POOLED READOUT STRICT HOLDOUT — READ-ONLY, IN-MEMORY.

Corrects the in-sample flaw (#1070). Fit R and mu on 40 TRAIN prompts only;
evaluate gate A (prose pair) and gate C (donor top-k) on 20 HELD-OUT prompts.
Includes DERANGED pairing control in pooled space.

Pooled logits = ((mean_all(h_final_norm) - mu) @ W.T), W = Wu @ R.T.
No files written, no WSP.
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
LAMS = [1e-2, 1e-1]

from probe_construct_readout_map import TRAIN, HOLDOUT  # noqa: E402

PROSE_A, PROSE_B = "The capital of France is", "Hello, my name is"


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("POOLED READOUT STRICT HOLDOUT (fit 40 train / test 20 holdout + deranged)")
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

    hh = custom.final_norm.register_forward_hook(hook)

    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    t0 = time.perf_counter()
    try:
        capture_list = list(dict.fromkeys(TRAIN + HOLDOUT + [PROSE_A, PROSE_B]))
        for i, p in enumerate(capture_list):
            ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
            with torch.no_grad():
                _ = custom(torch.tensor([ids], dtype=torch.long))
            hpos = out["h"][0]  # [T,896]
            # C11 #1087: terminal-anchored weighted mean, w=0.80 (plus mean_all for comparison)
            W_TERM = 0.80
            hf_mean = hpos.mean(dim=0)
            hf_term = (
                (1 - W_TERM) * hpos[:-1].mean(dim=0) + W_TERM * hpos[-1]
                if hpos.shape[0] > 1
                else hpos[-1]
            )
            with torch.no_grad():
                dm = donor(
                    torch.tensor([ids], dtype=torch.long), output_hidden_states=True
                )
            hd = (
                dm.hidden_states[-1][0].detach().float().mean(dim=0)
            )  # pooled donor hidden
            dl = (
                dm.logits[0, -1].detach().float()
            )  # donor LAST logit (gate compares to this)
            cache[p] = {"hf": hf_mean, "hf_term": hf_term, "hd": hd, "dl": dl}
            if (i + 1) % 15 == 0:
                print(
                    f"  captured {i + 1}/{len(TRAIN) + len(HOLDOUT)} ({time.perf_counter() - t0:.1f}s)"
                )
    finally:
        hh.remove()

    Xtr = torch.stack([cache[p]["hf"] for p in TRAIN], 0)
    Ytr = torch.stack([cache[p]["hd"] for p in TRAIN], 0)
    rng = random.Random(20261122)

    for lam in LAMS:
        A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], dtype=Xtr.dtype)
        R_true = torch.linalg.solve(A, Xtr.T @ Ytr)
        mu = Xtr.mean(dim=0, keepdim=True)

        # deranged pairing control
        idx = list(range(len(TRAIN)))
        while True:
            rng.shuffle(idx)
            if all(idx[j] != j for j in range(len(idx))):
                break
        Y_perm = torch.stack([Ytr[idx[j]] for j in range(len(idx))], 0)
        R_perm = torch.linalg.solve(A, Xtr.T @ Y_perm)
        scale = (Ytr.norm(dim=1).mean() / Xtr.norm(dim=1).mean()).item()
        R_rand = torch.randn(896, 896) * (scale / (896**0.5))

        def evaluate(R: torch.Tensor) -> Dict[str, float]:
            W = Wu @ R.T
            o1s, o5s, cs = [], [], []
            for p in HOLDOUT:
                v = cache[p]
                with torch.no_grad():
                    cl = (v["hf"] - mu[0]) @ W.T
                d = v["dl"]
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

        rt, rp, rr = evaluate(R_true), evaluate(R_perm), evaluate(R_rand)
        # gate A on the frozen prose pair using pooled+centered through true R
        W = Wu @ R_true.T
        la = (cache[PROSE_A]["hf"] - mu[0]) @ W.T
        lb = (cache[PROSE_B]["hf"] - mu[0]) @ W.T
        a_cos = float(F.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0)).item())
        a_differ = int(la.argmax()) != int(lb.argmax())

        print(f"\nlam={lam}")
        print(
            f"  GATE A (holdout prose): cos={a_cos:.4f} differ={a_differ} -> {'PASS' if a_cos < 0.99 and a_differ else 'FAIL'}"
        )
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
            f"  GATE C: {'PASS' if gate_c else 'FAIL'} | PERMUTATION: {'SURVIVES' if perm else 'COLLAPSES'}"
        )

    print("\n" + "=" * 78)
    print("DONE — no files written, WSP untouched")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
