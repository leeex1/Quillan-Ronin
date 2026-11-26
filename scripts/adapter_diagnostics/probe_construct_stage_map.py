#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STAGE-SOURCE READOUT CONSTRUCTION — READ-ONLY, IN-MEMORY, TRAIN/HOLDOUT.

Stage alignment diagnostic (#1000) found decomposition aligns to donor hidden
at cos +0.56 while final_norm is only +0.14. This probe builds the closed-form
map from the BEST stage (decomposition) instead of final_norm:

  h_decomp (896) --[ridge R]--> h_donor_hat (896) --[W_u]--> logits

Fold W_eff = W_u @ R into txt_dec.weight [151936,896] (gate D intact).
Fit on TRAIN prompts, evaluate on HELD-OUT prompts. No gaming.
"""

from __future__ import annotations

import argparse
import glob
import os
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
PE_RESCALE = 0.01924

from probe_construct_readout_map import TRAIN, HOLDOUT, top_tok  # reuse prompt sets


def capture_stage(model: Any, ids: List[int], stage: str) -> torch.Tensor:
    mods = {
        "ingestion": model.ingestion,
        "decomposition": model.decomposition,
        "diffusion_core": model.diffusion_core,
        "moe": model.moe,
        "final_norm": model.final_norm,
    }
    if stage not in mods:
        raise ValueError(stage)
    out: Dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: Tuple[Any, ...], o: torch.Tensor) -> None:
        t = o[0] if isinstance(o, (tuple, list)) else o
        if isinstance(t, torch.Tensor):
            out["h"] = t.detach().float().clone()

    h = mods[stage].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(torch.tensor([ids], dtype=torch.long))
    finally:
        h.remove()
    return out["h"][0]  # [T,896]


def donor_hl(donor: Any, ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = donor(torch.tensor([ids], dtype=torch.long), output_hidden_states=True)
    return out.hidden_states[-1][0].detach().float(), out.logits[0].detach().float()


def solve(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def evaluate(
    model: Any,
    donor: Any,
    tok: Any,
    R: torch.Tensor,
    Wu: torch.Tensor,
    stage: str,
    prompts: List[str],
    label: str,
) -> Dict[str, float]:
    W_eff = Wu @ R
    o1_all, o5_all, cos_all = [], [], []
    print(f"\n=== EVAL [{label}] stage={stage} — {len(prompts)} HELD-OUT ===")
    for p in prompts:
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        hc = capture_stage(model, ids, stage)
        hd, dl = donor_hl(donor, ids)
        t = hc.shape[0] - 1
        hhat = hc[t : t + 1] @ R
        cl = (hhat @ W_eff.T)[0]
        dlog = dl[t]
        c1 = set(torch.topk(cl, 1).indices.tolist())
        d1 = set(torch.topk(dlog, 1).indices.tolist())
        c5 = set(torch.topk(cl, 5).indices.tolist())
        d5 = set(torch.topk(dlog, 5).indices.tolist())
        o1, o5 = len(c1 & d1), len(c5 & d5)
        cos = float(F.cosine_similarity(cl.unsqueeze(0), dlog.unsqueeze(0)).item())
        o1_all.append(o1)
        o5_all.append(o5)
        cos_all.append(cos)
        print(
            f"  {p[:40]!r:43} top1={o1} top5={o5} cos={cos:+.4f}  {top_tok(tok, cl, 4)}"
        )
    res = {
        "mean_top1": sum(o1_all) / len(o1_all),
        "mean_top5": sum(o5_all) / len(o5_all),
        "mean_cos": sum(cos_all) / len(cos_all),
    }
    print(
        f"  MEAN top1={res['mean_top1']:.4f} top5={res['mean_top5']:.4f} cos={res['mean_cos']:+.4f}"
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--stage", type=str, default="decomposition")
    ap.add_argument("--lambdas", type=str, default="1e-3,1e-2,1e-1")
    args = ap.parse_args()

    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print(f"STAGE-SOURCE CONSTRUCTION stage={args.stage} (train/holdout, read-only)")
    print("=" * 78)

    plq.CKPT = args.checkpoint.resolve()
    custom, meta = plq.build_real_model()
    for name, param in custom.named_parameters():
        if name.endswith("pos_embed"):
            with torch.no_grad():
                param.data.mul_(PE_RESCALE)
            break
    print(f"custom params={meta['param_count']:,}")

    snaps = glob.glob(
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
        )
    )
    tok = AutoTokenizer.from_pretrained(snaps[0])
    donor = AutoModelForCausalLM.from_pretrained(snaps[0]).eval()
    cache = torch.load(str(DONOR_CACHE), map_location="cpu", weights_only=False)
    Wu = cache["embed_tokens"].float()
    print(f"donor loaded; Wu={tuple(Wu.shape)}")

    Xs, Ys = [], []
    t0 = time.perf_counter()
    for i, p in enumerate(TRAIN):
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        hc = capture_stage(custom, ids, args.stage)
        hd, _ = donor_hl(donor, ids)
        T = min(hc.shape[0], hd.shape[0])
        Xs.append(hc[:T])
        Ys.append(hd[:T])
        if (i + 1) % 10 == 0:
            print(f"  TRAIN {i + 1}/{len(TRAIN)} ({time.perf_counter() - t0:.1f}s)")
    X = torch.cat(Xs, 0)
    Y = torch.cat(Ys, 0)
    print(f"TRAIN pairs X={tuple(X.shape)} Y={tuple(Y.shape)}")

    results = {}
    for lam in [float(x) for x in args.lambdas.split(",")]:
        R = solve(X, Y, lam)
        err = float(((X @ R - Y).norm() / Y.norm()).item())
        print(f"\n### lam={lam} train_rel_err={err:.4f} ###")
        results[lam] = evaluate(
            custom, donor, tok, R, Wu, args.stage, HOLDOUT, f"lam={lam}"
        )

    print("\n" + "=" * 78)
    for lam, r in results.items():
        print(
            f"lam={lam:<8} holdout top1={r['mean_top1']:.4f} top5={r['mean_top5']:.4f} cos={r['mean_cos']:+.4f}"
        )
    print("No files written.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
