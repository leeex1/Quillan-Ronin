#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLOSED-FORM READOUT CONSTRUCTION PROBE (board #998) — READ-ONLY, IN-MEMORY.

User's construction idea, executed properly with a TRAIN/HOLDOUT split:

  custom_hidden h_c (896)  --[ridge map R]-->  donor_hidden h_d_hat (896)
  logits = W_u @ h_d_hat,   W_u = Qwen embed_tokens (= tied lm_head)

Fold W_eff = W_u @ R into txt_dec.weight [151936,896] — same shape, gate D intact.

Anti-gaming: R is fit on TRAIN prompts only. All reported numbers are on HELD-OUT
prompts the map never saw. If only gate prompts work, discard as gaming.

NO checkpoint/config/gate/source file is written. All mutations in memory.
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
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DONOR_CACHE = ROOT / "exports" / "donor_cache_v23" / "qwen_0.5b_donors.pt"
PE_RESCALE = 0.01924

GATE_PROMPTS = {
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
}

# Held-out prompts — deliberately varied, none in the gate set.
HOLDOUT = [
    "The largest planet in our solar system is",
    "Water boils at one hundred degrees",
    "She opened the door and walked",
    "The square root of sixteen is",
    "In 1969 humans first landed on the",
    "A cat sits on the warm",
    "The opposite of cold is",
    "Python is a popular programming",
    "He poured the coffee into a",
    "The sun rises in the",
    "Three multiplied by seven equals",
    "The capital of Japan is",
    "My favorite color is deep",
    "The train arrived at the station",
    " Birds fly south in the",
    "The book was written by a famous",
    "To bake bread you need flour and",
    "The president of the United States is",
    "Gravity pulls objects toward the",
    "A triangle has three sides and three",
]

# Train prompts — different from holdout and gate.
TRAIN = [
    "The river flows toward the",
    "Summer is hotter than the",
    "The student studied hard for the",
    "Computers process information using",
    "The chef prepared a delicious",
    "Mount Everest is the tallest",
    "The moon orbits around the",
    "Reading books improves your",
    "The dog chased the small",
    "Electricity flows through a",
    "The garden has many beautiful",
    "Winter brings snow and cold",
    "A good friend is always",
    "The teacher explained the lesson",
    "Ocean water is salty and",
    "The artist painted a colorful",
    "Running every morning keeps you",
    "The bridge crosses the wide",
    "Music can make people feel",
    "The scientist discovered a new",
    "A hammer is used to drive",
    "The customer paid with a",
    "Sunlight provides energy for",
    "The baby slept soundly in the",
    "Climbing the mountain requires",
    "The computer screen displayed a",
    "Fresh bread smells absolutely",
    "The teacher asked a difficult",
    "Rain fell on the green",
    "The car drove along the empty",
    "Learning a language takes time and",
    "The orchestra played a beautiful",
    "A diamond is extremely hard and",
    "The farmer harvested the golden",
    "Coffee tastes better when it is",
    "The explorer mapped the unknown",
    "Children learn by playing and",
    "The moonlight shone on the quiet",
    "A healthy diet includes fruits and",
    "The engine started with a loud",
]


def capture_custom(model: Any, ids: List[int]) -> torch.Tensor:
    out: Dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: Tuple[Any, ...], o: torch.Tensor) -> None:
        out["h"] = o.detach().float().clone()

    h = model.final_norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(torch.tensor([ids], dtype=torch.long))
    finally:
        h.remove()
    return out["h"][0]  # [T,896]


def donor_hidden_and_logits(
    dmodel: Any, ids: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = dmodel(torch.tensor([ids], dtype=torch.long), output_hidden_states=True)
    hs = out.hidden_states[-1][0].detach().float()  # [T,896] layer 24
    lg = out.logits[0].detach().float()  # [T,151936]
    return hs, lg


def build_pairs(
    custom: Any, donor: Any, tok: Any, prompts: List[str], label: str
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    Xs: List[torch.Tensor] = []
    Ys: List[torch.Tensor] = []
    Ls: List[torch.Tensor] = []
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        if len(ids) < 2:
            continue
        hc = capture_custom(custom, ids)
        hd, lg = donor_hidden_and_logits(donor, ids)
        T = min(hc.shape[0], hd.shape[0])
        Xs.append(hc[:T])
        Ys.append(hd[:T])
        Ls.append(lg[:T])
        if (i + 1) % 10 == 0:
            print(
                f"  [{label}] {i + 1}/{len(prompts)} pairs ({time.perf_counter() - t0:.1f}s)"
            )
    X = torch.cat(Xs, dim=0)  # [N,896]
    Y = torch.cat(Ys, dim=0)  # [N,896]
    return X, Y, Ls


def solve_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    # h_c @ R ~= h_d ; R = (X^T X + lam I)^-1 X^T Y
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    B = X.T @ Y
    return torch.linalg.solve(A, B)


def top_tok(tok: Any, lg: torch.Tensor, k: int = 5) -> str:
    ids = torch.topk(lg, k).indices.tolist()
    return " | ".join(
        f"{tok.decode([int(i)], skip_special_tokens=True)!r}" for i in ids
    )


def evaluate(
    R: torch.Tensor,
    Wu: torch.Tensor,
    custom: Any,
    donor: Any,
    tok: Any,
    prompts: List[str],
    label: str,
) -> Dict[str, float]:
    W_eff = Wu @ R  # [151936,896] — this is the tensor we would fold into txt_dec
    o1_all, o5_all, cos_all = [], [], []
    print(f"\n=== EVALUATION [{label}] — {len(prompts)} HELD-OUT prompts ===")
    for p in prompts:
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        hc = capture_custom(custom, ids)
        hd, dl = donor_hidden_and_logits(donor, ids)
        t = hc.shape[0] - 1  # last position = next-token prediction site
        hhat = hc[t : t + 1] @ R  # [1,896]
        cl = (hhat @ W_eff.T)[0]  # [151936]
        dlog = dl[t]  # [151936]
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
            f"  {p[:40]!r:43} top1={o1} top5={o5} cos={cos:+.4f}  custom={top_tok(tok, cl, 4)}"
        )
    res = {
        "mean_top1": sum(o1_all) / len(o1_all),
        "mean_top5": sum(o5_all) / len(o5_all),
        "mean_cos": sum(cos_all) / len(cos_all),
    }
    print(
        f"  MEAN top1={res['mean_top1']:.4f} top5={res['mean_top5']:.4f} cos={res['mean_cos']:+.4f}"
    )
    print(f"  (immutable gate needs mean top1>0 AND mean top5>0)")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--lambdas", type=str, default="1e-3,1e-2,1e-1,1.0")
    args = ap.parse_args()

    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("CLOSED-FORM READOUT CONSTRUCTION (train/holdout, read-only)")
    print("=" * 78)

    # custom model with in-memory PE rescale
    plq.CKPT = args.checkpoint.resolve()
    custom, meta = plq.build_real_model()
    for name, param in custom.named_parameters():
        if name.endswith("pos_embed"):
            with torch.no_grad():
                param.data.mul_(PE_RESCALE)
            print(f"custom PE rescaled: {name}")
            break
    print(f"custom params={meta['param_count']:,}")

    # donor tokenizer + model (same tokenizer as gate)
    snaps = glob.glob(
        os.path.expanduser(
            r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
        )
    )
    if not snaps:
        print("FATAL: Qwen2.5-0.5B snapshot not found")
        return 2
    dpath = snaps[0]
    tok = AutoTokenizer.from_pretrained(dpath)
    donor = AutoModelForCausalLM.from_pretrained(dpath).eval()
    print(f"donor loaded from {dpath}")

    cache = torch.load(str(DONOR_CACHE), map_location="cpu", weights_only=False)
    Wu = cache["embed_tokens"].float()  # [151936,896] tied lm_head
    print(f"donor head Wu={tuple(Wu.shape)}")

    # verify gate tokenizer matches
    gtok, _ = plq.load_tokenizer()
    probe = "The capital of France is"
    assert gtok.encode(probe, add_special_tokens=False) == tok.encode(
        probe, add_special_tokens=False
    ), "GATE TOKENIZER MISMATCH — abort"
    print("tokenizer match confirmed")

    # paired data on TRAIN prompts only
    X, Y, _ = build_pairs(custom, donor, tok, TRAIN, "TRAIN")
    print(f"TRAIN pairs: X={tuple(X.shape)} Y={tuple(Y.shape)}")

    results = {}
    for lam in [float(x) for x in args.lambdas.split(",")]:
        print(f"\n### RIDGE lambda={lam} ###")
        R = solve_ridge(X, Y, lam)
        # sanity on train (fit quality)
        err = float(((X @ R - Y).norm() / Y.norm()).item())
        print(f"  train relative error = {err:.4f}")
        results[lam] = evaluate(R, Wu, custom, donor, tok, HOLDOUT, f"lam={lam}")

    print("\n" + "=" * 78)
    for lam, r in results.items():
        print(
            f"lambda={lam:<8} holdout top1={r['mean_top1']:.4f} top5={r['mean_top5']:.4f} cos={r['mean_cos']:+.4f}"
        )
    best = max(results.items(), key=lambda kv: kv[1]["mean_top5"])
    print(f"BEST by holdout top5: lambda={best[0]} -> {best[1]}")
    print("No files written.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
