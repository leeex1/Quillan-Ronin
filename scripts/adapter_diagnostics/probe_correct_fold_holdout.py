#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CORRECT-FOLD STRICT HOLDOUT TEST — READ-ONLY, IN-MEMORY.

Ridge R: h_c @ R ~= h_d.  Donor logits = h_d @ Wu.T.
CORRECT fold: txt_dec.weight = Wu @ R.T, fed RAW h_c.
(buggy probes used Wu @ R fed h_c @ R -> extra R.T)

Fit R on 40 TRAIN prompts, evaluate on 20 HELD-OUT prompts. No gaming.
"""

from __future__ import annotations

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
CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
PE_RESCALE = 0.01924

from probe_construct_readout_map import TRAIN, HOLDOUT, top_tok  # noqa: E402


def capture(model: Any, ids: List[int], stage: str) -> torch.Tensor:
    mods = {
        "decomposition": model.decomposition,
        "ingestion": model.ingestion,
        "final_norm": model.final_norm,
        "moe": model.moe,
    }
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
    return out["h"][0]


def donor_hl(donor: Any, ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = donor(torch.tensor([ids], dtype=torch.long), output_hidden_states=True)
    return out.hidden_states[-1][0].detach().float(), out.logits[0].detach().float()


def main() -> int:
    import probe_longctx_quality as plq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("CORRECT-FOLD STRICT HOLDOUT (train 40 / holdout 20)")
    print("=" * 78)

    plq.CKPT = CKPT
    custom, meta = plq.build_real_model()
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
    print(f"custom={meta['param_count']:,}; donor loaded; Wu={tuple(Wu.shape)}")

    for stage in ["decomposition", "ingestion"]:
        Xs, Ys = [], []
        t0 = time.perf_counter()
        for i, p in enumerate(TRAIN):
            ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
            hc = capture(custom, ids, stage)
            hd, _ = donor_hl(donor, ids)
            T = min(hc.shape[0], hd.shape[0])
            Xs.append(hc[:T])
            Ys.append(hd[:T])
        X = torch.cat(Xs, 0)
        Y = torch.cat(Ys, 0)
        print(
            f"\n### stage={stage} train pairs={tuple(X.shape)} ({time.perf_counter() - t0:.1f}s) ###"
        )

        for lam in [1e-3, 1e-2, 1e-1]:
            A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
            R = torch.linalg.solve(A, X.T @ Y)
            W_correct = Wu @ R.T  # CORRECT fold
            o1_all, o5_all, cos_all = [], [], []
            for p in HOLDOUT:
                ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
                hc = capture(custom, ids, stage)
                _, dl = donor_hl(donor, ids)
                t = hc.shape[0] - 1
                with torch.no_grad():
                    cl = (hc[t : t + 1] @ W_correct.T)[0].float()
                d = dl[t]
                o1 = int(
                    bool(
                        set(torch.topk(cl, 1).indices.tolist())
                        & set(torch.topk(d, 1).indices.tolist())
                    )
                )
                o5 = len(
                    set(torch.topk(cl, 5).indices.tolist())
                    & set(torch.topk(d, 5).indices.tolist())
                )
                o1_all.append(o1)
                o5_all.append(o5)
                cos_all.append(
                    float(F.cosine_similarity(cl.unsqueeze(0), d.unsqueeze(0)).item())
                )
            print(
                f"  lam={lam:<7} HOLDOUT mean top1={sum(o1_all) / len(o1_all):.4f} "
                f"top5={sum(o5_all) / len(o5_all):.4f} cos={sum(cos_all) / len(cos_all):+.4f} "
                f"(nonzero t1={sum(o1_all):.0f}/{len(o1_all)}, t5>0={sum(1 for x in o5_all if x > 0):.0f}/{len(o5_all)})"
            )
    print("\n" + "=" * 78)
    print("DONE — no files written")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
