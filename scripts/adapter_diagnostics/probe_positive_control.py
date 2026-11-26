#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POSITIVE CONTROL — READ-ONLY, NO MODEL LOAD.

Decisive test recommended by C6 (#1013) / C25 / C2:
  feed the DONOR's true final hidden into the CUSTOM decoder;
  does it reproduce the DONOR's logits?

Also runs the NEGATIVE CONTROL C34 demanded:
  an EQUAL-NORM RANDOM map from custom hidden to donor hidden —
  if it also yields cos +0.34, the 'representational transfer' claim collapses.

Uses only frozen caches + checkpoint tensors. No files written.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
ACTS = DIAG / "_acts"


def load_decoder_weight() -> torch.Tensor:
    with safe_open(str(CKPT), framework="pt", device="cpu") as f:
        return f.get_tensor("txt_dec.weight").float()


def main() -> int:
    donor = torch.load(
        str(ACTS / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    custom = torch.load(
        str(ACTS / "custom_rescaled_acts.pt"), map_location="cpu", weights_only=False
    )
    W = load_decoder_weight()  # [151936,896]
    prompts = list(donor.keys())
    print("=" * 78)
    print("POSITIVE CONTROL: custom decoder(donor final hidden) -> donor logits?")
    print("=" * 78)

    cos_vals: List[float] = []
    t1 = t5 = 0
    for p in prompts:
        hd = donor[p]["final_hidden"][0].float()  # [T,896]
        dl = donor[p]["logits"][0].float()  # [T,151936]
        with torch.no_grad():
            cl = hd @ W.T  # [T,151936]
        t = hd.shape[0] - 1
        cos = float(F.cosine_similarity(cl[t].unsqueeze(0), dl[t].unsqueeze(0)).item())
        o1 = int(
            set(torch.topk(cl[t], 1).indices.tolist())
            & set(torch.topk(dl[t], 1).indices.tolist())
            != set()
        )
        o5 = len(
            set(torch.topk(cl[t], 5).indices.tolist())
            & set(torch.topk(dl[t], 5).indices.tolist())
        )
        cos_vals.append(cos)
        t1 += o1
        t5 += o5
        print(f"  {p[:42]!r:45} cos={cos:+.4f} top1={o1} top5={o5}")
    print(
        f"  MEAN cos={sum(cos_vals) / len(cos_vals):+.4f}  total top1={t1}/{len(prompts)} top5={t5}/{len(prompts)}"
    )
    print("  INTERPRET: cos~1 AND top1=all -> decoder path faithful (control PASSES)")

    # NEGATIVE CONTROL: equal-norm random map from decomposition hidden -> donor hidden
    print("\n" + "=" * 78)
    print("NEGATIVE CONTROL: equal-norm RANDOM map (does cos +0.34 survive?)")
    print("=" * 78)
    torch.manual_seed(1234)
    # build paired decomposition / donor hidden from caches
    Xs, Ys = [], []
    for p in prompts:
        x = custom[p]["decomposition"][0].float()
        y = donor[p]["final_hidden"][0].float()
        T = min(x.shape[0], y.shape[0])
        Xs.append(x[:T])
        Ys.append(y[:T])
    X = torch.cat(Xs, 0)
    Y = torch.cat(Ys, 0)
    scale = (Y.norm(dim=1).mean() / X.norm(dim=1).mean()).item()
    Rrand = torch.randn(896, 896) * (scale / (896**0.5))
    with torch.no_grad():
        yr = X @ Rrand
    rand_cos = [
        float(F.cosine_similarity(yr[i].unsqueeze(0), Y[i].unsqueeze(0)).item())
        for i in range(X.shape[0])
    ]
    print(
        f"  random-map hidden cos: mean={sum(rand_cos) / len(rand_cos):+.4f} "
        f"min={min(rand_cos):+.4f} max={max(rand_cos):+.4f}"
    )
    print("  compare: #1001 learned-map held-out LOGIT cos +0.29..+0.34")
    print("  if random ~= learned, representational-transfer claim COLLAPSES")
    print("=" * 78)
    print("No files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
