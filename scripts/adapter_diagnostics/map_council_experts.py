#!/usr/bin/env python3
"""Map the 34 council MoE experts in the LIVE checkpoints: gate state + per-expert weight energy.
Goal state = 34 experts, dense-pull, all active every token. Measures the gap."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model


def inspect(ckpt):
    m = load_model(ckpt)
    print(f"\n=== {ckpt.name} ===", flush=True)
    moes = [
        (i, l.council_moe) for i, l in enumerate(m.layers) if l.council_moe is not None
    ]
    print(f"moe layers: {[i for i, _ in moes]}", flush=True)
    for li, moe in moes:
        gate = float(moe.gate_alpha.detach())

        # per-expert gamma energy for w1/wgate/w2 (packed ternary -> gamma carries scale)
        def energy(buf):
            e = buf.float().abs().amax(dim=tuple(range(1, buf.dim())))  # [num_experts]
            return e

        w1 = energy(moe.w1_gamma)
        wg = energy(moe.wgate_gamma)
        w2 = energy(moe.w2_gamma)
        per = torch.stack([w1, wg, w2], dim=-1).amax(-1)  # [34]
        live = int((per > 1e-6).sum().item())
        dead = int((per <= 1e-6).sum().item())
        print(
            f"  L{li}: gate_alpha={gate:.4f} gate_effective(tanh)={torch.tanh(torch.tensor(gate)).item():.4f} "
            f"experts_live={live}/34 experts_dead={dead} "
            f"gamma_max={float(per.max()):.3f} gamma_min={float(per.min()):.3f}",
            flush=True,
        )
        dead_idx = [j for j in range(len(per)) if float(per[j]) <= 1e-6]
        if dead_idx:
            print(f"    dead expert indices: {dead_idx}", flush=True)


for n in [
    "quillan-mini-sovereign-bitnet.safetensors",
    "quillan-flagship-sovereign-bitnet.safetensors",
]:
    p = ROOT / "exports" / n
    if p.exists():
        inspect(p)
    else:
        print(f"MISSING {n}")
