#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compat contract for the UltrametricCouncilRouter (PR #15 follow-up).

The router adds 10 learnable tensors per layer (backbone, route_heads,
expert_head, tree_scale + 6 buffers). Consequence, locked in by this test:
  - strict=True load of a pre-router checkpoint MUST fail, and ONLY on
    `*.moe.ultrametric_router.*` keys (60 for n_layer=6).
  - strict=False warm-start MUST succeed with exactly those missing keys.
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni


def main():
    torch.manual_seed(0)
    cfg = QuillanOniConfig(n_layer=2, max_seq_len=32, hidden_dim=128, n_head=4)
    model = QuillanRoninOni(cfg)
    # Simulate a pre-router checkpoint: current state minus router keys.
    full = model.state_dict()
    legacy = {k: v for k, v in full.items() if "ultrametric_router" not in k}
    assert len(full) - len(legacy) == 20, \
        f"expected 20 router tensors at n_layer=2, got {len(full) - len(legacy)}"

    try:
        model.load_state_dict(legacy, strict=True)
        raise SystemExit("FAIL: strict load should have raised on router keys")
    except RuntimeError as e:
        missing = [l for l in str(e).splitlines() if "ultrametric_router" in l]
        assert missing, f"strict failure not about router keys: {e!s:.300s}"
        print(f"strict=True correctly refuses ({len(missing)} router-key lines)")

    res = model.load_state_dict(legacy, strict=False)
    assert all("ultrametric_router" in k for k in res.missing_keys), res.missing_keys
    assert not res.unexpected_keys, res.unexpected_keys
    print(f"strict=False warm-start OK, {len(res.missing_keys)} missing (all router)")

    model.eval()
    with torch.no_grad():
        out = model(torch.randint(0, 50257, (1, 16)))
    print("forward OK", tuple(out.shape))


if __name__ == "__main__":
    main()
