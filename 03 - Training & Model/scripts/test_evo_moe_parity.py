#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EvoMoE grouped-dispatch parity harness (C14-KAIDO).

Compares _forward_grouped against the original _forward_loop on the SAME
module instance and the SAME routing decision, so any difference is purely the
dispatch mechanism.

The router is stubbed with a fixed weight matrix so both paths receive an
identical top-4 selection -- otherwise sampling would diverge.
"""
import sys, pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evo_moe import EvoMoE  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel_err(a, b):
    return ((a - b).abs().max() / a.abs().max().clamp(min=1e-12)).item()


print("=" * 74)
print("EVOMOE GROUPED-DISPATCH PARITY")
print("=" * 74)

torch.manual_seed(0)
H, E = 256, 34
m = EvoMoE(hidden_dim=H, n_experts=E).eval()

# Deterministic router so both paths route identically.
with torch.no_grad():
    m.router.weight.copy_(torch.randn_like(m.router.weight) * 0.35)

print("\n[1] Balanced / random routing")
for trial, (B, S) in enumerate([(1, 32), (2, 64), (4, 128), (1, 7)]):
    torch.manual_seed(100 + trial)
    x = torch.randn(B, S, H)

    flat_x = x.reshape(-1, H)
    logits = m.router(x)
    gates = torch.softmax(logits, dim=-1)
    tg, ti = gates.topk(4, dim=-1)
    fti, ftg = ti.reshape(-1, 4), tg.reshape(-1, 4)
    ov = torch.zeros_like(flat_x)

    m.vectorized = False
    y_loop = m._forward_loop(flat_x, fti, ftg, ov, B, S, H)
    m.vectorized = True
    ov2 = torch.zeros_like(flat_x)
    y_group = m._forward_grouped(flat_x, fti, ftg, ov2, B, S, H)

    e = rel_err(y_loop, y_group)
    check(f"B={B} S={S} grouped == loop", e < 1e-5, f"max rel err = {e:.3e}")

print("\n[2] Skewed routing (one expert takes almost everything)")
torch.manual_seed(7)
B, S = 2, 64
x = torch.randn(B, S, H)
flat_x = x.reshape(-1, H)
# force a lopsided top-4 so the padding guard must engage
sel = torch.zeros(B, S, 4, dtype=torch.long)
sel[..., 0] = 0
sel[..., 1] = 11
sel[..., 2] = 22
sel[..., 3] = 33
w = torch.rand(B, S, 4)
w = w / w.sum(-1, keepdim=True)
fti, ftg = sel.reshape(-1, 4), w.reshape(-1, 4)

ov1 = torch.zeros_like(flat_x); yl = m._forward_loop(flat_x, fti, ftg, ov1, B, S, H)
ov2 = torch.zeros_like(flat_x); yg = m._forward_grouped(flat_x, fti, ftg, ov2, B, S, H)
check("fully-degenerate routing matches", rel_err(yl, yg) < 1e-5,
      f"max rel err = {rel_err(yl, yg):.3e}")

print("\n[3] Extreme skew -> padding guard falls back per-expert")
# all tokens to expert 3 only (empty groups elsewhere)
sel = torch.full((B, S, 4), 3, dtype=torch.long)
w = torch.full((B, S, 4), 0.25)
fti, ftg = sel.reshape(-1, 4), w.reshape(-1, 4)
ov1 = torch.zeros_like(flat_x); yl = m._forward_loop(flat_x, fti, ftg, ov1, B, S, H)
ov2 = torch.zeros_like(flat_x); yg = m._forward_grouped(flat_x, fti, ftg, ov2, B, S, H)
check("single-hot-expert matches", rel_err(yl, yg) < 1e-5,
      f"max rel err = {rel_err(yl, yg):.3e}")

print("\n[4] Cache invalidation after weight update")
torch.manual_seed(11)
x = torch.randn(1, 40, H)
flat_x = x.reshape(-1, H)
ti = torch.randint(0, E, (40, 4)); tg = torch.rand(40, 4)
m.vectorized = True
ov = torch.zeros_like(flat_x); before = m._forward_grouped(flat_x, ti, tg, ov, 1, 40, H)
with torch.no_grad():
    m.experts[5][0].weight.mul_(1.9)
ov = torch.zeros_like(flat_x); after = m._forward_grouped(flat_x, ti, tg, ov, 1, 40, H)
check("weights picked up without stale cache", not torch.allclose(before, after),
      "output changed after mutation -> cache correctly invalidated")
ov1 = torch.zeros_like(flat_x); yl = m._forward_loop(flat_x, ti, tg, ov1, 1, 40, H)
ov2 = torch.zeros_like(flat_x); yg = m._forward_grouped(flat_x, ti, tg, ov2, 1, 40, H)
check("still matches loop after mutation", rel_err(yl, yg) < 1e-5,
      f"max rel err = {rel_err(yl, yg):.3e}")

print("\n[5] Heterogeneous group ranks intact")
ranks = [m.experts[i][0].weight.shape[0] for i in range(E)]
check("34 experts", len(ranks) == 34, f"{len(ranks)}")
check("ranks are 8/24/48 heterogeneous",
      set(ranks[:11]) == {8} and set(ranks[11:22]) == {24} and set(ranks[22:34]) == {48},
      f"groups={sorted(set(ranks))}")

print("\n" + "=" * 74)
if FAIL:
    print(f"RESULT: {len(FAIL)} FAILURE(S): {FAIL}")
    print("=" * 74)
    sys.exit(1)
print("RESULT: ALL EVOMOE PARITY CHECKS PASSED")
print("=" * 74)
sys.exit(0)