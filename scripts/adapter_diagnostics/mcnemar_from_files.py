#!/usr/bin/env python3
"""McNemar from two saved correctness-bit tensors (no model load)."""

import sys, math, os, torch

tmp = os.environ.get("TEMP", ".")
a = torch.load(os.path.join(tmp, "preds_base.pt"), weights_only=True)
b = torch.load(os.path.join(tmp, "preds_r64.pt"), weights_only=True)
b01 = int(((a == 0) & (b == 1)).sum())  # base wrong -> r64 right
b10 = int(((a == 1) & (b == 0)).sum())  # base right -> r64 wrong
n = b01 + b10
if n > 0:
    k = min(b01, b10)
    p = min(1.0, sum(math.comb(n, i) for i in range(k + 1)) / (2**n) * 2)
else:
    p = 1.0
print(f"base={int(a.sum())}/120 r64={int(b.sum())}/120")
print(f"discordant: base_wrong->r64_right={b01}  base_right->r64_wrong={b10}  n={n}")
print(
    f"McNemar exact two-sided p={p:.6f}  {'PASS p<0.01' if p < 0.01 else ('MARGINAL p<0.05' if p < 0.05 else 'FAIL p>=0.05')}"
)
