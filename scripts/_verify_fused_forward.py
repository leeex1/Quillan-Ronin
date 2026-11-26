#!/usr/bin/env python3
"""Verification for quillan_fused_forward.py (board #1171 deliverable)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")
from quillan_fused_forward import (  # noqa: E402
    EXPECTED_BASE_SHAPES,
    EXPECTED_DTYPE_CODES,
    FusedSocketConfig,
    QuillanFusedSocket,
    build_fused_model,
)

SOCKET = r"C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158-socket-v2.safetensors"
DT = {"I8": torch.int8, "F16": torch.float16, "F32": torch.float32}

print("=== TEST 1: random 530-key stub forward on ids [1,2,3] ===")
t0 = time.time()
stub = {}
for key, shape in EXPECTED_BASE_SHAPES.items():
    dt = DT[EXPECTED_DTYPE_CODES[key]]
    if dt == torch.int8:
        t = torch.zeros(shape, dtype=dt)
    else:
        t = torch.randn(shape, dtype=dt) * 0.02
    stub[key] = t
print(f"stub built: {len(stub)} keys in {time.time() - t0:.1f}s")

cfg = FusedSocketConfig(compute_dtype="float32", use_rope=True, use_sparse_mask=True)
model = build_fused_model(cfg, stub)
print(f"model built: {sum(1 for _ in model.state_dict())} state_dict keys")

ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
with torch.no_grad():
    out = model(ids)
print(f"OUTPUT SHAPE: {tuple(out.shape)}")
assert tuple(out.shape) == (1, 3, 151936), f"bad shape {out.shape}"
assert torch.isfinite(out).all(), "non-finite logits"
print(
    f"TEST 1 PASS: shape [1,3,151936], finite logits, no crash ({time.time() - t0:.1f}s)"
)

print("\n=== TEST 2: state_dict keys vs socket-v2 safe_open base keys ===")
from safetensors import safe_open  # noqa: E402

with safe_open(SOCKET, framework="pt", device="cpu") as f:
    all_keys = list(f.keys())
    base_keys = {k for k in all_keys if not k.endswith(".bitnet_scale")}
model_keys = set(model.state_dict().keys())
print(f"socket total tensors : {len(all_keys)}")
print(f"socket base keys     : {len(base_keys)}")
print(f"model state_dict keys: {len(model_keys)}")
print(
    f"missing (in socket, not model): {sorted(base_keys - model_keys)[:10]} count={len(base_keys - model_keys)}"
)
print(
    f"extra   (in model, not socket): {sorted(model_keys - base_keys)[:10]} count={len(model_keys - base_keys)}"
)
assert model_keys == base_keys, "KEY SET MISMATCH"
assert len(model_keys) == 530
# shape equality across all 530
with safe_open(SOCKET, framework="pt", device="cpu") as f:
    shape_bad = []
    for k in base_keys:
        exp = tuple(f.get_slice(k).get_shape())
        got = tuple(model.state_dict()[k].shape)
        if exp != got:
            shape_bad.append((k, exp, got))
print(f"shape mismatches     : {len(shape_bad)} {shape_bad[:5]}")
assert not shape_bad
print("TEST 2 PASS: exact 530-key set + all shapes match socket-v2 header")

print("\n=== TEST 3 (bonus): real socket-v2 load via load_socket_v2 ===")
t1 = time.time()
from quillan_fused_forward import load_socket_v2  # noqa: E402

real = load_socket_v2(SOCKET, FusedSocketConfig(compute_dtype="float16"))
print(
    f"real socket loaded in {time.time() - t1:.1f}s; state_dict keys={len(real.state_dict())}"
)
assert set(real.state_dict().keys()) == base_keys
ids2 = torch.tensor([[1, 2, 3]], dtype=torch.long)
with torch.no_grad():
    out2 = real(ids2)
print(
    f"REAL WEIGHT OUTPUT SHAPE: {tuple(out2.shape)}; finite={bool(torch.isfinite(out2).all())}"
)
assert tuple(out2.shape) == (1, 3, 151936)
print("TEST 3 PASS")
print("\nALL TESTS PASSED")
