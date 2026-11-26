#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FORGE v2 / task #963 -- C0 donor-native attention SOCKET verification.

Asserts the four pre-registered socket properties (no backprop, no WSP change):
  1. mini geometry: 14 query heads x 64 dim, attn_dim == hidden == 896, GQA=2
  2. RoPE actually rotates Q and K (cos(before, after) strictly < 1 - delta)
  3. GQA path (2 KV heads -> 14 heads) produces finite attention + cache
  4. short-sequence forward is bit-deterministic across two independent runs
Plus a legacy-socket no-op guard (hidden 64 / 1024 stay 30-head, RoPE OFF).

Run:
  & 'C:\\Users\\Admin\\.local\\bin\\python3.11.exe' -u \\
      'C:\\02_QUILLAN\\scripts\\adapter_diagnostics\\test_attention_socket.py'
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = ROOT / "03 - Training & Model"
for _p in (str(MODEL_DIR), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quillan_v8_saturated import (  # noqa: E402
    CouilAttention,
    QuillanArchConfig,
    QuillanRoninSovereign,
    resolve_attention_geometry,
)

RESULTS: list[tuple[str, bool]] = []
MINI_H = 896
ROPE_COS_DELTA = 5e-3  # cos(before, after) must be < 1 - this


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


def _mini_socket() -> CouilAttention:
    torch.manual_seed(963)
    attn = CouilAttention(MINI_H).eval()
    assert attn.heads == 14 and attn.head_dim == 64 and attn.attn_dim == 896
    return attn


def test_geometry() -> None:
    a = _mini_socket()
    ok = (
        a.heads == 14
        and a.head_dim == 64
        and a.attn_dim == 896
        and a.attn_dim == a.hidden_dim
        and a.kv_heads == 2
        and a.use_rope is True
        and tuple(a.q_proj.weight.shape) == (896, 896)
        and tuple(a.k_proj.weight.shape) == (128, 896)
        and tuple(a.v_proj.weight.shape) == (128, 896)
        and tuple(a.o_proj.weight.shape) == (896, 896)
        and resolve_attention_geometry(MINI_H) == (14, 64, 2, True)
    )
    check(
        "1 mini geometry 14x64 / attn_dim=896 / GQA=2",
        ok,
        f"heads={a.heads} head_dim={a.head_dim} attn_dim={a.attn_dim} "
        f"kv_heads={a.kv_heads} rope={a.use_rope} "
        f"q={tuple(a.q_proj.weight.shape)} k={tuple(a.k_proj.weight.shape)} "
        f"o={tuple(a.o_proj.weight.shape)} keys={len(a.state_dict())}",
    )
    # Full model built to the mini config must inherit the same socket.
    cfg = QuillanArchConfig(
        hidden_dim=896,
        ffn_dim=4864,
        vocab_size=151936,
        num_experts=34,
        eggroll_rank=64,
    )
    m = QuillanRoninSovereign(cfg).eval()
    ma = m.diffusion_core.couil_attn
    ok_model = (ma.heads, ma.head_dim, ma.attn_dim, ma.kv_heads, ma.use_rope) == (
        14,
        64,
        896,
        2,
        True,
    ) and len(m.state_dict()) == 530
    check(
        "1b mini QuillanRoninSovereign inherits socket (530 keys unchanged)",
        ok_model,
        f"heads={ma.heads} head_dim={ma.head_dim} attn_dim={ma.attn_dim} "
        f"kv_heads={ma.kv_heads} rope={ma.use_rope} keys={len(m.state_dict())}",
    )
    del m, ma, a
    gc.collect()


def test_legacy_noop() -> None:
    ok = True
    for h, expected_dim in ((64, 2), (1024, 34)):
        a = CouilAttention(h).eval()
        ok = ok and (
            a.heads == 30
            and a.head_dim == expected_dim
            and a.attn_dim == 30 * expected_dim
            and a.kv_heads == 30
            and a.use_rope is False
            and tuple(a.k_proj.weight.shape) == (a.attn_dim, h)
        )
        del a
    check(
        "0 legacy socket unchanged (30 heads, head_dim=hidden//30, RoPE off)",
        ok,
        "hidden=64->30x2, hidden=1024->30x34, kv_heads=heads, no RoPE",
    )
    gc.collect()


def test_rope_changes_qk() -> None:
    a = _mini_socket()
    B, H, L, D = 1, a.heads, 9, a.head_dim
    with torch.no_grad():
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, a.kv_heads, L, D).repeat_interleave(7, dim=1)
        q0, k0 = q.clone(), k.clone()
        q1, k1 = a.rope_qk(q, k, position_start=0)
        cos_q = float(F.cosine_similarity(q0.flatten(), q1.flatten(), dim=0).item())
        cos_k = float(F.cosine_similarity(k0.flatten(), k1.flatten(), dim=0).item())
        # Position sensitivity: shifting the start must move the rotation.
        q2, _ = a.rope_qk(q0, k0, position_start=17)
        cos_shift = float(F.cosine_similarity(q1.flatten(), q2.flatten(), dim=0).item())
        # External freqs are honoured (already-accepted forward params).
        fc, fs = a.rope_freqs(0, L, q.device, q.dtype)
        q3, _ = a.rope_qk(q0, k0, position_start=0, freqs_cos=fc, freqs_sin=fs)
        cos_external = float(
            F.cosine_similarity(q1.flatten(), q3.flatten(), dim=0).item()
        )
    ok = (
        cos_q < 1.0 - ROPE_COS_DELTA
        and cos_k < 1.0 - ROPE_COS_DELTA
        and cos_shift < 1.0 - ROPE_COS_DELTA
        and cos_external > 1.0 - 1e-5
        and bool(torch.isfinite(q1).all() and torch.isfinite(k1).all())
    )
    check(
        "2 RoPE rotates Q/K and is position + external-freqs sensitive",
        ok,
        f"cos_q={cos_q:.6f} cos_k={cos_k:.6f} cos_shift={cos_shift:.6f} "
        f"cos_external_match={cos_external:.8f} delta={ROPE_COS_DELTA}",
    )
    del a, q, k, q0, k0, q1, k1, q2, q3, fc, fs
    gc.collect()


def test_gqa_finite_attention() -> None:
    a = _mini_socket()
    B, L = 1, 12
    with torch.no_grad():
        x = torch.randn(B, L, MINI_H)
        out, present = a(x, past_key_value=None, use_cache=True, position_start=0)
        # 2 KV heads projected -> expanded to 14 heads in cache/attention.
        q, k, v = a.project(a.norm(x))
        q2, k2 = a.rope_qk(q, k, position_start=0)
        scores = (q2 @ k2.transpose(-2, -1)) * (a.head_dim**-0.5)
        finite = (
            bool(torch.isfinite(out).all())
            and bool(torch.isfinite(q).all())
            and bool(torch.isfinite(k).all())
            and bool(torch.isfinite(v).all())
            and bool(torch.isfinite(scores).all())
            and present is not None
            and tuple(present[0].shape) == (B, 14, L, 64)
            and tuple(present[1].shape) == (B, 14, L, 64)
            and tuple(out.shape) == (B, L, MINI_H)
            and k.shape[1] == 14
            and a.k_proj.out_features == 2 * 64
        )
    check(
        "3 GQA 2 KV heads -> 14 heads, finite attention + KV cache",
        finite,
        f"out={tuple(out.shape)} k_cache={tuple(present[0].shape)} "
        f"k_proj_out={a.k_proj.out_features} finite=True",
    )
    del a, x, out, present, q, k, v, q2, k2, scores
    gc.collect()


def test_short_seq_deterministic() -> None:
    a = _mini_socket()
    with torch.no_grad():
        x = torch.randn(1, 16, MINI_H)
        outs = [a(x, position_start=0)[0].clone() for _ in range(2)]
        identical = bool(torch.equal(outs[0], outs[1]))
        cache = None
        # Full forward vs incremental cache must agree for the socket path.
        cfg = QuillanArchConfig(
            hidden_dim=896,
            ffn_dim=4864,
            vocab_size=151936,
            num_experts=34,
            eggroll_rank=64,
        )
        del a
        gc.collect()
        m = QuillanRoninSovereign(cfg).eval()
        ids = torch.randint(0, 151936, (1, 8))
        full = m(ids)[0]
        cache = m.new_cache(8 + 4)
        inc = m(ids, past_key_values=cache, use_cache=True, position_start=0)[0]
        max_abs = float((full - inc).abs().max().item())
        ok = identical and max_abs <= 1e-5 and bool(torch.isfinite(full).all())
    check(
        "4 short-seq forward deterministic + cached==uncached",
        ok,
        f"bit_identical={identical} full_vs_cached_max_abs={max_abs:.3e}",
    )
    del outs, full, inc, cache, ids, x, m
    gc.collect()


def main() -> None:
    print("FORGE v2 task #963 -- C0 attention socket verification")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    test_legacy_noop()
    test_geometry()
    test_rope_changes_qk()
    test_gqa_finite_attention()
    test_short_seq_deterministic()
    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"SUMMARY: {passed}/{len(RESULTS)} socket checks passed")
    if passed != len(RESULTS):
        sys.exit(1)
    print("ATTENTION_SOCKET_TEST PASS")


if __name__ == "__main__":
    main()
