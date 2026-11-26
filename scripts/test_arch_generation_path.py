#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FORGE v2 — Quillan custom-architecture generation-path regression suite.

CPU-only, tiny config. Covers the five required checks:
  (a) short-sequence (<=256) OLD vs NEW logits/tokens equivalence
  (b) >=2M positions addressable without a 2M-D learned table
  (c) cache reuse: second generate() with same prefix reuses cache, identical continuation
  (d) 4096-token prompt: peak intermediate allocation measured and bounded
  (e) no O(L^2) allocation: largest single tensor instrumented and asserted

Run:
  & 'C:\\Users\\Admin\\.local\\bin\\python3.11.exe' \\
      'C:\\02_QUILLAN\\scripts\\test_arch_generation_path.py'
"""

import gc
import sys
import time
import warnings

MODEL_DIR = r"C:\02_QUILLAN\03 - Training & Model"
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from quillan_v8_saturated import (
    DECLARED_MAX_POSITIONS,
    DEFAULT_KV_CACHE_CAPACITY,
    MAX_ATTN_SCORE_BYTES,
    MAX_DIRECT_CAUSAL_QUERIES,
    QuillanArchConfig,
    QuillanKVCache,
    QuillanRoninSovereign,
    _causal_mask_inplace,
)

warnings.filterwarnings("ignore", category=UserWarning)
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += " :: " + detail
    print(line)


def make_model():
    torch.manual_seed(1234)
    cfg = QuillanArchConfig(
        hidden_dim=64,
        ffn_dim=96,
        vocab_size=256,
        num_experts=4,
        num_experts_active=2,
        top_k=2,
        eggroll_rank=8,
        max_seq_len=1024,
        max_position_embeddings=DECLARED_MAX_POSITIONS,
        kv_cache_capacity=8192,
        prefill_chunk_size=128,
    )
    return QuillanRoninSovereign(cfg).eval(), cfg


def legacy_greedy_generate(model, prompt, max_tokens):
    """Verbatim PRE-FIX generation loop: re-feed generated[-256:], no cache."""
    generated = list(prompt)
    device = next(model.parameters()).device
    for _ in range(max_tokens):
        inp = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
        out = model(inp)
        logits = out[0] if isinstance(out, tuple) else out
        curr = logits[:, -1, :].clone()
        if generated:
            curr[0, generated[-1]] -= 50.0
        if len(generated) > len(prompt):
            for tid in set(generated[len(prompt) :][-64:]):
                if curr[0, tid] > 0:
                    curr[0, tid] /= 1.15
                else:
                    curr[0, tid] *= 1.15
        tok = int(torch.argmax(curr, dim=-1).item())
        generated.append(tok)
        if tok == 50256:
            break
    return generated[len(prompt) :]


def test_a_short_seq_equivalence(model):
    torch.manual_seed(7)
    ids = torch.randint(0, 256, (1, 200))
    with torch.no_grad():
        ref = model(ids)
    cache = model.new_cache(200 + 8)
    with torch.no_grad():
        cached = model(ids, past_key_values=cache, use_cache=True, position_start=0)[0]
    d_full = float((ref - cached).abs().max())

    c2 = model.new_cache(200 + 8)
    with torch.no_grad():
        for i in range(200):
            o = model(
                ids[:, i : i + 1], past_key_values=c2, use_cache=True, position_start=i
            )
        d_step = float((o[0][:, -1] - ref[:, -1]).abs().max())

    pe_new = model.ingestion.position_embeddings(0, 200, ids.device, torch.float32)
    pe_old = model.ingestion.pos_embed[:, :200, :]
    d_pe = float((pe_new - pe_old).abs().max())

    q = torch.randn(1, 3, 16, 4)
    s = q @ q.transpose(-2, -1)
    new_masked = _causal_mask_inplace(s.clone(), 0)
    old_masked = s.clone().masked_fill(
        ~torch.tril(torch.ones(16, 16, dtype=torch.bool)), float("-inf")
    )
    d_mask = (
        0.0
        if torch.equal(new_masked, old_masked)
        else float((new_masked - old_masked).abs().max())
    )

    prompt = [int(x) for x in torch.randint(0, 256, (12,)).tolist()]
    old_cont = legacy_greedy_generate(model, prompt, 4)
    new_cont = model.generate(prompt, max_tokens=4, temp=0.01)

    tol = 1e-5
    ok = (
        d_full <= tol
        and d_step <= tol
        and d_pe == 0.0
        and d_mask == 0.0
        and old_cont == new_cont
    )
    check(
        "A short-seq OLD vs NEW equivalence (L<=256)",
        ok,
        f"full_max_abs={d_full:.3e} step_max_abs={d_step:.3e} "
        f"pe_max_abs={d_pe:.1e} mask_max_abs={d_mask:.1e} "
        f"old_tokens={old_cont} new_tokens={new_cont} tol={tol}",
    )
    del ids, cache, c2, ref, cached, o, s, q, new_masked, old_masked
    gc.collect()


def test_b_positions(model, cfg):
    pe = model.ingestion.position_embeddings(
        DECLARED_MAX_POSITIONS - 4, 4, torch.device("cpu"), torch.float32
    )
    shape_ok = tuple(pe.shape) == (1, 4, cfg.hidden_dim) and bool(
        torch.isfinite(pe).all()
    )
    sd = model.state_dict()
    key = "ingestion.pos_embed"
    key_ok = key in sd and tuple(sd[key].shape) == (1, cfg.max_seq_len, cfg.hidden_dim)
    largest_table_dim = max(
        int(v.shape[0]) for v in sd.values() if hasattr(v, "shape") and v.dim() > 0
    )
    no_2m_table = largest_table_dim < DECLARED_MAX_POSITIONS

    p0 = model.ingestion.position_embeddings(0, 1, "cpu", torch.float32)
    p1 = model.ingestion.position_embeddings(1_000_000, 1, "cpu", torch.float32)
    p2 = model.ingestion.position_embeddings(
        DECLARED_MAX_POSITIONS - 1, 1, "cpu", torch.float32
    )
    distinct = (
        float((p0 - p1).abs().max()) > 1e-6 and float((p1 - p2).abs().max()) > 1e-6
    )
    ok = shape_ok and key_ok and no_2m_table and distinct
    check(
        "B >=2M positions, no 2M-D table",
        ok,
        f"declared={DECLARED_MAX_POSITIONS} pe@2M={tuple(pe.shape)} "
        f"pos_embed_key_preserved={key_ok} largest_state_dim={largest_table_dim} "
        f"positions_distinct={distinct}",
    )
    del pe, p0, p1, p2
    gc.collect()


def test_c_cache_reuse(model):
    prompt = [i % 256 for i in range(24)]
    seen = {"tokens": 0}
    original = model.forward

    def counting(*args, **kwargs):
        ids = args[0] if args else kwargs.get("txt")
        seen["tokens"] += int(ids.shape[1])
        return original(*args, **kwargs)

    model.forward = counting
    try:
        cont1, cache = model.generate(
            prompt, max_tokens=3, temp=0.01, return_cache=True
        )
        first = seen["tokens"]
        seen["tokens"] = 0
        cont2, cache2 = model.generate(
            prompt, max_tokens=3, temp=0.01, cache=cache, return_cache=True
        )
        second = seen["tokens"]
    finally:
        model.forward = original

    ok = cont1 == cont2 and cache2 is cache and second < first
    check(
        "C cache reuse identical continuation",
        ok,
        f"tokens_processed_pass1={first} pass2={second} "
        f"cont1={cont1} cont2={cont2} same_cache_object={cache2 is cache}",
    )
    del cache, cache2
    gc.collect()


def _has_large_square(shapes):
    for shp in shapes:
        dims = [d for d in shp if isinstance(d, int)]
        if len([d for d in dims if d >= 2048]) >= 2:
            return True, tuple(shp)
    return False, None


def test_d_e_memory(model):
    prompt = [i % 256 for i in range(4096)]
    t0 = time.time()
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            out = model.generate(prompt, max_tokens=2, temp=0.01, chunk_size=128)
    elapsed = time.time() - t0
    events = prof.events()
    allocs = [int(e.cpu_memory_usage) for e in events if int(e.cpu_memory_usage) > 0]
    largest = max(allocs) if allocs else 0
    total_alloc = sum(allocs)
    largest_op = ""
    for e in events:
        if int(e.cpu_memory_usage) == largest and largest > 0:
            largest_op = e.name
            break

    square_events = []
    for e in events:
        bad, shp = _has_large_square(getattr(e, "input_shapes", ()) or ())
        if bad:
            square_events.append(("in", e.name, shp))
        bad, shp = _has_large_square(getattr(e, "output_shapes", ()) or ())
        if bad:
            square_events.append(("out", e.name, shp))

    bound = MAX_ATTN_SCORE_BYTES  # 256 MiB hard budget on any single score tensor
    ok_d = largest <= bound
    ok_e = len(square_events) == 0
    check(
        "D 4096-token peak intermediate allocation bounded",
        ok_d,
        f"largest_single={largest} B ({largest / 1024**2:.2f} MiB) "
        f"op={largest_op} cumulative_allocated={total_alloc / 1024**2:.2f} MiB "
        f"bound={bound} B ({bound / 1024**2:.0f} MiB) elapsed={elapsed:.1f}s "
        f"continuation={out}",
    )
    check(
        "E no O(L^2) tensor for 4096 tokens",
        ok_e,
        f"max_square_shapes_found={len(square_events)} "
        f"examples={square_events[:3]} "
        f"largest_single_vs_O(L2)_float={largest}/{4096 * 4096 * 4} bytes",
    )
    del prompt, events, allocs
    gc.collect()


def test_f_bounded_window_declared(model, cfg):
    try:
        model.new_cache(DECLARED_MAX_POSITIONS)
        ok = False
        detail = "full 2M cache unexpectedly allocated"
    except ValueError as exc:
        ok = "host RAM" in str(exc) and "NOT truncated" in str(exc)
        detail = str(exc)[:220]
    check("F full-2M KV refused with explicit RAM limitation", ok, detail)

    real_bpt = (2 * 1024 + 2 * 30 * 34) * 4  # default real config, fp32
    print(
        "MEMORY MATH (real 1024-dim config, fp32): "
        f"{real_bpt} B/token ({real_bpt / 1024:.2f} KiB); "
        f"70k prompt = {real_bpt * 70_000 / 1024**2:.1f} MiB; "
        f"2M declared = {real_bpt * DECLARED_MAX_POSITIONS / 1024**3:.2f} GiB "
        f"(does NOT fit ~28 GB host RAM with 2.63 GB weights) -> "
        f"PRODUCTION bounded active window = {DEFAULT_KV_CACHE_CAPACITY} tokens "
        f"({real_bpt * DEFAULT_KV_CACHE_CAPACITY / 1024**3:.2f} GiB); "
        f"this test's tiny-config window = {cfg.kv_cache_capacity} tokens. "
        f"num_ctx/DECLARED_MAX_POSITIONS stays 2,000,000; system prompt NOT truncated."
    )


def main():
    print("FORGE v2 arch generation-path regression suite")
    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"declared_positions={DECLARED_MAX_POSITIONS} "
        f"default_window={DEFAULT_KV_CACHE_CAPACITY} "
        f"direct_cap={MAX_DIRECT_CAUSAL_QUERIES}"
    )
    model, cfg = make_model()
    test_a_short_seq_equivalence(model)
    test_b_positions(model, cfg)
    test_c_cache_reuse(model)
    test_d_e_memory(model)
    test_f_bounded_window_declared(model, cfg)
    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"SUMMARY: {passed}/{len(RESULTS)} checks passed")
    if passed != len(RESULTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
