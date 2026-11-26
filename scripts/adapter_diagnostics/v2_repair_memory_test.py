#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 REPAIR -- BOUNDED IN-MEMORY EXPERIMENT (spec #919).

NO source edits. NO checkpoint writes. Applies A+B+C to the loaded real model
in memory, then measures the exact gate criteria:
  * prompt sensitivity (two 5-token prompts must produce DIFFERENT logits)
  * output coherence (alpha_ratio > 0, not digits/punctuation/gibberish)
  * donor top-k overlap (must be > 0, not exactly 0)

Usage: python -u scripts/adapter_diagnostics/v2_repair_memory_test.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import quillan_v8_saturated as arch  # noqa: E402
from probe_longctx_quality import build_real_model, load_tokenizer  # noqa: E402

PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
)
POS_SCALE = 0.01924392  # forge #918 exact scale-match (0.01198079 / 0.62257522)
EPS_FIXED = 1e-12  # BitNet 1.58b absmean: eps is a div-guard, NOT a floor
MAX_NEW = 10


def log(msg: str) -> None:
    print(msg, flush=True)


def last_logits(model, ids):
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    lg = out[0] if isinstance(out, tuple) else out
    return lg[0, -1, :].float().clone()


def diff(a, b):
    d = (a - b).abs()
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return (
        f"max={d.max().item():.6e} mean={d.mean().item():.6e} "
        f"pct<1e-4={(d < 1e-4).float().mean().item() * 100:.2f}% cos={cos:.6f}"
    )


def alpha_ratio(text: str) -> float:
    nonspace = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in nonspace) / max(1, len(nonspace))


def topk_overlap(custom_lg, donor_lg, tok, k_values=(1, 5, 10, 20)):
    """Last-position donor top-k overlap; chance = k^2 / vocab."""
    vocab = custom_lg.shape[-1]
    out = {}
    for k in k_values:
        c = set(torch.topk(custom_lg, k).indices.tolist())
        d = set(torch.topk(donor_lg, k).indices.tolist())
        out[f"top{k}"] = (len(c & d), len(c & d) / k, k * k / vocab)
    return out


def main() -> None:
    t0 = time.perf_counter()
    tok, _ = load_tokenizer()
    model, meta = build_real_model()
    log(f"MODEL params={meta['param_count']:,} ckpt={meta['checkpoint']}")

    # ---- Baseline (current checkpoint, current eps) ----
    A = [int(x) for x in tok.encode(PROMPTS[0], add_special_tokens=False)]
    B = [int(x) for x in tok.encode(PROMPTS[2], add_special_tokens=False)]
    la, lb = last_logits(model, A), last_logits(model, B)
    log("BASELINE last-logit A/B: " + diff(la, lb))
    log(f"  top1 A={int(la.argmax())} B={int(lb.argmax())}")
    del la, lb

    # ================= APPLY A + B + C IN MEMORY =================
    # A: BitNet 1.58b absmean quantization -- remove the 0.01 magnitude floor.
    n_bit = 0
    for m in model.modules():
        if isinstance(m, arch.BitLinear):
            m.eps = EPS_FIXED
            m._w_quant_cache = None
            n_bit += 1
    log(f"[A] eps -> {EPS_FIXED} on {n_bit} BitLinear modules; caches cleared")

    # B: scale the fabricated sinusoidal pos_embed to donor txt_emb magnitude.
    with torch.no_grad():
        before = model.ingestion.pos_embed.float().abs().mean().item()
        model.ingestion.pos_embed.mul_(POS_SCALE)
        after = model.ingestion.pos_embed.float().abs().mean().item()
    log(f"[B] pos_embed mean_abs {before:.5f} -> {after:.5f} (x{POS_SCALE})")

    # C: ingestion.norm -> standard LayerNorm gain scale (same as final_norm).
    with torch.no_grad():
        w_before = model.ingestion.norm.weight.float().abs().mean().item()
        ref = model.final_norm.weight.float().abs().mean().item()
        model.ingestion.norm.weight.copy_(model.final_norm.weight)
        w_after = model.ingestion.norm.weight.float().abs().mean().item()
    log(
        f"[C] ingestion.norm mean_abs {w_before:.5f} -> {w_after:.5f} (ref final_norm {ref:.5f})"
    )

    # Proof A worked: q/k/v/o outputs no longer exactly zero.
    with torch.no_grad():
        attn = model.diffusion_core.couil_attn
        wq = attn.q_proj.weight
        wq_abs = wq.float().abs().mean().item()
        wq_q = arch._weight_quant(wq, EPS_FIXED)
        wq_q_abs = wq_q.float().abs().mean().item()
        probe_x = torch.randn(4, wq.shape[1])
        qout = attn.q_proj(probe_x).float().abs().mean().item()
    log(
        f"[A proof] q_proj weight mean_abs={wq_abs:.6e} quant={wq_q_abs:.6e} "
        f"q_proj(x) out_mean_abs={qout:.6e} (baseline was exactly 0)"
    )

    # ---- Gate metric 1: prompt sensitivity ----
    la2, lb2 = last_logits(model, A), last_logits(model, B)
    log("POST-FIX last-logit A/B: " + diff(la2, lb2))
    log(f"  top1 A={int(la2.argmax())} B={int(lb2.argmax())}")
    for tag, lg in (("A", la2), ("B", lb2)):
        v, i = torch.topk(lg, 5)
        log(
            f"  top5({tag}): {[(tok.decode([int(t)], skip_special_tokens=True), round(float(p), 2)) for t, p in zip(i.tolist(), v.tolist())]}"
        )

    # ---- Gate metric 2: coherence ----
    log("=== GENERATION (post-fix, greedy 10 tokens) ===")
    gens = {}
    for text in PROMPTS:
        ids = [int(x) for x in tok.encode(text, add_special_tokens=False)]
        with torch.no_grad():
            cont = model.generate(
                ids, max_tokens=MAX_NEW, temp=0.0, repetition_penalty=1.0
            )
        gen = tok.decode([int(t) for t in cont], skip_special_tokens=True)
        ar = alpha_ratio(gen)
        gens[text] = gen
        log(f"  {text!r} -> gen={gen!r} alpha_ratio={ar:.4f}")
    distinct = len(set(gens.values()))
    mean_ar = sum(alpha_ratio(g) for g in gens.values()) / len(gens)
    log(
        f"COHERENCE distinct_generations={distinct}/{len(gens)} mean_alpha_ratio={mean_ar:.4f}"
    )

    # ---- Gate metric 3: donor top-k overlap ----
    donor_acts = torch.load(
        str(DIAG / "_acts" / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    log("=== DONOR TOP-K OVERLAP (last position) ===")
    overlap_rows = {}
    for text in PROMPTS:
        ids = [int(x) for x in tok.encode(text, add_special_tokens=False)]
        cl = last_logits(model, ids)
        dl = donor_acts[text]["logits"][0, -1, :].float()
        ov = topk_overlap(cl, dl, tok)
        overlap_rows[text] = ov
        log(
            f"  {text!r}: "
            + " ".join(f"{k}={v[0]}/{k} (chance {v[2]:.5f})" for k, v in ov.items())
        )
    mean_top1 = sum(v["top1"][0] for v in overlap_rows.values()) / len(overlap_rows)
    mean_top5 = sum(v["top5"][0] for v in overlap_rows.values()) / len(overlap_rows)
    log(f"DONOR OVERLAP mean top1={mean_top1:.3f} top5={mean_top5:.3f}")

    # ---- FINAL GATE ----
    prompt_sensitive = (la2 - lb2).abs().max().item() > 1e-3
    coherent = mean_ar > 0.10 and distinct >= 2
    overlap_ok = mean_top1 > 0
    log("=" * 64)
    log(
        f"GATE prompt_sensitive={prompt_sensitive} coherent_english={coherent} "
        f"donor_topk_nonzero={overlap_ok}"
    )
    if prompt_sensitive and coherent and overlap_ok:
        log("GATE RESULT: PASS -- re-export of v2 is warranted")
    else:
        blockers = []
        if not prompt_sensitive:
            blockers.append("logits still prompt-blind")
        if not coherent:
            blockers.append(f"degenerate output (mean_alpha_ratio={mean_ar:.4f})")
        if not overlap_ok:
            blockers.append("donor top-k overlap still exactly 0")
        log("GATE RESULT: FAIL -- STRUCTURAL BLOCKER: " + "; ".join(blockers))
        log("Do NOT re-export. Do NOT tune WSP. Escalate to coordinator.")
    log(f"elapsed={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
