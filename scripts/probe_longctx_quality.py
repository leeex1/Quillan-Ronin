#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FORGE v2 — CPU-only LONG-CONTEXT QUALITY probe for the repaired generate() path.

NEW FILE. Does not edit the model, the arch test, or any live config.

WHAT THIS MEASURES (and what it does not)
-----------------------------------------
The repaired path in quillan_v8_saturated.py (#881) is mechanically correct:
chunked KV prefill, real incremental cache, NTK-aware sinusoidal PE beyond the
learned 1024-token table. That is CAPACITY, not QUALITY. Positions >=1024 were
never trained. This probe feeds the REAL trained checkpoint
(exports/quillan-mini-sovereign-bitnet158.safetensors, 815,396,270 params)
a deterministic single-token copy task at controlled absolute depths and asks:

    does retrieval accuracy collapse once the fact sits beyond position 1024?

Design properties that prevent a false positive / false negative:
  * POSITIVE CONTROL FIRST: the same task is run at depth ~64 inside a short
    prompt. If the model cannot retrieve there, the task is not a valid quality
    instrument and the script refuses to draw any long-context conclusion.
  * The fact is a SINGLE tokenizer token chosen at runtime from a candidate
    list, so "rank of the answer token in the prefill logits" is well defined.
  * Scoring uses BOTH (a) rank/top-1 of that token in the raw last-position
    prefill distribution (format-independent retrieval signal) and (b) exact
    substring presence in the generated continuation (behavioural signal).
  * Every inference runs inside torch.no_grad(); caches/tensors are deleted and
    gc.collect() is called between trials (FORGE Zero-Leak Audit Law).

Usage
-----
  python probe_longctx_quality.py --calibrate
  python probe_longctx_quality.py --full --max-seconds 600
  python probe_longctx_quality.py --full --tiny          # harness self-test only

Results are written to scripts/longctx_quality_probe_results.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(r"C:\02_QUILLAN")
SCRIPT_DIR = ROOT / "scripts"
MODEL_DIR = ROOT / "03 - Training & Model"
CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158.safetensors"
TOK_DIR = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
)
RESULT_PATH = SCRIPT_DIR / "longctx_quality_probe_results.json"

for _p in (str(SCRIPT_DIR), str(MODEL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from quillan_v8_saturated import (  # noqa: E402
    DEFAULT_KV_CACHE_CAPACITY,
    QuillanArchConfig,
    QuillanRoninSovereign,
)

# Real mini checkpoint constructor dims (board #894/#895).
H, F_DIM, V, LAYERS = 896, 4864, 151936, 24
NUM_EXPERTS, NUM_ACTIVE, TOP_K, EGGROLL_RANK = 34, 4, 4, 64

SEED = 896
KEY_CANDIDATES = [
    "zyzzyva",
    "quixotic",
    "xenolith",
    "blorptastic",
    "phlogiston",
    "muon",
    "grimoire",
    "vellum",
    "obsidian",
    "cinnabar",
    "palimpsest",
    "luculent",
    "susurrus",
    "widdershins",
    "nacreous",
]
# Prompt format candidates; the positive control selects the strongest one.
QUESTION_FORMATS = [
    "SECURITY RECALL TEST QUESTION: The magic word is",
    "SECURITY RECALL TEST QUESTION: The magic word is:",
    'SECURITY RECALL TEST QUESTION: The magic word is "',
    "SECURITY RECALL TEST QUESTION: Repeat the magic word:",
]
FACT_TMPL = "SECURITY RECALL TEST. The magic word is {key}. Remember exactly.\n"


def log(msg: str) -> None:
    print(msg, flush=True)


def cleanup(*objs: Any) -> None:
    for o in objs:
        del o
    gc.collect()


def set_capacity(model: QuillanRoninSovereign, cap: int) -> None:
    """QuillanArchConfig is frozen: replace it, do not mutate in place."""
    if cap > model.cfg.kv_cache_capacity:
        model.cfg = dataclasses.replace(model.cfg, kv_cache_capacity=int(cap))


# --------------------------------------------------------------------------- #
# Model / tokenizer loading
# --------------------------------------------------------------------------- #
def load_tokenizer() -> Tuple[Any, str]:
    from transformers import AutoTokenizer

    if not (TOK_DIR / "tokenizer.json").exists():
        raise FileNotFoundError(f"tokenizer.json missing under {TOK_DIR}")
    tok = AutoTokenizer.from_pretrained(str(TOK_DIR), local_files_only=True)
    if int(max(tok.get_vocab().values())) >= V:
        raise ValueError("tokenizer vocab exceeds checkpoint embedding width")
    return tok, "Qwen2.5-0.5B-Instruct (matching 151936-wide embedding)"


def build_real_model() -> Tuple[QuillanRoninSovereign, Dict[str, Any]]:
    import safetensors.torch as st

    if not CKPT.exists():
        raise FileNotFoundError(f"checkpoint missing: {CKPT}")
    t0 = time.perf_counter()
    packed = st.load_file(str(CKPT))
    scales = {
        k[: -len(".bitnet_scale")]: v
        for k, v in packed.items()
        if k.endswith(".bitnet_scale")
    }
    raw = {k: v for k, v in packed.items() if not k.endswith(".bitnet_scale")}
    sd: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if v.dtype == torch.int8:
            if k not in scales:
                raise KeyError(f"int8 tensor without bitnet_scale: {k}")
            sd[k] = v.float() * scales[k].float()
        else:
            sd[k] = v.float()
    del packed, scales, raw
    gc.collect()
    load_s = time.perf_counter() - t0

    cfg = QuillanArchConfig(
        text_only=True,
        multimodal=False,
        hidden_dim=H,
        ffn_dim=F_DIM,
        vocab_size=V,
        num_experts=NUM_EXPERTS,
        num_experts_active=NUM_ACTIVE,
        top_k=TOP_K,
        use_lora=True,
        device="cpu",
        eggroll_rank=EGGROLL_RANK,
        kv_cache_capacity=DEFAULT_KV_CACHE_CAPACITY,
    )
    t0 = time.perf_counter()
    model = QuillanRoninSovereign(cfg).eval()
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    incompatible = model.load_state_dict(sd, strict=True)
    load_state_s = time.perf_counter() - t0
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load mismatch: {incompatible}")
    del sd
    gc.collect()
    meta = {
        "checkpoint": str(CKPT),
        "param_count": sum(p.numel() for p in model.parameters()),
        "safetensors_decode_s": round(load_s, 2),
        "model_build_s": round(build_s, 2),
        "strict_load_s": round(load_state_s, 2),
        "bits_per_token_bytes": model.kv_cache_bytes_per_token(),
    }
    return model, meta


def build_tiny_model() -> Tuple[QuillanRoninSovereign, Dict[str, Any]]:
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
        kv_cache_capacity=DEFAULT_KV_CACHE_CAPACITY,
        prefill_chunk_size=128,
    )
    model = QuillanRoninSovereign(cfg).eval()
    return model, {
        "checkpoint": "RANDOM-INIT TINY (mechanics harness only — NOT a quality subject)",
        "param_count": sum(p.numel() for p in model.parameters()),
        "bits_per_token_bytes": model.kv_cache_bytes_per_token(),
    }


# --------------------------------------------------------------------------- #
# Task construction
# --------------------------------------------------------------------------- #
def choose_single_token_keys(tok: Any, count: int = 3) -> List[str]:
    """Find DISTINCT single-token words absent from the filler corpus.

    Qwen BPE splits most rare English words, so candidate strings are checked
    first and the vocabulary is scanned as a fallback. A key must round-trip
    (decode([id]) -> text -> encode == [id]) or the rank metric is undefined.
    """
    soul = ROOT / "SOUL.md"
    corpus = (
        soul.read_text(encoding="utf-8", errors="replace").lower()
        if soul.exists()
        else ""
    )
    banned = set(
        "magic word remember exactly security recall test question repeat".split()
    )

    keys: List[str] = []
    for cand in KEY_CANDIDATES:
        enc = tok.encode(cand, add_special_tokens=False)
        if (
            len(enc) == 1
            and cand.isalpha()
            and cand not in corpus
            and cand not in banned
        ):
            keys.append(cand)

    if len(keys) < count:
        for tid in range(V):
            text = tok.decode([tid], skip_special_tokens=True)
            if not (text and text.isalpha() and text.islower() and 4 <= len(text) <= 9):
                continue
            if text in corpus or text in banned or text in keys:
                continue
            if tok.encode(text, add_special_tokens=False) == [tid]:
                keys.append(text)
                if len(keys) >= max(count, 8):
                    break

    if len(keys) < count:
        raise RuntimeError(
            f"only {len(keys)} single-token key candidates; need {count}"
        )
    return keys[:count]


def filler_tokens(tok: Any, n: int, seed: int) -> List[int]:
    """Deterministic pseudo-natural filler; real SOUL.md text when available."""
    soul = ROOT / "SOUL.md"
    base_text = (
        soul.read_text(encoding="utf-8", errors="replace")
        if soul.exists()
        else (
            "The sovereign blade holds still inside the storm of thought. "
            "Craft is devotion and truth is load-bearing. "
        )
    )
    base = [int(x) for x in tok.encode(base_text, add_special_tokens=False)]
    if len(base) < 32:
        base = [
            int(x)
            for x in tok.encode(
                "The quick brown fox jumps over the lazy dog. ",
                add_special_tokens=False,
            )
        ]
    rng = random.Random(seed)
    out: List[int] = []
    while len(out) < n:
        start = rng.randrange(0, max(1, len(base) - 64))
        out.extend(base[start : start + 64])
    return out[:n]


def build_prompt(
    filler: Sequence[int],
    fact_ids: Sequence[int],
    q_ids: Sequence[int],
    depth_tokens: int,
    total_tokens: int,
) -> List[int]:
    tail = total_tokens - depth_tokens - len(fact_ids) - len(q_ids)
    if tail < 8:
        raise ValueError("insufficient filler after fact")
    prompt = (
        list(filler[:depth_tokens])
        + list(fact_ids)
        + list(filler[depth_tokens : depth_tokens + tail])
        + list(q_ids)
    )
    return prompt


def score_trial(
    model: QuillanRoninSovereign,
    tok: Any,
    prompt: List[int],
    key_id: int,
    max_tokens: int = 8,
) -> Dict[str, Any]:
    """One prefill + one cached continuation. Returns graded retrieval metrics."""
    total = len(prompt)
    need = total + max_tokens + 4
    # Widen only this probe's active window; declared positions stay 2,000,000.
    set_capacity(model, need + 4096)

    with torch.no_grad():
        # max_tokens=0 -> prefill only; cache.last_logits is the RAW distribution
        # at the final prompt position (no decode penalties applied).
        _, cache = model.generate(prompt, max_tokens=0, temp=0.0, return_cache=True)
        raw = cache.last_logits[0].float().clone()  # (vocab,)
        probs = F.softmax(raw, dim=-1)
        key_logit = float(raw[key_id].item())
        rank = int((raw > key_logit).sum().item()) + 1
        top1_id = int(torch.argmax(raw).item())
        top1_p = float(probs[top1_id].item())
        key_p = float(probs[key_id].item())
        del raw, probs

        # Reuse the same cache: no second prefill, 1-token recompute.
        cont = model.generate(
            prompt,
            max_tokens=max_tokens,
            temp=0.0,
            repetition_penalty=1.0,
            cache=cache,
        )
    text = tok.decode([int(t) for t in cont], skip_special_tokens=True)
    key_str = tok.decode([key_id], skip_special_tokens=True)
    hit = key_str.strip() in text
    del cache
    gc.collect()
    return {
        "prompt_tokens": total,
        "key_token_id": key_id,
        "key_token_rank": rank,
        "key_top1": rank == 1,
        "key_top5": rank <= 5,
        "key_prob": round(key_p, 6),
        "top1_prob": round(top1_p, 6),
        "generated_text": text[:120],
        "generated_ids": [int(t) for t in cont],
        "generation_contains_key": bool(hit),
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(model: QuillanRoninSovereign, tok: Any, key_id: int) -> Dict[str, Any]:
    timings: Dict[str, Any] = {}
    for n in (256, 1024):
        prompt = build_prompt(
            filler_tokens(tok, n + 64, SEED + n),
            tok.encode(FACT_TMPL.format(key="zyzzyva"), add_special_tokens=False),
            tok.encode(QUESTION_FORMATS[0], add_special_tokens=False),
            depth_tokens=64,
            total_tokens=n,
        )
        set_capacity(model, len(prompt) + 16)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_tokens=0, temp=0.0, return_cache=True)
        dt = max(time.perf_counter() - t0, 1e-3)
        timings[str(n)] = {
            "prefill_s": round(dt, 3),
            "tokens_per_sec": round(n / dt, 1),
        }
        log(f"CALIBRATE n={n} prefill_s={dt:.2f} tok/s={n / dt:.1f}")
        cleanup(prompt)
    # Rate from the 1024-token prefill. OPTIMISTIC for longer contexts:
    # generate() auto-shrinks the chunk as needed grows, so projections are
    # lower bounds. Never treated as a measured long-context runtime.
    rate = timings["1024"]["tokens_per_sec"]
    proj = {str(n): round(n / rate, 1) for n in (8192, 32768, 131072)}
    return {
        "prefill_tokens_per_sec": round(rate, 1),
        "timings": timings,
        "projected_prefill_seconds": proj,
        "note": "rate from 1024-token prefill; longer contexts shrink chunk and are slower",
    }


# --------------------------------------------------------------------------- #
# Positive control + matrix
# --------------------------------------------------------------------------- #
def positive_control(
    model: QuillanRoninSovereign,
    tok: Any,
    key: str,
    key_id: int,
) -> Dict[str, Any]:
    """Verify the model can retrieve the fact at depth ~64 before any LC claim."""
    results = []
    for fmt in QUESTION_FORMATS:
        fact_ids = tok.encode(FACT_TMPL.format(key=key), add_special_tokens=False)
        q_ids = tok.encode(fmt, add_special_tokens=False)
        prompt = build_prompt(filler_tokens(tok, 640, SEED), fact_ids, q_ids, 64, 512)
        r = score_trial(model, tok, prompt, key_id)
        results.append(
            {
                "format": fmt,
                **{
                    k: r[k]
                    for k in (
                        "key_token_rank",
                        "key_top1",
                        "key_top5",
                        "key_prob",
                        "generation_contains_key",
                        "generated_text",
                    )
                },
            }
        )
        cleanup(prompt)
    best = max(
        results, key=lambda d: (d["key_top1"], d["key_top5"], -d["key_token_rank"])
    )
    answerable = bool(best["key_top5"] or best["generation_contains_key"])
    log(
        f"POSITIVE CONTROL best_rank={best['key_token_rank']} top1={best['key_top1']} "
        f"top5={best['key_top5']} gen_hit={best['generation_contains_key']} "
        f"answerable={answerable} fmt={best['format']!r}"
    )
    return {"answerable": answerable, "best": best, "all_formats": results}


def run_matrix(
    model: QuillanRoninSovereign,
    tok: Any,
    keys: Sequence[str],
    fmt: str,
    lengths: Sequence[int],
    depth_fracs: Sequence[float],
    trials: int,
    deadline: float,
    calib: Dict[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    interrupted: Optional[str] = None
    for L in lengths:
        for frac in depth_fracs:
            depth = max(8, int(L * frac))
            proj = L / max(1.0, calib["prefill_tokens_per_sec"])
            if time.perf_counter() + proj * trials > deadline:
                interrupted = f"budget exhausted before L={L} depth_frac={frac}"
                log("STOP: " + interrupted)
                break
            for trial in range(trials):
                key = keys[trial % len(keys)]
                key_id = int(tok.encode(key, add_special_tokens=False)[0])
                fact_ids = tok.encode(
                    FACT_TMPL.format(key=key), add_special_tokens=False
                )
                q_ids = tok.encode(fmt, add_special_tokens=False)
                filler = filler_tokens(tok, L + 128, SEED + L + depth + 977 * trial)
                prompt = build_prompt(filler, fact_ids, q_ids, depth, L)
                t0 = time.perf_counter()
                r = score_trial(model, tok, prompt, key_id)
                elapsed = time.perf_counter() - t0
                row = {
                    "context_length": L,
                    "depth_fraction": frac,
                    "fact_absolute_position": depth,
                    "trial": trial,
                    "key": key,
                    "elapsed_sec": round(elapsed, 2),
                    "projected_sec": round(proj, 1),
                    **r,
                }
                rows.append(row)
                log(
                    f"L={L:6d} depth={depth:6d} t={trial} key={key:11s} "
                    f"rank={r['key_token_rank']:6d} top1={int(r['key_top1'])} "
                    f"top5={int(r['key_top5'])} p={r['key_prob']:.3e} "
                    f"gen_hit={int(r['generation_contains_key'])} sec={elapsed:.1f}"
                )
                cleanup(prompt, filler)
        if interrupted:
            break

    # Aggregate by depth band: accuracy as a function of absolute position.
    by_depth: Dict[str, Any] = {}
    for r in rows:
        band = str(r["fact_absolute_position"])
        by_depth.setdefault(band, []).append(
            {
                "rank": r["key_token_rank"],
                "top1": r["key_top1"],
                "top5": r["key_top5"],
                "gen_hit": r["generation_contains_key"],
            }
        )
    summary = {}
    for band, vals in sorted(by_depth.items(), key=lambda kv: int(kv[0])):
        n = len(vals)
        summary[band] = {
            "n": n,
            "top1_acc": round(sum(v["top1"] for v in vals) / n, 3),
            "top5_acc": round(sum(v["top5"] for v in vals) / n, 3),
            "gen_acc": round(sum(v["gen_hit"] for v in vals) / n, 3),
            "mean_rank": round(sum(v["rank"] for v in vals) / n, 1),
        }
    return {
        "rows": rows,
        "summary_by_absolute_depth": summary,
        "interrupted": interrupted,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument(
        "--tiny", action="store_true", help="random-init harness self-test only"
    )
    ap.add_argument("--max-seconds", type=int, default=600)
    ap.add_argument("--lengths", type=int, nargs="*", default=[8192, 32768, 131072])
    ap.add_argument("--depths", type=float, nargs="*", default=[0.05, 0.5, 0.9])
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    if not (args.calibrate or args.full):
        ap.error("choose --calibrate or --full")

    random.seed(SEED)
    torch.manual_seed(SEED)
    result: Dict[str, Any] = {
        "schema": "forge.longctx.quality.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "owner": "forge",
        "request_id": 896,
        "mode": "tiny_harness_selftest" if args.tiny else "real_trained_checkpoint",
    }
    t_start = time.perf_counter()

    if args.tiny:
        model, meta = build_tiny_model()
        import types

        tok = types.SimpleNamespace(
            encode=lambda s, add_special_tokens=False: [ord(c) % 256 for c in s],
            decode=lambda ids, skip_special_tokens=True: "".join(chr(i) for i in ids),
            get_vocab=lambda: {},
        )
        keys = ["@"]
        key, key_id = keys[0], ord(keys[0]) % 256
        result["model_meta"] = meta
        result["claim_scope"] = (
            "MECHANICS ONLY — random weights have no retrieval skill"
        )
    else:
        model, meta = build_real_model()
        tok, tok_desc = load_tokenizer()
        keys = choose_single_token_keys(tok, args.trials)
        key = keys[0]
        key_id = int(tok.encode(key, add_special_tokens=False)[0])
        result["model_meta"] = meta
        result["tokenizer"] = tok_desc
        result["key_tokens"] = [
            {"text": k, "id": int(tok.encode(k, add_special_tokens=False)[0])}
            for k in keys
        ]
        result["chance_top1"] = 1.0 / V
        result["claim_scope"] = "REAL TRAINED CHECKPOINT — quality measurement"

    log(
        f"MODEL {result['mode']} params={meta['param_count']:,} "
        f"kv_bytes/token={meta['bits_per_token_bytes']}"
    )

    calib = calibrate(model, tok, key_id)
    result["calibration"] = calib
    log(f"CALIBRATION {calib}")

    if args.calibrate:
        result["elapsed_sec"] = round(time.perf_counter() - t_start, 2)
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"WROTE {RESULT_PATH}")
        return

    pc = positive_control(model, tok, key, key_id)
    result["positive_control"] = pc
    if not pc["answerable"]:
        result["verdict"] = (
            "INSTRUMENT_INVALID: model cannot retrieve even at depth 64; "
            "no long-context quality conclusion is permitted from this run."
        )
        log(result["verdict"])
    else:
        deadline = t_start + args.max_seconds
        matrix = run_matrix(
            model,
            tok,
            keys,
            pc["best"]["format"],
            args.lengths,
            args.depths,
            args.trials,
            deadline,
            calib,
        )
        result["matrix"] = matrix
        summary = matrix["summary_by_absolute_depth"]
        if summary:
            shallow = [v for k, v in summary.items() if int(k) <= 1024]
            deep = [v for k, v in summary.items() if int(k) > 1024]
            s_acc = (
                sum(v["top5_acc"] for v in shallow) / len(shallow) if shallow else None
            )
            d_acc = sum(v["top5_acc"] for v in deep) / len(deep) if deep else None
            result["depth_comparison"] = {
                "shallow_top5_acc_le_1024": s_acc,
                "deep_top5_acc_gt_1024": d_acc,
                "delta": None
                if (s_acc is None or d_acc is None)
                else round(d_acc - s_acc, 3),
            }
            result["verdict"] = (
                f"MEASURED top-5 retrieval shallow(<=1024)={s_acc} "
                f"deep(>1024)={d_acc}; see summary_by_absolute_depth."
            )
        else:
            result["verdict"] = "NO ROWS COMPLETED within budget; projection only."

    result["elapsed_sec"] = round(time.perf_counter() - t_start, 2)
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"WROTE {RESULT_PATH} elapsed={result['elapsed_sec']}s")


if __name__ == "__main__":
    main()
