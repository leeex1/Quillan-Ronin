#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""PyTorch held-out evaluation harness for the custom-architecture checkpoint.

NEW FILE ONLY. This script does NOT edit server.py, MCP configs, model source,
GGUF artifacts, the Modelfile, or the existing llama.cpp harness
(holdout_baseline_v2.py). It imports that harness verbatim to reuse the exact
contamination gate and strict answer-surface parser.

Target artifact
    C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158.safetensors
    QuillanRoninSovereign @ hidden=896 / ffn=4864 / vocab=151936 / 24 blocks

Data contract (reused, never regenerated or modified)
    scripts/eval_sets/items_v2.jsonl                 160 items (80/80)
    scripts/eval_sets/perplexity_corpus_heldout.txt  fixed PPL corpus
    FROZEN_HELD_OUT split only is evaluated (80 items); CALIBRATION is used
    solely by the imported contamination gate.

HONESTY INVARIANTS
  * The checkpoint embedding is 151936 wide -> Qwen2.5-derived. The matching
    Qwen2.5-0.5B-Instruct tokenizer is used. GPT-2 is NEVER used.
  * If that tokenizer is not available on disk, CE/PPL are reported as
    unavailable (never fabricated from a mismatched vocabulary).
  * Cross-entropy must stay STRICTLY ABOVE 1.0 (overfit floor, board #859).
    A violation is flagged loudly in stdout and in the result JSON.
  * Raw generations (decoded text + token ids) are persisted for audit.
  * No benchmark claim is made by this script; it only measures.

Usage
    python holdout_custom_arch_eval.py --validate-only
    python holdout_custom_arch_eval.py --smoke
    python holdout_custom_arch_eval.py --full [--force]
    python holdout_custom_arch_eval.py --full --force
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(r"C:\02_QUILLAN")
SCRIPT_DIR = ROOT / "scripts"
EVAL_DIR = SCRIPT_DIR / "eval_sets"
EXPORT_DIR = ROOT / "exports"
CORE_MODEL_DIR = ROOT / "03 - Training & Model"
ITEMS_FILE = EVAL_DIR / "items_v2.jsonl"
PPL_CORPUS = EVAL_DIR / "perplexity_corpus_heldout.txt"

CKPT = EXPORT_DIR / "quillan-mini-sovereign-bitnet158.safetensors"
RESULT_PATH = EXPORT_DIR / "holdout_custom_arch_mini.json"

# Donor snapshot that carries the matching Qwen2.5 tokenizer (vocab 151936).
MINI_SNAP = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
)

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CORE_MODEL_DIR))

# Imported reference harness: contamination gate + answer-surface convention.
import holdout_baseline_v2 as base  # noqa: E402

from quillan_v8_saturated import (  # noqa: E402
    DECLARED_MAX_POSITIONS,
    DEFAULT_KV_CACHE_CAPACITY,
    MAX_DIRECT_CAUSAL_QUERIES,
    QuillanArchConfig,
    QuillanRoninSovereign,
)

# --------------------------------------------------------------------------- #
# Frozen constants
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = "holdout_custom_arch_eval.0.1"
MODEL_KEY = "quillan_mini_custom_arch"

# Constructor dims (authoritative: verify_mighty_mouse.py board #894/#895).
H, F_DIM, V, L = 896, 4864, 151936, 24
NUM_EXPERTS = 34
NUM_EXPERTS_ACTIVE = 4
TOP_K = 4
EGGROLL_RANK = 64
EXPECTED_PARAMS = 815_396_270

SEED = base.SEED  # 42
TEMPERATURE = 0.0  # deterministic greedy
REPETITION_PENALTY = 1.0  # no extra penalty; matches GGUF top_k=1 baseline
LOSS_GATE_MIN = base.LOSS_GATE_MIN  # 1.0
PPL_WINDOW = 512  # direct-forward window; < MAX_DIRECT_CAUSAL_QUERIES
DOMAINS = base.DOMAINS
MAX_TOKENS = base.MAX_TOKENS
# The 815M dense-MoE arch costs ~3.5 s per CPU forward call (measured). A smoke
# with the true 16..192-token caps would take ~35 min, so smoke VERIFICATION
# uses a short, explicitly recorded decode cap. Smoke numbers are NOT benchmark
# claims; full mode always uses the unmodified MAX_TOKENS contract.
SMOKE_MAX_TOKENS = 8
FULL_RUNTIME_BUDGET_S = 600  # refuse full battery above ~10 min without --force

# The repaired generate() hardcodes <|endoftext|>=50256 as its stop id while the
# Qwen2.5 tokenizer uses eos=151643. Generation therefore runs to max_tokens;
# raw token ids are still decoded with skip_special_tokens=True. Recorded here
# so the discrepancy is auditable rather than silent.
ARCH_EOS_ID = 50256


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_state_dict_from_safetensors(
    path: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Dequantize the packed int8+scale export into a strict load_state_dict dict."""
    if not path.exists():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    packed = safetensors.torch.load_file(str(path))
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
    meta = {
        "packed_tensor_count": len(packed),
        "int8_tensors": sum(1 for v in raw.values() if v.dtype == torch.int8),
        "bitnet_scale_tensors": len(scales),
        "dequantized_keys": len(sd),
    }
    return sd, meta


def build_model() -> QuillanRoninSovereign:
    cfg = QuillanArchConfig(
        text_only=True,
        multimodal=False,
        hidden_dim=H,
        ffn_dim=F_DIM,
        vocab_size=V,
        num_experts=NUM_EXPERTS,
        num_experts_active=NUM_EXPERTS_ACTIVE,
        top_k=TOP_K,
        use_lora=True,
        device="cpu",
        eggroll_rank=EGGROLL_RANK,
    )
    return QuillanRoninSovereign(cfg).eval()


def load_tokenizer() -> Tuple[Optional[Any], Optional[str], Dict[str, Any]]:
    """Load the Qwen2.5 tokenizer that matches the 151936-wide embedding.

    Returns (tokenizer, error_reason, descriptor). On failure tokenizer is None
    and error_reason is set; PPL/CE must then be reported as unavailable.
    """
    desc: Dict[str, Any] = {
        "expected_vocab_embedding": V,
        "path": str(MINI_SNAP) if MINI_SNAP.exists() else None,
        "family": "Qwen2.5-0.5B-Instruct",
        "gpt2_used": False,
    }
    if not (MINI_SNAP / "tokenizer.json").exists():
        return None, "tokenizer_json_missing_on_disk", desc
    try:
        tok = AutoTokenizer.from_pretrained(str(MINI_SNAP), local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - report, never fabricate
        return None, f"tokenizer_load_failed: {type(exc).__name__}: {exc}", desc
    desc["tokenizer_class"] = type(tok).__name__
    desc["vocab_size"] = int(tok.vocab_size)
    desc["eos_token_id"] = tok.eos_token_id
    desc["eos_token"] = tok.eos_token
    desc["max_token_id"] = max(int(x) for x in tok.get_vocab().values())
    desc["vocab_compatible_with_checkpoint"] = bool(desc["max_token_id"] < V)
    if not desc["vocab_compatible_with_checkpoint"]:
        return None, "tokenizer_vocab_exceeds_checkpoint_embedding", desc
    return tok, None, desc


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_one(
    model: QuillanRoninSovereign,
    tok: Any,
    item: Dict[str, Any],
    token_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """Deterministic generation through the repaired KV-cache path."""
    ids: List[int] = list(tok.encode(item["prompt"], add_special_tokens=False))
    if not ids:
        raise ValueError(f"empty tokenization for {item['id']}")
    contract_cap = MAX_TOKENS[item["answer_format"]]
    used_cap = contract_cap if token_cap is None else min(contract_cap, token_cap)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_tokens=used_cap,
            temp=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
        )
    # return_cache defaults False -> plain list of token ids.
    gen_ids = [int(t) for t in out]
    elapsed = time.perf_counter() - t0
    decoded = tok.decode(gen_ids, skip_special_tokens=True)
    parsed = base.parse_answer(decoded, item)
    return {
        "id": item["id"],
        "domain": item["domain"],
        "split": item["split"],
        "answer_format": item["answer_format"],
        "expected": item["expected"],
        "prompt_token_count": len(ids),
        "contract_max_tokens": contract_cap,
        "used_max_tokens": used_cap,
        "token_capped": bool(used_cap < contract_cap),
        "generated_token_ids": gen_ids,
        "raw_generation": decoded,
        "parsed_answer": parsed,
        "correct": bool(parsed is not None and parsed == item["expected"]),
        "elapsed_sec": round(elapsed, 3),
        "eos_id_note": f"arch_stop_id={ARCH_EOS_ID}; tokenizer_eos={tok.eos_token_id}",
    }


# --------------------------------------------------------------------------- #
# Held-out cross-entropy / PPL
# --------------------------------------------------------------------------- #
def compute_perplexity(
    model: QuillanRoninSovereign, tok: Optional[Any], tok_error: Optional[str]
) -> Dict[str, Any]:
    """Next-token CE/PPL on the fixed corpus with the matching tokenizer.

    Non-overlapping direct-forward windows (< direct causal cap) score every
    token except the first of each window. Window positions restart at 0, which
    is inside the learned 1024-position range -> consistent and bounded.
    """
    result: Dict[str, Any] = {
        "method": "pytorch_next_token_ce",
        "corpus_path": str(PPL_CORPUS),
        "window_tokens": PPL_WINDOW,
        "direct_causal_cap": MAX_DIRECT_CAUSAL_QUERIES,
        "loss_gate_min": LOSS_GATE_MIN,
        "tokenizer_error": tok_error,
        "ce_loss": None,
        "ppl": None,
        "scored_tokens": 0,
        "error": None,
    }
    if tok_error is not None or tok is None:
        result["error"] = (
            "tokenizer_unavailable__ppl_not_computed"
            if tok_error is None
            else tok_error
        )
        result["loss_gate_pass"] = None
        return result
    if not PPL_CORPUS.exists():
        result["error"] = "ppl_corpus_missing"
        result["loss_gate_pass"] = None
        return result

    text = PPL_CORPUS.read_text(encoding="utf-8")
    ids = [int(x) for x in tok.encode(text, add_special_tokens=False)]
    result["corpus_token_count"] = len(ids)
    if len(ids) < PPL_WINDOW // 2:
        result["error"] = "insufficient_corpus_tokens"
        result["loss_gate_pass"] = None
        return result

    device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    scored = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ids), PPL_WINDOW):
            chunk = ids[start : start + PPL_WINDOW]
            if len(chunk) < 2:
                break
            x = torch.tensor([chunk], dtype=torch.long, device=device)
            logits = model(x)  # (1, L, V); eval -> logits only
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[0].float()  # (L, V)
            targets = x[0, 1:]  # next-token targets
            nll = F.cross_entropy(logits[:-1, :], targets, reduction="sum")
            total_nll += float(nll)
            scored += int(targets.numel())
            del x, logits, targets, nll
    elapsed = time.perf_counter() - t0
    if scored == 0:
        result["error"] = "no_tokens_scored"
        result["loss_gate_pass"] = None
        return result
    ce = total_nll / scored
    ppl = math.exp(ce) if ce < 700.0 else float("inf")
    result.update(
        {
            "ce_loss": round(ce, 6),
            "ppl": ppl if math.isfinite(ppl) else None,
            "scored_tokens": scored,
            "elapsed_sec": round(elapsed, 3),
            "ce_source": "pytorch_F.cross_entropy",
            "loss_gate_pass": bool(ce > LOSS_GATE_MIN),
            "floor_violation": bool(ce <= LOSS_GATE_MIN),
        }
    )
    return result


# --------------------------------------------------------------------------- #
# Selection / aggregation
# --------------------------------------------------------------------------- #
def select_smoke(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    for domain in DOMAINS:
        held = [
            it
            for it in items
            if it["domain"] == domain and it["split"] == "FROZEN_HELD_OUT"
        ]
        picked.extend(held[:2])
    return picked


def select_held_out(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [it for it in items if it["split"] == "FROZEN_HELD_OUT"]


def aggregate(result: Dict[str, Any], responses: List[Dict[str, Any]]) -> None:
    per_domain: Dict[str, Dict[str, Any]] = {}
    for domain in DOMAINS:
        subset = [r for r in responses if r["domain"] == domain]
        if subset:
            correct = sum(1 for r in subset if r["correct"])
            per_domain[domain] = {
                "n": len(subset),
                "correct": correct,
                "accuracy": round(correct / len(subset), 4),
            }
    n = len(responses)
    correct = sum(1 for r in responses if r["correct"])
    result["per_domain"] = per_domain
    result["aggregate"] = {
        "held_out_n": n,
        "held_out_correct": correct,
        "held_out_accuracy": round(correct / n, 4) if n else None,
        "avg_generation_sec": round(sum(r["elapsed_sec"] for r in responses) / n, 3)
        if n
        else None,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eval(
    items: List[Dict[str, Any]], smoke: bool, token_cap: Optional[int] = None
) -> Dict[str, Any]:
    torch.manual_seed(SEED)
    print(f"\n=== {MODEL_KEY} :: {CKPT.name} ===")
    digest = base.sha256_file(CKPT)
    print(f"checkpoint_sha256 {digest}")

    t_load = time.perf_counter()
    sd, sd_meta = load_state_dict_from_safetensors(CKPT)
    model = build_model()
    expected_keys = set(model.state_dict().keys())
    missing, unexpected = model.load_state_dict(sd, strict=True)
    del sd
    load_s = time.perf_counter() - t_load
    print(
        f"STRICT LOAD missing={len(missing)} unexpected={len(unexpected)} "
        f"keys={len(expected_keys)} load_sec={load_s:.1f}"
    )
    if len(missing) != 0 or len(unexpected) != 0:
        raise RuntimeError(
            f"strict load failed: missing={len(missing)} unexpected={len(unexpected)}"
        )
    params = sum(p.numel() for p in model.parameters())
    print(f"total_params={params} (expected {EXPECTED_PARAMS})")
    if params != EXPECTED_PARAMS:
        raise RuntimeError(f"param count mismatch: {params} != {EXPECTED_PARAMS}")

    tok, tok_error, tok_desc = load_tokenizer()
    print(
        "TOKENIZER "
        + json.dumps(
            {
                "class": tok_desc.get("tokenizer_class"),
                "vocab": tok_desc.get("vocab_size"),
                "path": tok_desc.get("path"),
                "error": tok_error,
            },
            ensure_ascii=False,
        )
    )
    if tok_error is not None:
        print("WARNING: tokenizer unavailable -> generation cannot proceed honestly")
        raise RuntimeError(f"tokenizer unavailable: {tok_error}")

    resp = [generate_one(model, tok, it, token_cap=token_cap) for it in items]
    for r in resp:
        flag = "OK" if r["correct"] else "MISS"
        print(
            f"  [{flag}] {r['id']:<14} {r['domain']:<22} "
            f"expected={r['expected']!r} parsed={r['parsed_answer']!r} "
            f"gen={r['generated_token_ids']}"
        )

    print("Computing held-out CE/PPL on fixed corpus ...")
    ppl = compute_perplexity(model, tok, tok_error)
    ce = ppl["ce_loss"]
    ppl_txt = f"{ppl['ppl']:.4f}" if ppl["ppl"] is not None else "unavailable"
    ce_txt = f"{ce:.4f}" if ce is not None else "unavailable"
    gate = "PASS" if ppl.get("loss_gate_pass") else "FAIL"
    print(
        f"PPL ce_loss={ce_txt} ppl={ppl_txt} scored_tokens={ppl['scored_tokens']} "
        f"loss_gate(>{LOSS_GATE_MIN})={gate}"
    )
    if ppl.get("floor_violation"):
        print(
            f"*** FLOOR VIOLATION: ce_loss={ce_txt} <= {LOSS_GATE_MIN} "
            f"(overfit gate #859) ***"
        )

    cfg = model.cfg
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_key": MODEL_KEY,
        "model_path": str(CKPT),
        "checkpoint_sha256": digest,
        "artifact_sha256": digest,  # schema alias for scorecard consumers
        "generated_utc_epoch": time.time(),
        "mode": "smoke" if smoke else "full",
        "harness": str(Path(__file__).resolve()),
        "reference_harness": str(SCRIPT_DIR / "holdout_baseline_v2.py"),
        "flags": {
            "seed": SEED,
            "temperature": TEMPERATURE,
            "top_k": 1,
            "repetition_penalty": REPETITION_PENALTY,
            "decoding": "greedy_argmax_via_repaired_kv_cache_generate",
            "ppl_window_tokens": PPL_WINDOW,
            "arch_eos_id": ARCH_EOS_ID,
            "tokenizer_eos_id": tok_desc.get("eos_token_id"),
            "smoke_token_cap": token_cap,
            "smoke_numbers_are_benchmark_eligible": token_cap is None,
        },
        "model_info": {
            "class": "QuillanRoninSovereign",
            "source_file": str(CORE_MODEL_DIR / "quillan_v8_saturated.py"),
            "hidden_dim": H,
            "ffn_dim": F_DIM,
            "vocab_size": V,
            "num_layers": L,
            "num_experts": NUM_EXPERTS,
            "num_experts_active": NUM_EXPERTS_ACTIVE,
            "top_k": TOP_K,
            "eggroll_rank": EGGROLL_RANK,
            "total_params": params,
            "expected_total_params": EXPECTED_PARAMS,
            "strict_missing": len(missing),
            "strict_unexpected": len(unexpected),
            "load_sec": round(load_s, 3),
            **sd_meta,
        },
        "tokenizer": tok_desc,
        "contamination": base.verify_no_contamination(items),
        "context": {
            "declared_max_positions": int(DECLARED_MAX_POSITIONS),
            "bounded_active_window": int(DEFAULT_KV_CACHE_CAPACITY),
            "system_prompt_injected": False,
            "note": (
                "Same scoring surface as holdout_baseline_v2.py: no ~65k-token "
                "system prompt replay; generation uses the repaired incremental "
                "KV cache, PPL uses bounded non-overlapping direct-forward windows."
            ),
            "ppl_position_start": 0,
        },
        "score_contract": {
            "direction": "lower_ce_and_ppl_is_better",
            "loss_gate": f"ce_loss > {LOSS_GATE_MIN}",
            "loss_gate_min": LOSS_GATE_MIN,
            "loss_gate_pass": ppl.get("loss_gate_pass"),
            "benchmark_eligible": token_cap is None,
            "smoke_note": (
                "Smoke run uses a short recorded decode cap for end-to-end "
                "verification only; accuracy is NOT a benchmark result. "
                "CE/PPL always uses the full fixed corpus with the matching "
                "Qwen2.5 tokenizer."
            ),
        },
        "perplexity": ppl,
        "per_item": resp,
    }
    aggregate(result, resp)
    ag = result["aggregate"]
    print(
        f"held_out_acc={ag['held_out_accuracy']} "
        f"({ag['held_out_correct']}/{ag['held_out_n']})"
    )
    return result


def write_result(result: Dict[str, Any]) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"WROTE {RESULT_PATH}")
    return RESULT_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="allow full battery if estimate >10min"
    )
    args = parser.parse_args()

    items = base.load_items(ITEMS_FILE)
    contam = base.verify_no_contamination(items)
    print("CONTAMINATION CHECK:", json.dumps(contam, ensure_ascii=False))
    if args.validate_only:
        print(
            f"VALIDATE ONLY: items={len(items)} "
            f"held_out={sum(1 for i in items if i['split'] == 'FROZEN_HELD_OUT')} "
            f"ckpt_exists={CKPT.exists()}"
        )
        return 0

    smoke = not args.full
    picked = select_smoke(items) if smoke else select_held_out(items)
    print(f"MODE={'smoke' if smoke else 'full'} selected_items={len(picked)}")

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    if not PPL_CORPUS.exists():
        raise FileNotFoundError(PPL_CORPUS)

    # Measured floor: ~3.5 s per CPU forward call on this 815M dense-MoE arch.
    contract_tokens = sum(MAX_TOKENS[i["answer_format"]] for i in picked)
    est_full = int(len(picked) * 3.5 + contract_tokens * 3.5 + 120)
    if args.full:
        print(
            f"ESTIMATED FULL BATTERY RUNTIME: <= {est_full}s ({est_full / 60:.1f} min)"
        )
        if est_full > FULL_RUNTIME_BUDGET_S and not args.force:
            print(
                f"REFUSING full battery (>{FULL_RUNTIME_BUDGET_S}s estimate). "
                "Re-run with --force to override."
            )
            return 2

    token_cap = SMOKE_MAX_TOKENS if smoke else None
    result = run_eval(picked, smoke, token_cap=token_cap)
    write_result(result)

    if smoke:
        avg = result["aggregate"].get("avg_generation_sec") or 1.0
        ppl_sec = result["perplexity"].get("elapsed_sec") or 0.0
        # Separate fixed prefill (~3.5s) from per-token decode using the smoke cap.
        per_tok = max(0.1, (avg - 3.5) / max(1, SMOKE_MAX_TOKENS))
        held = select_held_out(items)
        held_tokens = sum(MAX_TOKENS[i["answer_format"]] for i in held)
        est = len(held) * 3.5 + held_tokens * per_tok + ppl_sec + 30
        print(
            f"MEASURED per_token_sec={per_tok:.2f} smoke_avg_item_sec={avg:.2f} "
            f"held_out_contract_tokens={held_tokens}"
        )
        print(
            f"ESTIMATED FULL BATTERY RUNTIME FROM SMOKE: ~{est:.0f}s "
            f"({est / 60:.1f} min) for 80 held-out items"
        )
        print("Full battery NOT executed (smoke mode).")
    print(f"RESULT PATH: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
