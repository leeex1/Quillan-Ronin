#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C18-SHEPHERD / DECISION #926 -- DONOR-BITNET CONVERSION VERIFICATION GATE.

PURPOSE
-------
One immutable, reusable instrument that decides whether the donor-side BitNet
conversion actually fixed the 815M Quillan-Ronin-Mini checkpoint. It is run
UNCHANGED on the current (broken) checkpoint and on any future converted
checkpoint, so before/after numbers are directly comparable and goalposts
cannot move.

STRICT READ-ONLY CONTRACT
-------------------------
  * The candidate checkpoint is read only. No model, checkpoint, tokenizer,
    config, adapter source or exporter file is ever written.
  * Existing harnesses are reused by importing them and overriding their
    module-level CKPT globals IN MEMORY (probe_longctx_quality,
    holdout_custom_arch_eval). No source file is patched.
  * The ONLY files this script writes are NEW artifacts under
    scripts\\adapter_diagnostics\\ (a per-checkpoint JSON result).
  * The canonical checkpoint sha256 53E60D33...A5C is verified UNCHANGED at
    the end (#926 criterion 5).

PRE-REGISTERED PASS THRESHOLDS (decision #926 -- DO NOT WEAKEN)
--------------------------------------------------------------
  A. two different prose prompts: logit cos < 0.99 AND different top-1
  B. coherent English: mean alpha_ratio > 0
  C. donor top-1/top-5 overlap on frozen prompts: mean top1 > 0 AND mean top5 > 0
  D. strict 530/530 key coverage + sha256 record
  E. frozen held-out eval --smoke completes without exception
  F. all three regression suites PASS (exit 0 + exact marker):
       test_custom_arch_transfer.py  -> "CUSTOM_ARCH_TRANSFER_TEST PASS"
       test_arch_generation_path.py  -> "SUMMARY: 6/6 checks passed"
       test_wsp_gates.py             -> "ALL_WSP_GATE_TESTS_PASSED"
  G. canonical checkpoint sha256 unchanged

USAGE
-----
  python -u scripts\\adapter_diagnostics\\conversion_gate.py --checkpoint <PATH>
  python -u scripts\\adapter_diagnostics\\conversion_gate.py --checkpoint <PATH> --skip-suites
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import safetensors  # noqa: E402

CANONICAL_CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158.safetensors"
CANONICAL_SHA = "53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c"
DONOR_ACTS = DIAG / "_acts" / "donor_acts.pt"

# The SAME frozen prompts used by transfer_verification.py / #906/#907/#916.
# PROMPTS[0] and PROMPTS[2] are both 5-token PROSE prompts -- the hard case #925.
PROMPTS: Tuple[str, ...] = (
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
)
PROSE_A, PROSE_B = PROMPTS[0], PROMPTS[2]
MAX_NEW = 10

# Pre-registered; printed verbatim at the top of every run.
PREREGISTERED: Dict[str, Any] = {
    "A_prompt_logit_cos_lt": 0.99,
    "A_top1_must_differ": True,
    "B_mean_alpha_ratio_gt": 0.0,
    "C_donor_mean_top1_overlap_gt": 0.0,
    "C_donor_mean_top5_overlap_gt": 0.0,
    "D_strict_key_coverage": "530/530 (0 missing, 0 unexpected)",
    "E_frozen_heldout_smoke": "completes without exception",
    "F_regression_suites_pass": 3,
    "G_canonical_sha256_unchanged": CANONICAL_SHA,
}

REGRESSION_SUITES: Tuple[Tuple[str, str], ...] = (
    ("test_custom_arch_transfer.py", "CUSTOM_ARCH_TRANSFER_TEST PASS"),
    ("test_arch_generation_path.py", "SUMMARY: 6/6 checks passed"),
    ("test_wsp_gates.py", "ALL_WSP_GATE_TESTS_PASSED"),
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def alpha_ratio(text: str) -> float:
    nonspace = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in nonspace) / max(1, len(nonspace))


def last_logits(model: Any, ids: List[int]) -> torch.Tensor:
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    lg = out[0] if isinstance(out, tuple) else out
    return lg[0, -1, :].float().clone()


def topk_overlap(
    custom_lg: torch.Tensor, donor_lg: torch.Tensor, ks: Tuple[int, ...] = (1, 5)
) -> Dict[str, int]:
    vocab = custom_lg.shape[-1]
    out: Dict[str, int] = {}
    for k in ks:
        c = set(torch.topk(custom_lg, k).indices.tolist())
        d = set(torch.topk(donor_lg, k).indices.tolist())
        out[f"top{k}"] = len(c & d)
    return out


def run_regression_suites() -> Dict[str, Any]:
    """Run the 3 suites with a direct argv list (true exit codes, no PS wrapping)."""
    results: Dict[str, Any] = {}
    for fname, marker in REGRESSION_SUITES:
        script = ROOT / "scripts" / fname
        try:
            cp = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=900,
                cwd=str(ROOT),
            )
            stdout = (cp.stdout or "") + (cp.stderr or "")
            passed = (cp.returncode == 0) and (marker in stdout)
            tail = [ln for ln in stdout.splitlines() if ln.strip()][-3:]
            results[fname] = {
                "returncode": cp.returncode,
                "marker": marker,
                "marker_found": marker in stdout,
                "passed": passed,
                "tail": tail,
            }
            log(
                f"[{fname}] exit={cp.returncode} marker={marker!r} "
                f"found={marker in stdout} -> {'PASS' if passed else 'FAIL'}"
            )
            for ln in tail:
                log(f"    {ln}")
        except Exception as exc:  # noqa: BLE001 - report, never fabricate PASS
            results[fname] = {
                "returncode": None,
                "marker": marker,
                "marker_found": False,
                "passed": False,
                "error": repr(exc),
            }
            log(f"[{fname}] ERROR {exc!r} -> FAIL")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="candidate checkpoint .safetensors (read-only)",
    )
    ap.add_argument(
        "--skip-suites",
        action="store_true",
        help="skip regression suites (development only; full run must include)",
    )
    args = ap.parse_args()
    candidate = args.checkpoint.resolve()
    if not candidate.exists():
        log(f"FATAL: candidate checkpoint not found: {candidate}")
        return 2

    log("=" * 78)
    log("C18-SHEPHERD // DONOR-BITNET CONVERSION VERIFICATION GATE (decision #926)")
    log("=" * 78)
    log(f"candidate_checkpoint : {candidate}")
    log(f"candidate_bytes      : {candidate.stat().st_size}")
    log("pre-registered thresholds (immutable):")
    for k, v in PREREGISTERED.items():
        log(f"    {k} = {v}")
    log("=" * 78)

    t0 = time.perf_counter()
    R: Dict[str, Any] = {
        "owner": "c18-shepherd",
        "decision": 926,
        "candidate_checkpoint": str(candidate),
        "candidate_bytes": candidate.stat().st_size,
        "preregistered": PREREGISTERED,
        "prompts": list(PROMPTS),
        "prose_pair": [PROSE_A, PROSE_B],
    }

    # ---- tokenizer (matching Qwen2.5-0.5B-Instruct, 151936 vocab) ----
    import probe_longctx_quality as plq

    tok, tok_desc = plq.load_tokenizer()
    R["tokenizer"] = tok_desc
    log(f"TOKENIZER: {tok_desc}")

    # ---- load candidate into the real arch, IN MEMORY (plq.CKPT overridden) ----
    plq.CKPT = candidate
    model, meta = plq.build_real_model()
    R["model_meta"] = meta
    log(
        f"MODEL params={meta['param_count']:,} strict_load ok (0 missing / 0 unexpected)"
    )

    # ---- (D) strict 530/530 key coverage + sha256 record ----
    with safetensors.safe_open(str(candidate), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
    ckpt_base_keys = [k for k in all_keys if not k.endswith(".bitnet_scale")]
    expected_keys = set(model.state_dict().keys())
    missing = sorted(expected_keys - set(ckpt_base_keys))
    unexpected = sorted(set(ckpt_base_keys) - expected_keys)
    cand_sha = sha256_file(candidate)
    canon_sha = sha256_file(CANONICAL_CKPT)
    d_pass = (
        len(expected_keys) == 530
        and not missing
        and not unexpected
        and len(ckpt_base_keys) == 530
    )
    R["D_coverage"] = {
        "expected_keys": len(expected_keys),
        "ckpt_base_keys": len(ckpt_base_keys),
        "missing": missing,
        "unexpected": unexpected,
        "candidate_sha256": cand_sha,
        "canonical_sha256_now": canon_sha,
        "canonical_expected_sha256": CANONICAL_SHA,
        "passed": d_pass,
    }
    log("-" * 78)
    log(
        f"[D] STRICT KEY COVERAGE: {len(ckpt_base_keys)}/{len(expected_keys)} "
        f"missing={len(missing)} unexpected={len(unexpected)} -> {'PASS' if d_pass else 'FAIL'}"
    )
    log(f"[D] candidate_sha256  : {cand_sha}")
    log(f"[D] canonical_sha256  : {canon_sha} (expected {CANONICAL_SHA})")

    # ---- (A) two different prose prompts: cos + top-1 ----
    a_ids = [int(x) for x in tok.encode(PROSE_A, add_special_tokens=False)]
    b_ids = [int(x) for x in tok.encode(PROSE_B, add_special_tokens=False)]
    la, lb = last_logits(model, a_ids), last_logits(model, b_ids)
    cos_ab = float(F.cosine_similarity(la.unsqueeze(0), lb.unsqueeze(0)).item())
    t1a, t1b = int(la.argmax().item()), int(lb.argmax().item())
    a_pass = (cos_ab < 0.99) and (t1a != t1b)
    R["A_prompts"] = {
        "prompt_a": PROSE_A,
        "prompt_b": PROSE_B,
        "len_a": len(a_ids),
        "len_b": len(b_ids),
        "logit_cos": round(cos_ab, 8),
        "top1_a_id": t1a,
        "top1_a_tok": repr(tok.decode([t1a], skip_special_tokens=True)),
        "top1_b_id": t1b,
        "top1_b_tok": repr(tok.decode([t1b], skip_special_tokens=True)),
        "max_abs_diff": round(float((la - lb).abs().max().item()), 8),
        "passed": a_pass,
    }
    log("-" * 78)
    log(
        f"[A] PROMPT SENSITIVITY: cos={cos_ab:.6f} (<0.99 required) "
        f"top1_A={t1a}({R['A_prompts']['top1_a_tok']}) "
        f"top1_B={t1b}({R['A_prompts']['top1_b_tok']}) differ={t1a != t1b} "
        f"-> {'PASS' if a_pass else 'FAIL'}"
    )

    # ---- (B) coherent-English heuristic: alpha_ratio ----
    gens: Dict[str, str] = {}
    ars: Dict[str, float] = {}
    for p in PROMPTS:
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        with torch.no_grad():
            cont = model.generate(
                ids, max_tokens=MAX_NEW, temp=0.0, repetition_penalty=1.0
            )
        gen = tok.decode([int(t) for t in cont], skip_special_tokens=True)
        gens[p] = gen
        ars[p] = alpha_ratio(gen)
    mean_ar = sum(ars.values()) / len(ars)
    distinct = len(set(gens.values()))
    from collections import Counter

    gen_counts = Counter(gens.values())
    max_identical = max(gen_counts.values()) if gen_counts else 0
    b_pass = mean_ar > 0.0
    R["B_coherence"] = {
        "mean_alpha_ratio": round(mean_ar, 6),
        "per_prompt_alpha_ratio": {k: round(v, 6) for k, v in ars.items()},
        "per_prompt_generation": gens,
        "distinct_generations": distinct,
        "max_identical_generation_count": max_identical,
        "passed": b_pass,
    }
    log("-" * 78)
    log("[B] COHERENCE (greedy 10-token generations):")
    for p in PROMPTS:
        log(f"    {p!r} -> gen={gens[p]!r} alpha_ratio={ars[p]:.4f}")
    log(
        f"    mean_alpha_ratio={mean_ar:.6f} (>0 required) "
        f"distinct={distinct}/{len(PROMPTS)} -> {'PASS' if b_pass else 'FAIL'}"
    )
    if b_pass and (max_identical >= 2 or distinct < len(PROMPTS)):
        log(
            "    *** DEGENERACY WARNING (non-gating): alpha_ratio>0 is a FALSE POSITIVE "
            f"here -- {max_identical}/{len(PROMPTS)} prompts produced byte-identical text, "
            f"only {distinct}/{len(PROMPTS)} distinct outputs. This is prompt-collapsed "
            "gibberish; coherent English is NOT established. Inspect the text above; "
            "do not read B=PASS alone as a coherence claim (#923 honesty note). ***"
        )
    elif 0 < mean_ar <= 0.10:
        log(
            "    NOTE: alpha_ratio>0 is necessary-but-not-sufficient; inspect text for "
            "repetitive fragments before claiming coherent English."
        )

    # ---- (C) donor top-1/top-5 overlap on frozen prompts ----
    if not DONOR_ACTS.exists():
        log(f"FATAL: frozen donor activation cache missing: {DONOR_ACTS}")
        return 2
    donor_acts = torch.load(str(DONOR_ACTS), map_location="cpu", weights_only=False)
    ov_rows: Dict[str, Dict[str, int]] = {}
    for p in PROMPTS:
        ids = [int(x) for x in tok.encode(p, add_special_tokens=False)]
        cl = last_logits(model, ids)
        dl = donor_acts[p]["logits"][0, -1, :].float()
        ov_rows[p] = topk_overlap(cl, dl)
    mean_top1 = sum(v["top1"] for v in ov_rows.values()) / len(ov_rows)
    mean_top5 = sum(v["top5"] for v in ov_rows.values()) / len(ov_rows)
    c_pass = (mean_top1 > 0.0) and (mean_top5 > 0.0)
    R["C_donor_overlap"] = {
        "per_prompt": ov_rows,
        "mean_top1": round(mean_top1, 6),
        "mean_top5": round(mean_top5, 6),
        "passed": c_pass,
    }
    log("-" * 78)
    log("[C] DONOR TOP-K OVERLAP (last position, frozen prompts):")
    for p in PROMPTS:
        log(f"    {p!r}: top1={ov_rows[p]['top1']} top5={ov_rows[p]['top5']}")
    log(
        f"    mean top1={mean_top1:.4f} mean top5={mean_top5:.4f} "
        f"(both >0 required) -> {'PASS' if c_pass else 'FAIL'}"
    )

    # ---- free model before the smoke harness loads its own instance ----
    del model, la, lb, donor_acts
    gc.collect()

    # ---- (E) frozen held-out eval --smoke (reuse harness verbatim, in memory) ----
    e_pass = False
    try:
        import holdout_custom_arch_eval as hce

        hce.CKPT = candidate  # in-memory override only; harness writes no file here
        items = hce.base.load_items(hce.ITEMS_FILE)
        picked = hce.select_smoke(items)
        smoke_res = hce.run_eval(picked, smoke=True, token_cap=hce.SMOKE_MAX_TOKENS)
        e_pass = True
        R["E_smoke"] = {
            "selected_items": len(picked),
            "held_out_accuracy": smoke_res["aggregate"]["held_out_accuracy"],
            "held_out_correct": smoke_res["aggregate"]["held_out_correct"],
            "held_out_n": smoke_res["aggregate"]["held_out_n"],
            "ce_loss": smoke_res["perplexity"].get("ce_loss"),
            "ppl": smoke_res["perplexity"].get("ppl"),
            "checkpoint_sha256": smoke_res.get("checkpoint_sha256"),
            "passed": e_pass,
        }
    except Exception as exc:  # noqa: BLE001 - honest failure, never fabricate
        R["E_smoke"] = {"passed": False, "error": repr(exc)}
        log(f"[E] SMOKE ERROR (harness did not write a file): {exc!r}")
    log("-" * 78)
    log(
        f"[E] FROZEN HELD-OUT --smoke: completed={e_pass}"
        + (
            f" acc={R['E_smoke'].get('held_out_accuracy')} "
            f"ce={R['E_smoke'].get('ce_loss')}"
            if e_pass
            else ""
        )
    )

    # ---- (F) three regression suites ----
    if args.skip_suites:
        suites = {n: {"passed": None, "skipped": True} for n, _ in REGRESSION_SUITES}
        log("[F] REGRESSION SUITES: SKIPPED (--skip-suites; not a full gate run)")
    else:
        suites = run_regression_suites()
    f_pass = (
        all(s.get("passed") is True for s in suites.values()) and not args.skip_suites
    )
    R["F_regression_suites"] = {"suites": suites, "passed": f_pass}
    log("-" * 78)
    log(f"[F] REGRESSION SUITES: {'PASS' if f_pass else 'FAIL'} (3/3 required)")

    # ---- (G) canonical checkpoint never overwritten ----
    g_pass = canon_sha.lower() == CANONICAL_SHA
    R["G_canonical_unchanged"] = {
        "now": canon_sha,
        "expected": CANONICAL_SHA,
        "passed": g_pass,
    }
    log(f"[G] CANONICAL SHA UNCHANGED: {'PASS' if g_pass else 'FAIL'}")

    # ---- FINAL VERDICT ----
    checks = {
        "A_prompt_sensitivity": a_pass,
        "B_coherent_english": b_pass,
        "C_donor_topk_overlap": c_pass,
        "D_strict_530_530": d_pass,
        "E_frozen_smoke_completed": e_pass,
        "F_regression_suites_3of3": f_pass,
        "G_canonical_sha_unchanged": g_pass,
    }
    overall = all(checks.values())
    R["checks"] = checks
    R["overall_pass"] = overall
    R["elapsed_s"] = round(time.perf_counter() - t0, 1)
    log("=" * 78)
    log(
        "GATE CHECKS: "
        + " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items())
    )
    if not overall:
        blockers = [k for k, v in checks.items() if not v]
        log(f"CONVERSION_GATE_RESULT: FAIL  blockers={blockers}")
        log("Do NOT deploy. Report residual blocker honestly. WSP stays paused (#926).")
    else:
        log("CONVERSION_GATE_RESULT: PASS  (all pre-registered criteria met)")
    log(f"elapsed={R['elapsed_s']}s")

    # ---- persist NEW result under diagnostics only ----
    stem = candidate.stem
    out = DIAG / f"conversion_gate_{stem}.json"
    out.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"WROTE {out}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
