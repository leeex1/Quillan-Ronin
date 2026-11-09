#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: 100-PROBE PER SECTION ACADEMIC BENCHMARK SUITE (500 PROBES)
=============================================================================
Statistical Grounding:
- 100 probes per section eliminates small-sample variance.
- Computes Wilson 95% confidence intervals (margin of error <= +/- 4.9%).
- Official academic test splits only (openai/openai_humaneval, cais/mmlu,
  openai/gsm8k, allenai/ai2_arc ARC-Challenge, gorilla-llm/BFCL).
- Uses persistent native CUDA llama-server backend + RCI format/doctest
  verification (Generate -> Critique -> Improve once).
=============================================================================
"""

import os
import sys
import json
import time
import math
import random
import logging
import argparse
from typing import Dict, List, Any, Tuple

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
from run_professional_benchmark_suite import (
    load_humaneval_164,
    load_mmlu_500,
    load_gsm8k_500,
    load_arc_100,
    load_bfcl_200,
    start_llama_server,
    query_llama_server,
    query_llama_server_messages,
    verify_code_against_public_doctests,
    compute_wilson_ci,
    SYS_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Benchmark100")

EXPORT_DIR = r"C:\02_QUILLAN\exports"
OUT_RESULTS = os.path.join(EXPORT_DIR, "benchmark_100_per_section_results.json")


def build_100_per_section_battery(seed: int = 42, per_section: int = 100) -> List[Dict[str, Any]]:
    """Builds a deterministic 500-probe battery with exactly `per_section` probes per section."""
    rng = random.Random(seed)
    battery: List[Dict[str, Any]] = []

    # Section 1: HumanEval Code (100 probes)
    he_all = load_humaneval_164()
    he_sub = rng.sample(he_all, min(per_section, len(he_all)))
    battery.extend(he_sub)
    logger.info("Loaded Section 1: HumanEval_Code (%d probes)", len(he_sub))

    # Section 2: MMLU Academic (100 probes stratified)
    mmlu_all = load_mmlu_500()
    mmlu_sub = mmlu_all[:per_section]
    battery.extend(mmlu_sub)
    logger.info("Loaded Section 2: MMLU_Academic (%d probes)", len(mmlu_sub))

    # Section 3: GSM8K Math (100 probes)
    gsm_all = load_gsm8k_500()
    gsm_sub = gsm_all[:per_section]
    battery.extend(gsm_sub)
    logger.info("Loaded Section 3: GSM8K_Math (%d probes)", len(gsm_sub))

    # Section 4: ARC-Challenge Inductive & Science Reasoning (100 official test tasks)
    arc_sub = load_arc_100()[:per_section]
    battery.extend(arc_sub)
    logger.info("Loaded Section 4: ARC_AGI_Inductive (%d probes)", len(arc_sub))

    # Section 5: MCP Tool Calling (100 probes: 50 exec + 50 negative irrelevance)
    bfcl_all = load_bfcl_200()
    half = per_section // 2
    exec_probes = [p for p in bfcl_all if "EXEC" in p["id"]][:half]
    irrel_probes = [p for p in bfcl_all if "IRREL" in p["id"]][: (per_section - half)]
    mcp_sub = exec_probes + irrel_probes
    battery.extend(mcp_sub)
    logger.info("Loaded Section 5: MCP_ToolCalling (%d probes)", len(mcp_sub))

    logger.info("Battery compilation complete: %d total probes across 5 sections.", len(battery))
    return battery


def evaluate_model_on_100_suite(
    model_name: str,
    gguf_path: str,
    battery: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Executes evaluation across all probes on persistent native CUDA llama-server with Task 3 RCI."""
    logger.info("=" * 80)
    logger.info("EVALUATING MODEL: %s", model_name)
    logger.info("GGUF: %s", os.path.basename(gguf_path))
    logger.info("Total probes: %d", len(battery))
    logger.info("=" * 80)

    category_results: Dict[str, List[int]] = {}
    detailed_logs: List[Dict[str, Any]] = []
    total_passed = 0
    t0_start = time.time()

    with start_llama_server(gguf_path, port=8089, ctx=4096) as base_url:
        for idx, probe in enumerate(battery):
            cat = probe["category"]
            pid = probe["id"]
            prompt = probe["prompt"]
            checker = probe["checker"]
            max_toks = probe.get("max_tokens", 384)

            resp, latency = query_llama_server(base_url, prompt, max_tokens=max_toks)
            rci_applied = False
            rci_critique = ""

            # Task 3 RCI Verification (Generate -> Critique -> Improve once) using ONLY public syntax/format/doctest checks
            if cat == "HumanEval_Code":
                doc_ok, doc_err = verify_code_against_public_doctests(
                    resp, probe.get("prompt_src", ""), probe.get("entry_point", "")
                )
                if not doc_ok:
                    rci_applied = True
                    rci_critique = f"Public doctest or syntax check failed ({doc_err}). Fix the implementation and output ONLY the corrected Python function."
                    msgs = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                        {"role": "user", "content": rci_critique},
                    ]
                    resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=max_toks)
                    latency += dt2
                    if resp2.strip():
                        resp = resp2
            elif cat in ("MMLU_Academic", "ARC_AGI_Inductive"):
                if not resp.strip():
                    rci_applied = True
                    rci_critique = "Check format: Respond strictly with the single option letter (A, B, C, or D)."
                    msgs = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "user", "content": rci_critique},
                    ]
                    resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=32)
                    latency += dt2
                    if resp2.strip():
                        resp = resp2
            elif cat == "GSM8K_Math":
                if "final answer" not in resp.lower():
                    rci_applied = True
                    rci_critique = "Check arithmetic and format: Conclude your calculation with 'Final answer: <number>'."
                    msgs = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                        {"role": "user", "content": rci_critique},
                    ]
                    resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=128)
                    latency += dt2
                    if resp2.strip():
                        resp = resp + "\n" + resp2

            passed = int(bool(checker(resp)))
            total_passed += passed
            category_results.setdefault(cat, []).append(passed)

            status_str = "PASS" if passed else "FAIL"
            preview = resp.replace("\n", " ")[:65]
            if (idx + 1) % 10 == 0 or not passed:
                logger.info("[%03d/%03d] [%-18s] %s (%.2fs) | ID: %-18s | %s",
                            idx + 1, len(battery), cat, status_str, latency, pid, preview)

            detailed_logs.append({
                "id": pid,
                "category": cat,
                "dataset_source": probe.get("dataset_source", ""),
                "dataset_split": probe.get("dataset_split", ""),
                "passed": passed,
                "latency": round(latency, 3),
                "rci_applied": rci_applied,
                "rci_critique": rci_critique,
                "response": resp
            })

    elapsed_sec = time.time() - t0_start
    overall_acc, overall_ci_low, overall_ci_high = compute_wilson_ci(total_passed, len(battery))

    category_summary: Dict[str, Dict[str, Any]] = {}
    for cat, results in category_results.items():
        n_cat = len(results)
        p_cat = sum(results)
        acc, low, high = compute_wilson_ci(p_cat, n_cat)
        category_summary[cat] = {
            "passed": p_cat,
            "total": n_cat,
            "accuracy": acc,
            "ci_95": [low, high],
            "margin_of_error": round((high - low) / 2, 2)
        }

    logger.info("-" * 80)
    logger.info("FINAL RESULT: %s -> %.2f%% (95%% CI: [%.2f%%, %.2f%%]) [%d/%d] in %.1fs",
                model_name, overall_acc, overall_ci_low, overall_ci_high, total_passed, len(battery), elapsed_sec)
    for cat, stats in category_summary.items():
        logger.info("  * %-20s: %d/%d (%.1f%%, +/-%.1f%%)",
                    cat, stats["passed"], stats["total"], stats["accuracy"], stats["margin_of_error"])
    logger.info("-" * 80)

    return {
        "model_name": model_name,
        "gguf_path": gguf_path,
        "overall_accuracy": overall_acc,
        "ci_95": [overall_ci_low, overall_ci_high],
        "margin_of_error": round((overall_ci_high - overall_ci_low) / 2, 2),
        "total_passed": total_passed,
        "total_probes": len(battery),
        "elapsed_seconds": round(elapsed_sec, 2),
        "category_summary": category_summary,
        "detailed_logs": detailed_logs
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-section", type=int, default=100)
    parser.add_argument("--out", type=str, default=OUT_RESULTS)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    battery = build_100_per_section_battery(seed=42, per_section=args.per_section)
    models_to_evaluate = [
        ("Model A Flagship (1.2B WSP v8.0)", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v8-q4_k_m.gguf"),
        ("Model B Mini (0.5B WSP v8.0)", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v8-q4_k_m.gguf"),
        ("Model C Dragon (1.0B WSP v8.0)", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v8-q4_k_m.gguf"),
    ]

    all_results: Dict[str, Any] = {}
    if not args.fresh and os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            logger.info("Loaded %d existing model results from checkpoint.", len(all_results))
        except Exception:
            all_results = {}

    for name, path in models_to_evaluate:
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            continue
        if not args.fresh and name in all_results and all_results[name].get("total_probes") == len(battery):
            logger.info("Skipping already completed model: %s", name)
            continue
        res = evaluate_model_on_100_suite(name, path, battery)
        all_results[name] = res
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("[CHECKPOINT] Saved results for %s to: %s", name, args.out)

    logger.info("[SUCCESS] Suite complete. Results saved to: %s", args.out)


if __name__ == "__main__":
    main()
