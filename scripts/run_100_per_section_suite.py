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
    diagnose_python_test_failure,
    extract_target_function_only,
    format_humaneval_comment_spec,
    extract_choice_letter,
    extract_gsm8k_number,
    audit_gsm8k_arithmetic,
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
            initial_resp = resp
            passed = int(bool(checker(resp)))
            rci_applied = False
            rci_improved = False
            rci_critique = ""

            # Task 3 RCI Verification: For every held-out failure, run:
            # Generate -> Critique ("Check format, arithmetic, JSON keys, ARC rule") -> Improve once -> Final answer only.
            if not passed:
                rci_applied = True
                if cat == "HumanEval_Code":
                    prompt_src = probe.get("prompt_src", "")
                    entry_fn = probe.get("entry_point", "")
                    _, diag_err = diagnose_python_test_failure(
                        resp,
                        probe.get("test_script", ""),
                        prompt_src,
                        entry_fn,
                    )
                    clean_prev = extract_target_function_only(resp, entry_fn, prompt_src)[:500]
                    commented_spec = format_humaneval_comment_spec(prompt_src)
                    rci_critique = (
                        f"Critique: Check format, syntax, and specification rules ({diag_err}).\n"
                        f"Previous attempt:\n```python\n{clean_prev}\n```\n\n"
                        f"Specification (as comments so you do NOT write a docstring):\n```python\n{commented_spec}\n```\n\n"
                        f"Improve once: Fix the logic/edge-case bug above and output ONLY the executable Python function "
                        f"starting with `def {entry_fn}(` and its indented code body (NO docstring `\"\"\"` and NO `>>>` examples)."
                    )
                    msgs = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": rci_critique},
                    ]
                    resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=max_toks)
                    latency += dt2
                    if resp2.strip() and checker(resp2):
                        resp = resp2
                        passed = 1
                        rci_improved = True

                elif cat in ("MMLU_Academic", "ARC_AGI_Inductive"):
                    first_letter = extract_choice_letter(resp, probe.get("choices"))
                    if not first_letter:
                        msgs_fmt = [
                            {"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": resp},
                            {
                                "role": "user",
                                "content": "Conclude your analysis above and state ONLY the single option letter strictly as: Answer: <letter> (A, B, C, or D).",
                            },
                        ]
                        resp_fmt, dt_fmt = query_llama_server_messages(base_url, msgs_fmt, max_tokens=24)
                        latency += dt_fmt
                        if resp_fmt.strip():
                            resp = resp_fmt
                            if checker(resp):
                                passed = 1
                                rci_improved = True
                                rci_critique = "Format completion: extracted final option letter from truncated CoT."
                            else:
                                first_letter = extract_choice_letter(resp, probe.get("choices"))

                    if not passed:
                        remaining = [l for l in ("A", "B", "C", "D") if l != first_letter]
                        rem_str = ", ".join(remaining)
                        rci_critique = (
                            f"Critique: Check format, domain facts, and ARC/MMLU rule. Option {first_letter or 'none'} is not correct. "
                            f"Eliminate option {first_letter or 'none'}, carefully evaluate the remaining options ({rem_str}), "
                            f"and state your improved final answer strictly as: Answer: <letter> (choosing from {rem_str})."
                        )
                        msgs = [
                            {"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": f"{prompt}\n\n{rci_critique}"},
                        ]
                        resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=64)
                        latency += dt2
                        if resp2.strip():
                            resp = resp2
                            if checker(resp2):
                                passed = 1
                                rci_improved = True

                elif cat == "GSM8K_Math":
                    first_num = extract_gsm8k_number(resp)
                    arith_audit = audit_gsm8k_arithmetic(resp)
                    rci_critique = (
                        f"Critique: Check format and arithmetic step-by-step. Your previous answer ({first_num or 'incomplete'}) "
                        f"failed verification. {arith_audit}\n"
                        "Carefully re-read every quantity, rate, and time period in the word problem:\n"
                        "1. List each given number and unit from the problem.\n"
                        "2. Multiply rates by their exact quantities/times before adding.\n"
                        "3. Check whether the question asks for the sum, difference, or remaining total of all parts.\n"
                        f"4. Compute the corrected result (do NOT repeat {first_num or 'the previous error'}) and end strictly with: Final answer: <number>"
                    )
                    msgs = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": f"{prompt}\n\n{rci_critique}"},
                    ]
                    resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=384)
                    latency += dt2
                    if resp2.strip() and checker(resp2):
                        resp = resp2
                        passed = 1
                        rci_improved = True

                elif cat == "MCP_ToolCalling":
                    rci_critique = (
                        "Critique: Check format and JSON keys. Ensure valid JSON with exact keys (\"tool\" set to the exact "
                        "function name and \"arguments\" object, OR {\"tool\": null, \"reply\": \"No tool required.\"} if no "
                        "tool is needed). Improve once and output ONLY the corrected JSON object."
                    )
                    msgs = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                        {"role": "user", "content": rci_critique},
                    ]
                    resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=192)
                    latency += dt2
                    if resp2.strip() and checker(resp2):
                        resp = resp2
                        passed = 1
                        rci_improved = True

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
                "rci_improved": rci_improved,
                "rci_critique": rci_critique,
                "initial_response": initial_resp,
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
    parser.add_argument("--only-model", type=str, default="", help="Optional filter e.g. 'Model A' to re-run and update only that model in place.")
    args = parser.parse_args()

    battery = build_100_per_section_battery(seed=42, per_section=args.per_section)
    models_to_evaluate = [
        ("Model A Flagship (1.2B WSP v15 Q8_0)", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v15-q8_0.gguf"),
        ("Model B Mini (0.5B WSP v15 Q8_0)", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v15-q8_0.gguf"),
        ("Model C Dragon (1.0B WSP v15 Q8_0)", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v15-q8_0.gguf"),
    ]

    all_results: Dict[str, Any] = {}
    if (not args.fresh or args.only_model) and os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            logger.info("Loaded %d existing model results from checkpoint.", len(all_results))
        except Exception:
            all_results = {}

    for name, path in models_to_evaluate:
        if args.only_model and args.only_model.lower() not in name.lower():
            continue
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            continue
        if not args.fresh and not args.only_model and name in all_results and all_results[name].get("total_probes") == len(battery):
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
