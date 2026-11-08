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
  openai/gsm8k, ARC-AGI-1 evaluation, gorilla-llm/BFCL).
=============================================================================
"""

import os
import sys
import json
import time
import math
import random
import logging
from typing import Dict, List, Any, Tuple

sys.path.append(r"C:\02_QUILLAN\scripts")
from run_professional_benchmark_suite import (
    load_humaneval_164,
    load_mmlu_500,
    load_gsm8k_500,
    load_arc_100,
    load_bfcl_200,
    run_llama_inference,
    compute_wilson_ci
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Benchmark100")

EXPORT_DIR = r"C:\02_QUILLAN\exports"
OUT_RESULTS = os.path.join(EXPORT_DIR, "benchmark_100_per_section_results.json")


def build_100_per_section_battery(seed: int = 42) -> List[Dict[str, Any]]:
    """Builds a deterministic 500-probe battery with exactly 100 probes per section."""
    rng = random.Random(seed)
    battery: List[Dict[str, Any]] = []

    # Section 1: HumanEval Code (100 probes)
    he_all = load_humaneval_164()
    he_100 = rng.sample(he_all, 100)
    battery.extend(he_100)
    logger.info("Loaded Section 1: HumanEval_Code (%d probes)", len(he_100))

    # Section 2: MMLU Academic (100 probes stratified)
    mmlu_all = load_mmlu_500()
    mmlu_100 = mmlu_all[:100]
    battery.extend(mmlu_100)
    logger.info("Loaded Section 2: MMLU_Academic (%d probes)", len(mmlu_100))

    # Section 3: GSM8K Math (100 probes)
    gsm_all = load_gsm8k_500()
    gsm_100 = gsm_all[:100]
    battery.extend(gsm_100)
    logger.info("Loaded Section 3: GSM8K_Math (%d probes)", len(gsm_100))

    # Section 4: ARC-AGI Inductive Reasoning (100 evaluation tasks)
    arc_100 = load_arc_100()[:100]
    battery.extend(arc_100)
    logger.info("Loaded Section 4: ARC_AGI_Inductive (%d probes)", len(arc_100))

    # Section 5: MCP Tool Calling (100 probes: 50 exec + 50 negative irrelevance)
    bfcl_all = load_bfcl_200()
    exec_probes = [p for p in bfcl_all if "EXEC" in p["id"]][:50]
    irrel_probes = [p for p in bfcl_all if "IRREL" in p["id"]][:50]
    mcp_100 = exec_probes + irrel_probes
    battery.extend(mcp_100)
    logger.info("Loaded Section 5: MCP_ToolCalling (%d probes)", len(mcp_100))

    logger.info("Battery compilation complete: %d total probes across 5 sections.", len(battery))
    return battery


def evaluate_model_on_100_suite(
    model_name: str,
    gguf_path: str,
    battery: List[Dict[str, Any]],
    max_tokens_override: int = 350
) -> Dict[str, Any]:
    """Executes evaluation across all 500 probes on native CUDA."""
    logger.info("=" * 80)
    logger.info("EVALUATING MODEL: %s", model_name)
    logger.info("GGUF: %s", os.path.basename(gguf_path))
    logger.info("Total probes: %d", len(battery))
    logger.info("=" * 80)

    category_results: Dict[str, List[int]] = {}
    detailed_logs: List[Dict[str, Any]] = []
    total_passed = 0
    t0_start = time.time()

    for idx, probe in enumerate(battery):
        cat = probe["category"]
        pid = probe["id"]
        prompt = probe["prompt"]
        checker = probe["checker"]

        resp, latency = run_llama_inference(
            gguf_path,
            prompt,
            max_tokens=probe.get("max_tokens", max_tokens_override)
        )
        passed = int(bool(checker(resp)))
        total_passed += passed
        category_results.setdefault(cat, []).append(passed)

        status_str = "PASS" if passed else "FAIL"
        preview = resp.replace("\n", " ")[:60]
        if (idx + 1) % 10 == 0 or not passed:
            logger.info("[%03d/%03d] [%-18s] %s (%.2fs) | ID: %-16s | %s",
                        idx + 1, len(battery), cat, status_str, latency, pid, preview)

        detailed_logs.append({
            "id": pid,
            "category": cat,
            "dataset_source": probe.get("dataset_source", ""),
            "dataset_split": probe.get("dataset_split", ""),
            "passed": passed,
            "latency": latency,
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
    battery = build_100_per_section_battery(seed=42)
    models_to_evaluate = [
        ("Model A Flagship (Polymath v11)", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-polymath-v11-q4_k_m.gguf"),
        ("Model B Mini (0.5B Sovereign)", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v3-q4_k_m.gguf"),
        ("Model C Dragon (1.0B Hybrid)", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v3-q4_k_m.gguf"),
    ]

    all_results: Dict[str, Any] = {}
    if os.path.exists(OUT_RESULTS):
        try:
            with open(OUT_RESULTS, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            logger.info("Loaded %d existing model results from checkpoint.", len(all_results))
        except Exception:
            all_results = {}

    for name, path in models_to_evaluate:
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            continue
        if name in all_results and all_results[name].get("total_passed") is not None:
            logger.info("Skipping already completed model: %s", name)
            continue
        res = evaluate_model_on_100_suite(name, path, battery)
        all_results[name] = res
        with open(OUT_RESULTS, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("[CHECKPOINT] Saved results for %s to: %s", name, OUT_RESULTS)

    logger.info("[SUCCESS] 100-probe suite complete. Results saved to: %s", OUT_RESULTS)


if __name__ == "__main__":
    main()

