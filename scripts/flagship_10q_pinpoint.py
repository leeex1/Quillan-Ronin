"""
Fast 10-Question Pinpoint Diagnostic for Flagship 1.2B
Tests 2 HumanEval + 2 MMLU + 2 GSM8K + 2 ARC-AGI-1 + 2 HLE = 10 questions.
Prints full per-question diagnosis so we can pinpoint weight issues directly.
"""
import os
import sys
import time
import json
import argparse
import logging

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import run_100_per_section_suite as s100
import run_arc_agi_123_and_hle_suite as sarc
from run_professional_benchmark_suite import (
    start_llama_server,
    query_llama_server,
    query_llama_server_messages,
    diagnose_python_test_failure,
    extract_target_function_only,
    format_humaneval_comment_spec,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Flagship_10Q")


def build_10q_battery(seed: int = 42):
    # 2 HumanEval, 2 MMLU, 2 GSM8K from s100
    b100 = s100.build_100_per_section_battery(seed=seed, per_section=2)
    # Keep only HumanEval, MMLU, GSM8K (6 probes)
    core_6 = [p for p in b100 if p["category"] in ("HumanEval_Code", "MMLU_Academic", "GSM8K_Math")]

    # 2 ARC-AGI-1 + 2 HLE from sarc
    arc_2 = sarc.load_arc_agi_1_100(per_section=2, seed=seed)
    hle_2 = sarc.load_hle_100(per_section=2, seed=seed)
    return core_6 + arc_2 + hle_2


def run_pinpoint(gguf_path: str, port: int = 8095):
    battery = build_10q_battery(seed=42)
    logger.info("=" * 80)
    logger.info("FLAGSHIP 10-QUESTION PINPOINT DIAGNOSTIC")
    logger.info("GGUF: %s", gguf_path)
    logger.info("Total probes: %d", len(battery))
    logger.info("=" * 80)

    passed_count = 0
    results = []
    t0 = time.time()

    with start_llama_server(gguf_path, port=port, ctx=4096) as base_url:
        for idx, probe in enumerate(battery, 1):
            cat = probe["category"]
            pid = probe["id"]
            prompt = probe["prompt"]
            checker = probe["checker"]
            max_toks = probe.get("max_tokens", 128)
            sys_p = sarc.ARC_HLE_SYS if cat in ("ARC_AGI_1", "HLE_Frontier") else (
                "You are a concise expert assistant. Follow output format instructions strictly."
            )

            resp, dt = query_llama_server(base_url, prompt, max_tokens=max_toks, sys_prompt=sys_p)
            passed = bool(checker(resp))
            diag = "PASS"

            if not passed:
                if cat == "HumanEval_Code":
                    code_only = extract_target_function_only(resp, probe.get("entry_point", ""))
                    diag = diagnose_python_test_failure(code_only + "\n" + probe.get("test_code", ""), probe.get("entry_point", ""))
                elif cat in ("ARC_AGI_1", "HLE_Frontier"):
                    ltr = sarc.extract_option_letter_abcde(resp, probe.get("valid_letters", "ABCD"))
                    diag = f"Wrong/Missing option (extracted={ltr!r})"
                else:
                    diag = "Wrong answer or format mismatch"

            passed_count += int(passed)
            status_str = "PASS" if passed else f"FAIL ({diag})"
            logger.info("[%2d/10] %-15s | %-22s | %.2fs | %s", idx, cat, pid[:22], dt, status_str)
            if not passed:
                logger.info("   RAW RESP: %r", resp[:240])

            results.append({
                "idx": idx,
                "category": cat,
                "id": pid,
                "passed": passed,
                "diag": diag,
                "latency_s": round(dt, 2),
                "resp": resp[:400],
            })

    elapsed = time.time() - t0
    logger.info("=" * 80)
    logger.info("SCORE: %d / %d (%.1f%%) in %.1fs", passed_count, len(battery), 100.0 * passed_count / len(battery), elapsed)
    logger.info("=" * 80)
    return passed_count, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=str, default=r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v12-q8_0.gguf")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    run_pinpoint(args.gguf, port=args.port)
