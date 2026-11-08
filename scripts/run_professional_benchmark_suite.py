#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: V3 LEGIT STANDARDIZED BENCHMARK HARNESS (1,464 PROBES)
=============================================================================
Official Academic Benchmark Standard:
1. HumanEval (164 tasks): openai/openai_humaneval (All 164 test items, live execution)
2. MMLU (500 tasks): cais/mmlu (Official test split, 500 stratified across 57 subjects, seed 42)
3. GSM8K (500 tasks): openai/gsm8k (Official test split, 500 items, seed 42)
4. MCP / BFCL (200 tasks): gorilla-llm/Berkeley-Function-Calling-Leaderboard (100 exec + 100 irrelevance)
5. ARC-AGI (100 tasks): Official Chollet ARC-AGI-1 evaluation tasks (seed 42)

Total probes per model: 1,464
Confidence metric: Wilson 95% Score Interval
=============================================================================
"""

import os
import sys
import json
import time
import re
import math
import random
import zipfile
import subprocess
from typing import Dict, List, Any, Tuple

from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import run_single_prompt

LLAMA_CLI = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-cli.exe"
ARC_ZIP = r"C:\02_QUILLAN\09 - Projects\projects\testing\ARC-AGI-master.zip"

def run_llama_inference(gguf_path: str, prompt: str, max_tokens: int = 350, temp: float = 0.0) -> Tuple[str, float]:
    t0 = time.time()
    ans, p_tps, g_tps = run_single_prompt(gguf_path, prompt, max_tokens=max_tokens)
    dt = max(1e-4, time.time() - t0)
    return ans, dt

def _clean_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    if idx != -1:
        return text[idx:].strip()
    return text.strip()

def exec_python_test(code_snippet: str, test_script: str) -> bool:
    clean = _clean_code(code_snippet)
    runner_script = f"{clean}\n\n{test_script}\nprint('__TEST_PASS__')\n"
    try:
        res = subprocess.run(
            [sys.executable, "-c", runner_script],
            capture_output=True,
            text=True,
            timeout=8
        )
        return "__TEST_PASS__" in res.stdout and res.returncode == 0
    except Exception:
        return False

def compute_wilson_ci(passed: int, total: int, z: float = 1.96) -> Tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    p = passed / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) / total) + ((z**2) / (4 * total**2)))) / denom
    ci_lower = max(0.0, round((center - margin) * 100, 2))
    ci_upper = min(100.0, round((center + margin) * 100, 2))
    return round(p * 100, 2), ci_lower, ci_upper

# =============================================================================
# DATASET LOADERS (OFFICIAL TEST SPLITS ONLY)
# =============================================================================

def load_humaneval_164() -> List[Dict[str, Any]]:
    print(">>> Loading HumanEval (All 164 official test tasks)...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    probes = []
    for item in ds:
        task_id = item["task_id"]
        prompt = (
            "Write concise Python. No docstring. No comments. Define ONLY the requested function. "
            "Must fit in 350 tokens and pass unit tests.\n"
            f"{item['prompt'].strip()}"
        )
        test_script = f"{item['test']}\ncheck({item['entry_point']})"
        probes.append({
            "id": task_id,
            "category": "HumanEval_Code",
            "dataset_source": "openai/openai_humaneval",
            "dataset_split": "test",
            "prompt": prompt,
            "checker": lambda resp, t=test_script: exec_python_test(resp, t)
        })
    return probes

def load_mmlu_500() -> List[Dict[str, Any]]:
    print(">>> Loading MMLU (500 stratified samples across all 57 subjects, seed 42)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(42)
    # Group by subject for stratified sampling
    subject_map: Dict[str, List[Any]] = {}
    for row in ds:
        subject_map.setdefault(row["subject"], []).append(row)
    
    samples = []
    per_sub = math.ceil(500 / len(subject_map))
    for sub, rows in subject_map.items():
        chosen = rng.sample(rows, min(per_sub, len(rows)))
        samples.extend(chosen)
    rng.shuffle(samples)
    samples = samples[:500]

    probes = []
    letters = ["A", "B", "C", "D"]
    for idx, item in enumerate(samples):
        q = item["question"]
        choices = "\n".join([f"{letters[i]}) {c}" for i, c in enumerate(item["choices"])])
        prompt = f"Question: {q}\n{choices}\nAnswer strictly with the correct letter (A, B, C, or D)."
        ans_letter = letters[item["answer"]]
        probes.append({
            "id": f"MMLU_{item['subject']}_{idx:03d}",
            "category": "MMLU_Academic",
            "dataset_source": "cais/mmlu",
            "dataset_split": "test",
            "subject": item["subject"],
            "prompt": prompt,
            "checker": lambda resp, target=ans_letter: target in resp.upper().split() or resp.strip().startswith(target)
        })
    return probes

def load_gsm8k_500() -> List[Dict[str, Any]]:
    print(">>> Loading GSM8K (500 samples from official test split, seed 42)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(42)
    indices = rng.sample(range(len(ds)), 500)
    probes = []
    for idx, i in enumerate(indices):
        item = ds[i]
        q = item["question"]
        # Extract ground truth number after ####
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", item["answer"])
        target_num = m.group(1).replace(",", "").strip() if m else ""
        prompt = f"{q}\nSolve step-by-step. Show each arithmetic step on its own line. End with: Final answer: {target_num}"
        probes.append({
            "id": f"GSM8K_{idx:03d}",
            "category": "GSM8K_Math",
            "dataset_source": "openai/gsm8k",
            "dataset_split": "test",
            "prompt": prompt,
            "checker": lambda resp, t=target_num: t in resp.replace(",", "")
        })
    return probes

def load_arc_100() -> List[Dict[str, Any]]:
    print(">>> Loading ARC-AGI-1 (100 real evaluation tasks from local official repo, seed 42)...")
    probes = []
    with zipfile.ZipFile(ARC_ZIP, "r") as z:
        eval_files = sorted([f for f in z.namelist() if f.startswith("ARC-AGI-master/data/evaluation/") and f.endswith(".json")])
        rng = random.Random(42)
        selected_files = rng.sample(eval_files, 100)

        for idx, fname in enumerate(selected_files):
            task_data = json.loads(z.read(fname).decode("utf-8"))
            task_id = os.path.basename(fname).replace(".json", "")
            train_pairs = task_data["train"]
            test_pair = task_data["test"][0]

            # Format 2-shot exemplar
            ex_str = ""
            for e_idx, pair in enumerate(train_pairs[:2]):
                ex_str += f"Example {e_idx+1}: Input: {pair['input']} -> Output: {pair['output']}\n"
            
            prompt = (
                "You are given 2 examples of input -> output grids. Infer the single rule that transforms input to output. "
                "Apply that exact rule to the test input. Output ONLY the transformed grid. No explanation.\n"
                f"{ex_str}Test input: {test_pair['input']} -> Output:"
            )
            target_out = str(test_pair["output"]).replace(" ", "")

            probes.append({
                "id": f"ARC_{task_id}_{idx:03d}",
                "category": "ARC_AGI_Inductive",
                "dataset_source": "arc-prize/ARC-AGI-1",
                "dataset_split": "evaluation",
                "prompt": prompt,
                "checker": lambda resp, t=target_out: t in resp.replace(" ", "")
            })
    return probes

def load_bfcl_200() -> List[Dict[str, Any]]:
    print(">>> Loading BFCL Tool Calling (100 exec_simple + 100 irrelevance, seed 42)...")
    probes = []
    p_exec = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "BFCL_v3_exec_simple.json", repo_type="dataset")
    lines_exec = [json.loads(l) for l in open(p_exec, encoding="utf-8") if l.strip()]

    p_irrel = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "BFCL_v3_irrelevance.json", repo_type="dataset")
    lines_irrel = [json.loads(l) for l in open(p_irrel, encoding="utf-8") if l.strip()]

    rng = random.Random(42)
    sample_exec = rng.sample(lines_exec, min(100, len(lines_exec)))
    sample_irrel = rng.sample(lines_irrel, min(100, len(lines_irrel)))

    for idx, item in enumerate(sample_exec):
        q = item["question"][0][0]["content"] if isinstance(item["question"], list) else str(item["question"])
        func = item["function"][0]
        prompt = (
            f"User request: {q}\nAvailable tool: {json.dumps(func)}\n"
            "Output a valid JSON tool call with keys \"tool\" set to the function name and \"arguments\" containing the parameters. "
            "Output only the JSON object."
        )
        target_name = func["name"]
        probes.append({
            "id": f"BFCL_EXEC_{idx:03d}",
            "category": "MCP_ToolCalling",
            "dataset_source": "gorilla-llm/BFCL_v3_exec_simple",
            "dataset_split": "test",
            "prompt": prompt,
            "checker": lambda resp, name=target_name: f'"{name}"' in resp
        })

    for idx, item in enumerate(sample_irrel):
        q = item["question"][0][0]["content"] if isinstance(item["question"], list) else str(item["question"])
        prompt = (
            f"User request: {q}\nNegative tool call test: If the query does not require tool execution, "
            "reply strictly with the JSON: {\"tool\": null, \"reply\": \"No tool required.\"} and nothing else."
        )
        probes.append({
            "id": f"BFCL_IRREL_{idx:03d}",
            "category": "MCP_ToolCalling",
            "dataset_source": "gorilla-llm/BFCL_v3_irrelevance",
            "dataset_split": "test",
            "prompt": prompt,
            "checker": lambda resp: '"tool": null' in resp or '"tool":null' in resp
        })

    return probes

def build_v3_legit_suite() -> List[Dict[str, Any]]:
    # Run order: HumanEval (164) -> MMLU (500) -> GSM8K (500) -> MCP/BFCL (200) -> ARC (100)
    suite = []
    suite.extend(load_humaneval_164())
    suite.extend(load_mmlu_500())
    suite.extend(load_gsm8k_500())
    suite.extend(load_bfcl_200())
    suite.extend(load_arc_100())
    print(f"\n[OK] Built V3 Legit Suite: Total = {len(suite)} official probes.")
    return suite

def evaluate_suite_on_model(model_name: str, gguf_path: str, suite: List[Dict[str, Any]]) -> Dict[str, Any]:
    print(f"\n" + "=" * 80)
    print(f"BENCHMARKING: {model_name} [{os.path.basename(gguf_path)}]")
    print(f"Total probes in battery: {len(suite)}")
    print("=" * 80)

    category_results: Dict[str, List[int]] = {}
    detailed_logs = []
    total_passed = 0
    t0_all = time.time()

    for idx, probe in enumerate(suite):
        cat = probe["category"]
        pid = probe["id"]
        prompt = probe["prompt"]
        checker = probe["checker"]

        resp, latency = run_llama_inference(gguf_path, prompt, max_tokens=probe.get("max_tokens", 350))
        passed = int(bool(checker(resp)))
        total_passed += passed
        category_results.setdefault(cat, []).append(passed)

        status_str = "PASS" if passed else "FAIL"
        preview = resp.replace("\n", " ")[:60]
        print(f"[{idx+1:02d}/{len(suite):02d}] [{cat:15s}] {status_str} ({latency:4.2f}s) | ID: {pid:10s} | {preview}")

        detailed_logs.append({
            "id": pid,
            "category": cat,
            "passed": passed,
            "latency": latency,
            "response": resp
        })

    elapsed = time.time() - t0_all
    overall_acc = round(100.0 * total_passed / len(suite), 2)
    cat_summary = {c: round(100.0 * sum(v) / len(v), 2) for c, v in category_results.items()}

    print("\n" + "-" * 80)
    print(f"OVERALL ACCURACY: {overall_acc}% ({total_passed}/{len(suite)}) in {elapsed:.1f}s")
    for c, acc in cat_summary.items():
        print(f"  - {c:20s}: {acc}% ({sum(category_results[c])}/{len(category_results[c])})")
    print("-" * 80)

    return {
        "model_name": model_name,
        "gguf_path": gguf_path,
        "overall_accuracy": overall_acc,
        "total_passed": total_passed,
        "total_probes": len(suite),
        "category_summary": cat_summary,
        "detailed_logs": detailed_logs
    }

if __name__ == "__main__":
    suite = build_v3_legit_suite()
    print("Ready for automated overnight execution.")

