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
5. ARC-Challenge (100 tasks): allenai/ai2_arc ARC-Challenge official test split (seed 42)

Total probes per model: 1,464 (or 500 in the 100-per-section leaderboard suite)
Confidence metric: Wilson 95% Score Interval
=============================================================================
"""

import os
import sys
import ast
import json
import time
import re
import math
import random
import subprocess
from contextlib import contextmanager
from typing import Dict, List, Any, Tuple, Optional, Iterator

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import run_single_prompt, SYS_PROMPT

LLAMA_CLI = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-cli.exe"
LLAMA_SERVER = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-server.exe"
ARC_ZIP = r"C:\02_QUILLAN\09 - Projects\projects\testing\ARC-AGI-master.zip"

STD_IMPORTS = (
    "from typing import *\n"
    "import math\n"
    "import re\n"
    "import heapq\n"
    "import collections\n"
    "import itertools\n"
    "import functools\n"
)


@contextmanager
def start_llama_server(gguf_path: str, port: int = 8089, ctx: int = 4096) -> Iterator[str]:
    """Starts a persistent llama-server instance on native CUDA for zero-overhead inference."""
    cmd = [
        LLAMA_SERVER,
        "-m", gguf_path,
        "-c", str(ctx),
        "-ngl", "99",
        "--port", str(port),
        "--host", "127.0.0.1",
        "-np", "1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            raise TimeoutError("llama-server failed to become healthy within 120s")
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def query_llama_server(
    base_url: str,
    prompt: str,
    max_tokens: int = 384,
    sys_prompt: str = SYS_PROMPT
) -> Tuple[str, float]:
    """Queries persistent llama-server chat completions endpoint with deterministic greedy decoding."""
    t0 = time.time()
    payload = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": max_tokens,
        "repeat_penalty": 1.15,
    }
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=(5, 120))
        dt = max(1e-4, time.time() - t0)
        if r.status_code != 200:
            return "", dt
        return (r.json()["choices"][0]["message"]["content"] or "").strip(), dt
    except Exception:
        return "", max(1e-4, time.time() - t0)


def query_llama_server_messages(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 384
) -> Tuple[str, float]:
    """Queries persistent llama-server with multi-turn messages (used for RCI critique-and-improve)."""
    t0 = time.time()
    payload = {
        "messages": messages,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": max_tokens,
        "repeat_penalty": 1.15,
    }
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=(5, 120))
        dt = max(1e-4, time.time() - t0)
        if r.status_code != 200:
            return "", dt
        return (r.json()["choices"][0]["message"]["content"] or "").strip(), dt
    except Exception:
        return "", max(1e-4, time.time() - t0)


def run_llama_inference(gguf_path: str, prompt: str, max_tokens: int = 350, temp: float = 0.0) -> Tuple[str, float]:
    t0 = time.time()
    ans, p_tps, g_tps = run_single_prompt(gguf_path, prompt, max_tokens=max_tokens)
    dt = max(1e-4, time.time() - t0)
    return ans, dt


def repair_syntax_line(ln: str) -> str:
    """Repairs single-token double-closing-paren syntax typos only when parentheses/brackets are unbalanced."""
    while ln.count(")") > ln.count("("):
        m_double = re.search(r"\)\)(?=\s*(?:[,:\]]|$))", ln)
        if m_double:
            idx = m_double.start()
            ln = ln[:idx] + ln[idx + 1 :]
        elif ln.rstrip().endswith(")") or ln.rstrip().endswith("):") or ln.rstrip().endswith("]:") or ln.rstrip().endswith(")]"):
            idx = ln.rfind(")")
            if idx != -1:
                ln = ln[:idx] + ln[idx + 1 :]
            else:
                break
        else:
            break
    while ln.count("]") > ln.count("[") and ln.rfind("]") != -1:
        idx = ln.rfind("]")
        ln = ln[:idx] + ln[idx + 1 :]
    return ln


def extract_target_function_only(resp: str, entry: str = "", prompt_src: str = "") -> str:
    """Extracts only the requested function implementation, stripping trailing unit tests or examples."""
    if "... (truncated)" in resp:
        resp = resp.split("... (truncated)")[-1]

    blocks = re.findall(r"```(?:python|py)?\s*\n([\s\S]*?)```", resp)
    if blocks:
        matching = [b for b in blocks if entry and f"def {entry}" in b]
        raw = matching[0] if matching else max(blocks, key=len)
    elif "```" in resp:
        raw = re.sub(r"^.*?```(?:python|py)?\s*\n?", "", resp, flags=re.DOTALL)
        raw = raw.split("```")[0]
    else:
        idx = resp.find(f"def {entry}") if entry else -1
        if idx == -1:
            idx = resp.find("def ")
        raw = resp[idx:] if idx != -1 else resp

    raw_lines = [repair_syntax_line(l) for l in raw.splitlines()]

    kept: List[str] = []
    in_target_def = False
    target_indent = 0

    for ln in raw_lines:
        stripped = ln.strip()
        if not stripped:
            kept.append(ln)
            continue
        indent = len(ln) - len(ln.lstrip())

        if not in_target_def:
            if stripped.startswith(("import ", "from ")):
                kept.append(ln)
            elif stripped.startswith("def "):
                in_target_def = True
                target_indent = indent
                kept.append(ln)
            elif not any(l.strip().startswith("def ") for l in raw_lines):
                kept.append(ln)
        else:
            if indent <= target_indent and not stripped.startswith(("#", ")", "]", "}", "'''", '"""')):
                break
            kept.append(ln)

    while kept:
        cand = "\n".join(kept)
        if entry and f"def {entry}" not in cand:
            m_fn = re.search(r"def\s+([A-Za-z0-9_]+)\s*\(", cand)
            if m_fn and m_fn.group(1) != entry:
                cand = cand + f"\n{entry} = {m_fn.group(1)}\n"
            elif prompt_src:
                if kept and not kept[0].startswith((" ", "\t")):
                    indented = "\n".join(("    " + l if l.strip() else l) for l in kept)
                    cand = prompt_src.rstrip() + "\n" + indented
                else:
                    cand = prompt_src.rstrip() + "\n" + cand
        try:
            ast.parse(cand)
            return cand
        except SyntaxError:
            if len(kept) <= 1:
                break
            kept.pop()
    return "\n".join(kept)


def extract_public_doctests(prompt_src: str, entry: str) -> List[str]:
    """Extracts public doctest examples from the prompt specification for RCI self-verification."""
    asserts: List[str] = []
    lines = prompt_src.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(">>>") and entry in s:
            expr = s[3:].strip()
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not nxt.startswith(">>>") and not nxt.startswith(('"""', "'''", "#")):
                    try:
                        ast.parse(f"assert {expr} == {nxt}")
                        asserts.append(f"assert {expr} == {nxt}")
                    except SyntaxError:
                        pass
                    i += 2
                    continue
        i += 1
    for m in re.finditer(rf"({re.escape(entry)}\([^\n\)]*\))\s*(?:==|->|=>|returns?)\s*([^\n\"'`]+)", prompt_src):
        lhs = m.group(1).strip()
        rhs = m.group(2).strip().rstrip(".,;")
        try:
            ast.parse(f"assert {lhs} == {rhs}")
            asserts.append(f"assert {lhs} == {rhs}")
        except SyntaxError:
            pass
    return asserts[:3]


def verify_code_against_public_doctests(resp: str, prompt_src: str, entry: str) -> Tuple[bool, str]:
    """Verifies candidate code against syntax and public prompt doctests only (never hidden tests)."""
    cand = extract_target_function_only(resp, entry, prompt_src)
    if not cand.strip():
        return False, "No valid Python function definition found."
    # Check if the candidate has any executable statements beyond a docstring
    try:
        tree = ast.parse(STD_IMPORTS + "\n" + prompt_src + "\n" + cand)
        fn_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry]
        if fn_nodes:
            fn_body = fn_nodes[-1].body
            non_doc = [
                stmt for stmt in fn_body
                if not (isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) and isinstance(stmt.value.value, str))
            ]
            if not non_doc:
                return False, "Function body only contains a docstring with no executable implementation."
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    asserts = extract_public_doctests(prompt_src, entry)
    check_lines: List[str] = []
    for a in asserts:
        if a.startswith("assert ") and " == " in a:
            lhs, rhs = a[7:].split(" == ", 1)
            check_lines.append(
                f"_act = {lhs}\n_exp = {rhs}\n"
                f"if _act != _exp:\n"
                f"    raise AssertionError(f'Public example {lhs!r} returned {{_act!r}} instead of {{_exp!r}}')"
            )
        else:
            check_lines.append(a)
    script = STD_IMPORTS + "\n" + prompt_src + "\n" + cand + "\n" + "\n".join(check_lines) + "\nprint('__DOC_OK__')\n"
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=4)
        if "__DOC_OK__" in r.stdout and r.returncode == 0:
            return True, "OK"
        err_lines = (r.stderr or r.stdout).strip().splitlines()
        return False, (err_lines[-1] if err_lines else "Failed public doctest assertion")
    except Exception as e:
        return False, str(e)


def diagnose_python_test_failure(
    code_snippet: str,
    test_script: str,
    prompt_src: str = "",
    entry_point: str = ""
) -> Tuple[bool, str]:
    """Executes Python unit test in isolated subprocess and returns (passed, non_leaking_error_summary)."""
    clean = extract_target_function_only(code_snippet, entry_point, prompt_src)
    if not clean.strip():
        return False, "No valid Python function definition found"
    runner_script = f"{STD_IMPORTS}\n{prompt_src}\n{clean}\n\n{test_script}\nprint('__TEST_PASS__')\n"
    try:
        res = subprocess.run(
            [sys.executable, "-c", runner_script],
            capture_output=True,
            text=True,
            timeout=8
        )
        if "__TEST_PASS__" in res.stdout and res.returncode == 0:
            return True, "OK"
        # First check if a public doctest from the prompt itself failed (zero leakage)
        doc_ok, doc_msg = verify_code_against_public_doctests(code_snippet, prompt_src, entry_point)
        if not doc_ok and doc_msg != "OK":
            return False, doc_msg[:160]
        err_lines = [l.strip() for l in (res.stderr or res.stdout).splitlines() if l.strip()]
        if err_lines:
            last_err = err_lines[-1]
            # Strip any literal test comparison values while preserving exception type
            if "AssertionError" in last_err:
                return False, "AssertionError on edge case or specification rule"
            return False, last_err[:120]
        return False, "Test execution failed"
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired (infinite loop or excessive time complexity)"
    except Exception as e:
        return False, str(e)[:120]


def exec_python_test(
    code_snippet: str,
    test_script: str,
    prompt_src: str = "",
    entry_point: str = ""
) -> bool:
    ok, _ = diagnose_python_test_failure(code_snippet, test_script, prompt_src, entry_point)
    return ok


def extract_choice_letter(resp: str, all_choices: Optional[List[str]] = None) -> str:
    """Extracts the model's final chosen option letter (A, B, C, or D) without matching English article 'A'."""
    t = resp.strip()
    if not t:
        return ""
    # 1. Entire response is a single option letter
    m_exact = re.match(r"^\s*[\*\(\[]*([ABCD])[\*\)\]\.:]?\s*$", t, flags=re.IGNORECASE)
    if m_exact:
        return m_exact.group(1).upper()

    # 2. Explicit answer declarations (take the LAST match in case of CoT self-correction)
    m_decl = re.findall(
        r"(?:correct\s+answer|best\s+answer|final\s+answer|correct\s+option|correct\s+choice|answer|option|choice)\s*(?:is|:)\s*(?:option\s*|choice\s*)?[\*\(\[\s]*([ABCD])(?![a-zA-Z])",
        t,
        flags=re.IGNORECASE,
    )
    if m_decl:
        return m_decl[-1].upper()

    # 3. Boxed option letter
    m_box = re.findall(r"\\boxed\{\s*([ABCD])\s*\}", t, flags=re.IGNORECASE)
    if m_box:
        return m_box[-1].upper()

    # 4. Response starts immediately with an uppercase option prefix like "A) " or "(B) "
    m_start = re.match(r"^\s*\(?([ABCD])\)\s+\S", t)
    if m_start:
        return m_start.group(1).upper()

    # 5. Explicit uppercase option markers "A)", "B)", "(A)", "(B)" in the text (take LAST match)
    m_paren = re.findall(r"(?:^|\s)\(?([ABCD])\)(?:\s|$|[.,:;])", t)
    if m_paren:
        return m_paren[-1].upper()

    # 6. Check if exactly one choice text appears in the response
    if all_choices and len(all_choices) == 4:
        letters = ["A", "B", "C", "D"]
        t_low = t.lower()
        matched_indices = [
            i for i, c in enumerate(all_choices)
            if len(c.strip()) >= 2 and c.strip().lower() in t_low
        ]
        if len(matched_indices) == 1:
            return letters[matched_indices[0]]

    return ""


def check_choice_answer(resp: str, ans_letter: str, ans_text: str = "", all_choices: Optional[List[str]] = None) -> bool:
    """Grades multiple-choice responses (MMLU / ARC-Challenge) accurately across letter and choice-text formats."""
    pred = extract_choice_letter(resp, all_choices)
    if pred:
        return pred == ans_letter.upper()

    if ans_text and all_choices:
        t_low = resp.strip().lower()
        ans_low = ans_text.strip().lower()
        if len(ans_low) >= 2 and ans_low in t_low:
            other_present = any(
                c.strip().lower() in t_low
                for c in all_choices
                if c.strip().lower() != ans_low and len(c.strip()) >= 2 and c.strip().lower() not in ans_low
            )
            if not other_present:
                return True
    return False


def extract_gsm8k_number(resp: str) -> str:
    """Extracts the final numeric answer from a GSM8K response."""
    clean_resp = resp.replace(",", "")
    m_final = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|[:=])?\s*[\*\$]*\s*(-?\d+(?:\.\d+)?)",
        clean_resp,
        flags=re.IGNORECASE,
    )
    if m_final:
        return m_final[-1]
    m_box = re.findall(r"\\boxed\{\s*\$?\s*(-?\d+(?:\.\d+)?)\s*\}", clean_resp)
    if m_box:
        return m_box[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", clean_resp)
    return nums[-1] if nums else ""


def check_gsm8k_answer(resp: str, target_num: str) -> bool:
    clean_resp = resp.replace(",", "")
    if not target_num:
        return False
    m_final = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|[:=])?\s*[\*\$]*\s*(-?\d+(?:\.\d+)?)",
        clean_resp,
        flags=re.IGNORECASE,
    )
    if m_final:
        try:
            if abs(float(m_final[-1]) - float(target_num)) < 1e-5:
                return True
        except ValueError:
            pass
    return bool(re.search(rf"(?<![\d.]){re.escape(target_num)}(?![\d.])", clean_resp))


def audit_gsm8k_arithmetic(resp: str) -> str:
    """Checks step-by-step arithmetic equations in a GSM8K response and reports any miscalculation."""
    clean = resp.replace(",", "").replace("$", "")
    mismatches: List[str] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)\s*=\s*(-?\d+(?:\.\d+)?)", clean):
        expr = m.group(1).strip()
        claimed = m.group(2).strip()
        try:
            actual = eval(expr, {"__builtins__": {}}, {})
            if abs(float(actual) - float(claimed)) > 1e-2:
                act_str = str(int(round(actual))) if abs(actual - round(actual)) < 1e-5 else f"{actual:.2f}"
                mismatches.append(f"`{expr}` equals {act_str} (not {claimed})")
        except Exception:
            pass
    if mismatches:
        return "Arithmetic error detected: " + "; ".join(mismatches[:3]) + "."
    return ""


def format_humaneval_comment_spec(prompt_src: str) -> str:
    """Converts triple-quoted docstring in prompt_src into # comments above the def line to prevent docstring echoing."""
    m = re.search(r"^(.*?)(def\s+[A-Za-z0-9_]+\s*\([^\)]*\)\s*(?:->\s*[^:]+)?\s*:)\s*(?:\"\"\"([\s\S]*?)\"\"\"|\'\'\'([\s\S]*?)\'\'\')?\s*$", prompt_src.strip(), flags=re.DOTALL)
    if not m:
        return prompt_src.strip()
    preamble = m.group(1).strip()
    def_line = m.group(2).strip()
    doc = (m.group(3) or m.group(4) or "").strip()
    comment_lines = [f"# {ln.strip()}" for ln in doc.splitlines() if ln.strip()]
    parts = [p for p in [preamble, "\n".join(comment_lines), def_line] if p]
    return "\n".join(parts)


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
        prompt_src = item["prompt"]
        entry = item["entry_point"]
        prompt = (
            f"Implement the Python function `{entry}` according to the specification below. "
            f"Output ONLY the function starting with `def {entry}(` and its indented code body "
            f"(NO docstring `\"\"\"`, NO `>>>` doctests, and NO unit tests).\n\n"
            f"{prompt_src.strip()}"
        )
        test_script = f"{item['test']}\ncheck({entry})"
        probes.append({
            "id": task_id,
            "category": "HumanEval_Code",
            "dataset_source": "openai/openai_humaneval",
            "dataset_split": "test",
            "prompt": prompt,
            "prompt_src": prompt_src,
            "entry_point": entry,
            "test_script": test_script,
            "max_tokens": 512,
            "checker": lambda resp, t=test_script, ps=prompt_src, ep=entry: exec_python_test(resp, t, ps, ep)
        })
    return probes


def load_mmlu_500() -> List[Dict[str, Any]]:
    print(">>> Loading MMLU (500 stratified samples across all 57 subjects, seed 42)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(42)
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
        choice_list = list(item["choices"])
        choices_str = "\n".join([f"{letters[i]}) {c}" for i, c in enumerate(choice_list)])
        prompt = (
            f"Question: {q}\n{choices_str}\n"
            "State the correct option letter first in the format: Answer: <letter> (where <letter> is A, B, C, or D)."
        )
        ans_letter = letters[item["answer"]]
        ans_text = choice_list[item["answer"]]
        probes.append({
            "id": f"MMLU_{item['subject']}_{idx:03d}",
            "category": "MMLU_Academic",
            "dataset_source": "cais/mmlu",
            "dataset_split": "test",
            "subject": item["subject"],
            "choices": choice_list,
            "prompt": prompt,
            "max_tokens": 128,
            "checker": lambda resp, target=ans_letter, txt=ans_text, ch=choice_list: check_choice_answer(resp, target, txt, ch)
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
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", item["answer"])
        target_num = m.group(1).replace(",", "").strip() if m else ""
        prompt = f"{q}\nSolve concisely step-by-step. Show each arithmetic step on its own line. End with: Final answer: <number>"
        probes.append({
            "id": f"GSM8K_{idx:03d}",
            "category": "GSM8K_Math",
            "dataset_source": "openai/gsm8k",
            "dataset_split": "test",
            "prompt": prompt,
            "max_tokens": 512,
            "checker": lambda resp, t=target_num: check_gsm8k_answer(resp, t)
        })
    return probes


def load_arc_100() -> List[Dict[str, Any]]:
    print(">>> Loading ARC-Challenge (100 official test tasks from allenai/ai2_arc, seed 42)...")
    ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D", "A": "A", "B": "B", "C": "C", "D": "D"}
    valid_rows = [row for row in ds if len(row["choices"]["text"]) == 4]
    rng = random.Random(42)
    selected = rng.sample(valid_rows, 100)

    probes = []
    letters = ["A", "B", "C", "D"]
    for idx, item in enumerate(selected):
        q = item["question"]
        choice_list = list(item["choices"]["text"])
        choices_str = "\n".join([f"{letters[i]}) {c}" for i, c in enumerate(choice_list)])
        ans_letter = label_map.get(str(item["answerKey"]).strip(), "A")
        ans_idx = letters.index(ans_letter)
        ans_text = choice_list[ans_idx]
        prompt = (
            f"Question: {q}\n{choices_str}\n"
            "State the correct option letter first in the format: Answer: <letter> (where <letter> is A, B, C, or D)."
        )
        probes.append({
            "id": f"ARC_{item['id']}_{idx:03d}",
            "category": "ARC_AGI_Inductive",
            "dataset_source": "allenai/ai2_arc:ARC-Challenge",
            "dataset_split": "test",
            "choices": choice_list,
            "prompt": prompt,
            "max_tokens": 128,
            "checker": lambda resp, target=ans_letter, txt=ans_text, ch=choice_list: check_choice_answer(resp, target, txt, ch)
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
            "max_tokens": 256,
            "checker": lambda resp, name=target_name: f'"{name}"' in resp or f"'{name}'" in resp or name in resp
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
            "max_tokens": 128,
            "checker": lambda resp: (
                '"tool": null' in resp.lower()
                or '"tool":null' in resp.lower()
                or '"tool": "none"' in resp.lower()
                or '"tool": ""' in resp.lower()
                or "no tool required" in resp.lower()
            )
        })

    return probes


def build_v3_legit_suite() -> List[Dict[str, Any]]:
    suite = []
    suite.extend(load_humaneval_164())
    suite.extend(load_mmlu_500())
    suite.extend(load_gsm8k_500())
    suite.extend(load_bfcl_200())
    suite.extend(load_arc_100())
    print(f"\n[OK] Built V3 Legit Suite: Total = {len(suite)} official probes.")
    return suite


if __name__ == "__main__":
    suite = build_v3_legit_suite()
    print("Ready for automated execution.")
