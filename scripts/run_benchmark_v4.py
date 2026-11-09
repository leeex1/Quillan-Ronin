#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin Benchmark Harness V4 (integrity-corrected).

Fixes over V3 (see audit_harness_integrity.py):
  * Inference runs through llama-server's OpenAI-compatible chat endpoint, so the
    model's own chat template is applied and responses contain no prompt echo.
  * Context raised from 1024 to 8192 tokens (V3 rejected 90/100 ARC prompts).
  * GSM8K prompt no longer contains the gold answer (V3 leaked it in the prompt).
  * BFCL scoring parses the JSON call and compares function name + arguments to
    ground truth; irrelevance prompts no longer dictate the expected output.
  * MMLU uses a strict letter extractor instead of substring search.
  * HumanEval executes prompt imports + completion + official tests in an
    isolated subprocess with a timeout in a temporary directory.

Usage:
  python run_benchmark_v4.py --models A,B,C [--n 100] [--smoke 3] [--out FILE]
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import requests

LLAMA_SERVER = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-server.exe"
ARC_ZIP = r"C:\02_QUILLAN\09 - Projects\projects\testing\ARC-AGI-master.zip"
EXPORTS = r"C:\02_QUILLAN\exports"

MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "A": ("Model A Flagship (WSP v8.0)", os.path.join(EXPORTS, "quillan-1b-sovereign-wsp-v8-q4_k_m.gguf")),
    "B": ("Model B Mini (WSP v8.0)", os.path.join(EXPORTS, "quillan-mini-0.5b-wsp-v8-q4_k_m.gguf")),
    "C": ("Model C Dragon (WSP v8.0)", os.path.join(EXPORTS, "quillan-dragon-1b-wsp-v8-q4_k_m.gguf")),
    "A0": ("Control: Llama-3.2-1B-Instruct clean", os.path.join(EXPORTS, "quillan-1b-clean-base-q4_k_m.gguf")),
    "B0": ("Control: Qwen2.5-0.5B-Instruct clean", os.path.join(EXPORTS, "quillan-mini-0.5b-clean-base-q4_k_m.gguf")),
    "C0": ("Control: Falcon3-1B-Instruct clean", os.path.join(EXPORTS, "quillan-dragon-1b-clean-base-q4_k_m.gguf")),
}

SYSTEM_PROMPT = "You are a precise, helpful assistant. Follow the requested output format exactly."
CODE_TIMEOUT_S = 10
REQUEST_TIMEOUT_S = 180


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname, "component": record.name, "msg": record.getMessage()}
        return json.dumps(payload, ensure_ascii=False)


def build_logger() -> logging.Logger:
    log = logging.getLogger("bench_v4")
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@contextmanager
def llama_server(gguf: str, log: logging.Logger, port: int = 8089, ctx: int = 8192) -> Iterator[str]:
    """Start llama-server for one model and stop it deterministically."""
    cmd = [LLAMA_SERVER, "-m", gguf, "-c", str(ctx), "-ngl", "99", "--port", str(port),
           "--host", "127.0.0.1", "-np", "1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
            try:
                if requests.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("llama-server did not become healthy within 120s")
        log.info(f"server ready model={os.path.basename(gguf)} ctx={ctx}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def chat(base: str, user: str, max_tokens: int) -> Tuple[str, float]:
    t0 = time.time()
    body = {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        "temperature": 0.0, "top_k": 1, "seed": 42, "max_tokens": max_tokens,
    }
    try:
        r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=(5, REQUEST_TIMEOUT_S))
        if r.status_code != 200:
            return f"__HTTP_{r.status_code}__ {r.text[:200]}", time.time() - t0
        return r.json()["choices"][0]["message"]["content"] or "", time.time() - t0
    except requests.RequestException as exc:
        return f"__REQUEST_ERROR__ {type(exc).__name__}", time.time() - t0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def wilson(passed: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = passed / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * p, 2), round(100 * max(0.0, c - m), 2), round(100 * min(1.0, c + m), 2)


def extract_letter(resp: str) -> Optional[str]:
    t = resp.strip()
    for pat in (
        r"^\(?([ABCD])\)?(?:[\).:\s]|$)",
        r"answer\s*(?:is|:)?\s*\(?([ABCD])\b",
        r"\b([ABCD])\)",
        r"\(([ABCD])\)",
        r"\b([ABCD])\b",
    ):
        m = re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def extract_final_number(resp: str) -> Optional[float]:
    m = re.findall(r"final answer\s*[:=]?\s*\$?\s*(-?[\d,]*\.?\d+)", resp, flags=re.IGNORECASE)
    if m:
        return _to_float(m[-1])
    nums = re.findall(r"-?[\d,]*\.?\d+", resp)
    return _to_float(nums[-1]) if nums else None


def extract_code(resp: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n([\s\S]*?)```", resp)
    if blocks:
        raw = max(blocks, key=len)
    else:
        idx = resp.find("def ")
        raw = resp[idx:] if idx != -1 else resp

    clean_lines: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith(("# Unit test", "# Test", "# Example", "def test_", "if __name__", "assert ", "print(")):
            break
        clean_lines.append(ln)

    while clean_lines:
        candidate = "\n".join(clean_lines)
        try:
            ast.parse(candidate)
            break
        except SyntaxError:
            if len(clean_lines) <= 2:
                break
            clean_lines.pop()

    return "\n".join(clean_lines)


def run_python(source: str) -> bool:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            r = subprocess.run([sys.executable, "-I", path], cwd=td, capture_output=True, text=True,
                               timeout=CODE_TIMEOUT_S, env={"PYTHONHASHSEED": "0", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
            return r.returncode == 0 and "__PASS__" in r.stdout
        except subprocess.TimeoutExpired:
            return False


def extract_json_obj(resp: str) -> Optional[Any]:
    t = re.sub(r"```(?:json)?", "", resp)
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


def _norm(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, str):
        s_strip = v.strip()
        if (s_strip.startswith("[") and s_strip.endswith("]")) or (s_strip.startswith("{") and s_strip.endswith("}")):
            try:
                return _norm(ast.literal_eval(s_strip))
            except Exception:
                pass
        if s_strip.lower() in ("true", "false"):
            return s_strip.lower() == "true"
        f = _to_float(s_strip)
        return round(f, 6) if f is not None else s_strip.lower()
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def parse_ground_truth_call(gt: str) -> Tuple[str, Dict[str, Any]]:
    node = ast.parse(gt.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError("ground truth is not a call")
    name = ast.unparse(node.func)
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
    return name, kwargs


# ---------------------------------------------------------------------------
# Probe builders (official test/evaluation splits only)
# ---------------------------------------------------------------------------
@dataclass
class Probe:
    pid: str
    category: str
    source: str
    prompt: str
    max_tokens: int
    check: Callable[[str], bool] = field(repr=False)


def humaneval(n: int, rng: random.Random) -> List[Probe]:
    from datasets import load_dataset
    ds = list(load_dataset("openai/openai_humaneval", split="test"))
    out = []
    for it in rng.sample(ds, min(n, len(ds))):
        prompt_src, entry, test = it["prompt"], it["entry_point"], it["test"]
        header = "\n".join(l for l in prompt_src.splitlines() if l.startswith(("import ", "from ")))

        def check(resp: str, ps=prompt_src, e=entry, tst=test, hdr=header) -> bool:
            code = extract_code(resp)
            if f"def {e}" not in code:
                code = ps + code
            return run_python(f"{hdr}\n{code}\n\n{tst}\ncheck({e})\nprint('__PASS__')\n")

        user = (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            f"{prompt_src}"
        )
        out.append(Probe(it["task_id"], "HumanEval", "openai/openai_humaneval:test", user, 384, check))
    return out


def mmlu(n: int, rng: random.Random) -> List[Probe]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_sub: Dict[str, List[Any]] = {}
    for row in ds:
        by_sub.setdefault(row["subject"], []).append(row)
    subjects = sorted(by_sub)
    picked: List[Any] = []
    i = 0
    while len(picked) < n:
        s = subjects[i % len(subjects)]
        pool = by_sub[s]
        if pool:
            picked.append(pool.pop(rng.randrange(len(pool))))
        i += 1
    out = []
    for k, it in enumerate(picked):
        opts = "\n".join(f"{L}) {c}" for L, c in zip("ABCD", it["choices"]))
        gold = "ABCD"[it["answer"]]
        user = (f"The following is a multiple choice question about {it['subject'].replace('_', ' ')}.\n\n"
                f"{it['question']}\n{opts}\n\nRespond with only the letter of the correct answer.")
        out.append(Probe(f"MMLU_{it['subject']}_{k:03d}", "MMLU", "cais/mmlu:test", user, 16,
                         lambda r, g=gold: extract_letter(r) == g))
    return out


def gsm8k(n: int, rng: random.Random) -> List[Probe]:
    from datasets import load_dataset
    ds = list(load_dataset("openai/gsm8k", "main", split="test"))
    out = []
    for k, it in enumerate(rng.sample(ds, n)):
        gold = _to_float(it["answer"].split("####")[-1].strip())
        user = (f"{it['question']}\n\nSolve step by step. On the last line write "
                f"'Final answer: <number>' with only the number.")
        out.append(Probe(f"GSM8K_{k:03d}", "GSM8K", "openai/gsm8k:test", user, 512,
                         lambda r, g=gold: (lambda v: v is not None and g is not None and abs(v - g) < 1e-6)(extract_final_number(r))))
    return out


def arc(n: int, rng: random.Random) -> List[Probe]:
    from datasets import load_dataset
    ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
                 "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
    valid_rows = [row for row in ds if len(row["choices"]["text"]) == 4]
    picked = rng.sample(valid_rows, min(n, len(valid_rows)))
    out = []
    for k, it in enumerate(picked):
        texts = it["choices"]["text"]
        opts = "\n".join(f"{L}) {c}" for L, c in zip("ABCD", texts))
        gold = label_map.get(str(it["answerKey"]).strip(), "A")
        user = (f"The following is a multiple choice science question.\n\n"
                f"{it['question']}\n{opts}\n\nRespond with only the letter of the correct answer.")
        out.append(Probe(f"ARC_{it['id']}", "ARC", "allenai/ai2_arc:ARC-Challenge:test", user, 16,
                         lambda r, g=gold: extract_letter(r) == g))
    return out


def bfcl(n: int, rng: random.Random) -> List[Probe]:
    from huggingface_hub import hf_hub_download

    def load(name: str) -> List[Dict[str, Any]]:
        p = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", name, repo_type="dataset")
        with open(p, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    instr = ('Respond with only a JSON object of the form {"tool": <function name>, "arguments": {<parameters>}}. '
             'If none of the tools can answer the request, respond with {"tool": null}.')
    out: List[Probe] = []
    for it in rng.sample(load("BFCL_v3_exec_simple.json"), n // 2):
        q = it["question"][0][0]["content"]
        name, kwargs = parse_ground_truth_call(it["ground_truth"][0])

        def check(resp: str, nm=name, kw=kwargs) -> bool:
            obj = extract_json_obj(resp)
            if not isinstance(obj, dict):
                return False
            tool_val = obj.get("tool") or obj.get("name") or obj.get("function")
            if tool_val != nm:
                return False
            args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj.get("parameters")
            if not isinstance(args, dict):
                return False
            return all(k in args and _norm(args[k]) == _norm(v) for k, v in kw.items())

        user = f"Tools:\n{json.dumps(it['function'])}\n\nUser request: {q}\n\n{instr}"
        out.append(Probe(it["id"], "BFCL", "BFCL_v3_exec_simple", user, 256, check))
    for it in rng.sample(load("BFCL_v3_irrelevance.json"), n - n // 2):
        q = it["question"][0][0]["content"]

        def check_null(resp: str) -> bool:
            obj = extract_json_obj(resp)
            if obj is None:
                return "null" in resp.lower() or "none" in resp.lower()
            tool_val = obj.get("tool", obj.get("name", None))
            return tool_val in (None, "null", "none", "")

        user = f"Tools:\n{json.dumps(it['function'])}\n\nUser request: {q}\n\n{instr}"
        out.append(Probe(it["id"], "BFCL", "BFCL_v3_irrelevance", user, 256, check_null))
    return out


def build_suite(n: int, seed: int = 42) -> List[Probe]:
    suite: List[Probe] = []
    for builder in (humaneval, mmlu, gsm8k, arc, bfcl):
        suite.extend(builder(n, random.Random(seed)))
    return suite


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def evaluate(label: str, gguf: str, suite: List[Probe], log: logging.Logger) -> Dict[str, Any]:
    per_cat: Dict[str, List[int]] = {}
    logs = []
    t0 = time.time()
    with llama_server(gguf, log) as base:
        for i, p in enumerate(suite):
            resp, dt = chat(base, p.prompt, p.max_tokens)
            try:
                ok = int(bool(p.check(resp)))
            except Exception as exc:  # scorer failure counts as fail, but is recorded
                ok = 0
                resp += f"\n__SCORER_ERROR__ {type(exc).__name__}: {exc}"
            per_cat.setdefault(p.category, []).append(ok)
            logs.append({"id": p.pid, "category": p.category, "source": p.source, "passed": ok,
                         "latency_s": round(dt, 3), "response": resp})
            if (i + 1) % 25 == 0:
                log.info(f"{label} progress {i + 1}/{len(suite)}")
    summary = {}
    for c, v in per_cat.items():
        acc, lo, hi = wilson(sum(v), len(v))
        summary[c] = {"passed": sum(v), "n": len(v), "acc": acc, "ci95": [lo, hi]}
    total = sum(sum(v) for v in per_cat.values())
    acc, lo, hi = wilson(total, len(suite))
    return {"model": label, "gguf": gguf, "overall": {"passed": total, "n": len(suite), "acc": acc, "ci95": [lo, hi]},
            "categories": summary, "elapsed_s": round(time.time() - t0, 1), "logs": logs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A,B,C,A0,B0,C0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, keep only this many probes per category")
    ap.add_argument("--out", default=os.path.join(EXPORTS, "benchmark_v4_results.json"))
    args = ap.parse_args()
    log = build_logger()

    suite = build_suite(args.n)
    if args.smoke:
        kept: Dict[str, int] = {}
        suite = [p for p in suite if kept.setdefault(p.category, 0) < args.smoke and not kept.__setitem__(p.category, kept[p.category] + 1)]
    log.info(f"suite built probes={len(suite)}")

    results: Dict[str, Any] = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            results = json.load(fh)
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        label, gguf = MODEL_REGISTRY[key]
        if label in results:
            log.info(f"skip completed {label}")
            continue
        if not os.path.exists(gguf):
            log.warning(f"missing gguf {gguf}")
            continue
        res = evaluate(label, gguf, suite, log)
        results[label] = res
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        os.replace(tmp, args.out)
        cats = " ".join(f"{c}={s['acc']}%" for c, s in res["categories"].items())
        log.info(f"DONE {label} overall={res['overall']['acc']}% {cats}")


if __name__ == "__main__":
    main()

