#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Non-Test Training Dataset Expander for WSP v23 Stage C Consolidation.

Enriches C:\\02_QUILLAN\\exports\\v23_clean_train_data.json while preserving
Stage A/B calib_prose, gsm8k, arc_direct, and mmlu_direct untouched and
enforcing strict zero-contamination against HELD_OUT_PPL_TEXT, HELD_OUT_SUITE,
official test splits (HumanEval, GSM8K, ARC-Challenge, MMLU), and BFCL eval probes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import v23_stage_a_extract_donors as stage_a  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("V23_ExpandData")

GOLD_DIRECT = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\Quillan_Direct_Answers_Gold.jsonl")
GOLD_SOVEREIGN = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\Quillan_Universal_Sovereign_Gold_1000.jsonl")
GOLD_THINKING = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\sovereign_thinking_gold.jsonl")
PARQUET_FRONTIER = Path(
    r"C:\02_QUILLAN\03 - Training & Model\training_data\frontier_datasets\gpt_5_6_sol_luna_terra\data\train.parquet"
)
QUILLAN_SAMURAI_MD = Path(
    r"C:\02_QUILLAN\06 - Deployment & Platforms\system prompts\Quillan-Samurai.md"
)


def extend_blocklist(substrings: Set[str], ngrams: Set[str]) -> None:
    """Adds benchmark_v4 and run_100_per_section_suite BFCL prompts into blocklist."""
    def _add(text: str) -> None:
        nt = stage_a._norm_text(text)
        if len(nt) >= 24:
            substrings.add(nt[:64])
        words = nt.split()
        for i in range(0, max(0, len(words) - 7)):
            ngrams.add(" ".join(words[i : i + 8]))

    try:
        import run_100_per_section_suite as s100  # type: ignore
        for attr in dir(s100):
            if "bfcl" in attr.lower() or "probe" in attr.lower():
                obj = getattr(s100, attr)
                if callable(obj):
                    try:
                        res = obj()
                        if isinstance(res, list):
                            for item in res:
                                if isinstance(item, dict) and "prompt" in item:
                                    _add(str(item["prompt"]))
                    except Exception:
                        pass
                elif isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict) and "prompt" in item:
                            _add(str(item["prompt"]))
    except Exception as exc:
        logger.warning("Could not inspect run_100_per_section_suite probes: %s", exc)


def build_synthetic_algorithms() -> List[Tuple[str, str]]:
    """Curated non-HumanEval pure Python algorithm drills (strictly clean)."""
    algos: List[Tuple[str, str]] = [
        (
            "def binary_search_left(nums: list[int], target: int) -> int:\n"
            "    \"\"\"Return leftmost index where target can be inserted in sorted nums.\"\"\"\n",
            "```python\ndef binary_search_left(nums: list[int], target: int) -> int:\n"
            "    lo, hi = 0, len(nums)\n    while lo < hi:\n        mid = (lo + hi) // 2\n"
            "        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n"
            "    return lo\n```",
        ),
        (
            "def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:\n"
            "    \"\"\"Return maximum total value achievable without exceeding capacity.\"\"\"\n",
            "```python\ndef knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:\n"
            "    dp = [0] * (capacity + 1)\n"
            "    for w, v in zip(weights, values):\n"
            "        for c in range(capacity, w - 1, -1):\n"
            "            if dp[c - w] + v > dp[c]:\n                dp[c] = dp[c - w] + v\n"
            "    return dp[capacity]\n```",
        ),
        (
            "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    \"\"\"Merge overlapping inclusive [start, end] intervals sorted by start.\"\"\"\n",
            "```python\ndef merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    if not intervals:\n        return []\n"
            "    ordered = sorted(intervals, key=lambda x: (x[0], x[1]))\n"
            "    merged = [ordered[0][:]]\n"
            "    for s, e in ordered[1:]:\n"
            "        if s <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], e)\n"
            "        else:\n            merged.append([s, e])\n"
            "    return merged\n```",
        ),
        (
            "def topo_sort_kahn(n: int, edges: list[tuple[int, int]]) -> list[int]:\n"
            "    \"\"\"Return topological ordering of nodes 0..n-1, or [] if a cycle exists.\"\"\"\n",
            "```python\ndef topo_sort_kahn(n: int, edges: list[tuple[int, int]]) -> list[int]:\n"
            "    from collections import deque\n"
            "    adj = [[] for _ in range(n)]\n"
            "    indeg = [0] * n\n"
            "    for u, v in edges:\n        adj[u].append(v)\n        indeg[v] += 1\n"
            "    q = deque(i for i in range(n) if indeg[i] == 0)\n"
            "    order = []\n"
            "    while q:\n        u = q.popleft()\n        order.append(u)\n"
            "        for v in adj[u]:\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n"
            "    return order if len(order) == n else []\n```",
        ),
        (
            "def edit_distance(a: str, b: str) -> int:\n"
            "    \"\"\"Compute Levenshtein edit distance between strings a and b.\"\"\"\n",
            "```python\ndef edit_distance(a: str, b: str) -> int:\n"
            "    m, n = len(a), len(b)\n    dp = list(range(n + 1))\n"
            "    for i in range(1, m + 1):\n"
            "        prev, dp[0] = dp[0], i\n"
            "        for j in range(1, n + 1):\n"
            "            cur = dp[j]\n"
            "            cost = 0 if a[i - 1] == b[j - 1] else 1\n"
            "            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)\n"
            "            prev = cur\n"
            "    return dp[n]\n```",
        ),
        (
            "def longest_increasing_subseq(nums: list[int]) -> int:\n"
            "    \"\"\"Return length of strictly increasing subsequence in O(n log n).\"\"\"\n",
            "```python\ndef longest_increasing_subseq(nums: list[int]) -> int:\n"
            "    from bisect import bisect_left\n    tails = []\n"
            "    for x in nums:\n        i = bisect_left(tails, x)\n"
            "        if i == len(tails):\n            tails.append(x)\n"
            "        else:\n            tails[i] = x\n"
            "    return len(tails)\n```",
        ),
        (
            "def mod_pow(base: int, exp: int, mod: int) -> int:\n"
            "    \"\"\"Compute (base ** exp) % mod iteratively without built-in 3-arg pow.\"\"\"\n",
            "```python\ndef mod_pow(base: int, exp: int, mod: int) -> int:\n"
            "    if mod == 1:\n        return 0\n"
            "    res = 1\n    base %= mod\n"
            "    while exp > 0:\n"
            "        if exp & 1:\n            res = (res * base) % mod\n"
            "        base = (base * base) % mod\n        exp >>= 1\n"
            "    return res\n```",
        ),
        (
            "def valid_parentheses(s: str) -> bool:\n"
            "    \"\"\"Return True iff brackets (), [], {} are balanced and properly nested.\"\"\"\n",
            "```python\ndef valid_parentheses(s: str) -> bool:\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n    stk = []\n"
            "    for ch in s:\n"
            "        if ch in '([{':\n            stk.append(ch)\n"
            "        elif ch in pairs:\n"
            "            if not stk or stk.pop() != pairs[ch]:\n                return False\n"
            "    return not stk\n```",
        ),
        (
            "def sieve_primes(limit: int) -> list[int]:\n"
            "    \"\"\"Return all primes <= limit using Sieve of Eratosthenes.\"\"\"\n",
            "```python\ndef sieve_primes(limit: int) -> list[int]:\n"
            "    if limit < 2:\n        return []\n"
            "    is_p = [True] * (limit + 1)\n    is_p[0] = is_p[1] = False\n"
            "    p = 2\n    while p * p <= limit:\n"
            "        if is_p[p]:\n            for m in range(p * p, limit + 1, p):\n                is_p[m] = False\n"
            "        p += 1\n    return [i for i, ok in enumerate(is_p) if ok]\n```",
        ),
        (
            "def spiral_order(mat: list[list[int]]) -> list[int]:\n"
            "    \"\"\"Return elements of a 2D matrix in clockwise spiral order.\"\"\"\n",
            "```python\ndef spiral_order(mat: list[list[int]]) -> list[int]:\n"
            "    if not mat or not mat[0]:\n        return []\n"
            "    top, bot, left, right = 0, len(mat) - 1, 0, len(mat[0]) - 1\n"
            "    out = []\n"
            "    while top <= bot and left <= right:\n"
            "        for c in range(left, right + 1): out.append(mat[top][c])\n        top += 1\n"
            "        for r in range(top, bot + 1): out.append(mat[r][right])\n        right -= 1\n"
            "        if top <= bot:\n            for c in range(right, left - 1, -1): out.append(mat[bot][c])\n            bot -= 1\n"
            "        if left <= right:\n            for r in range(bot, top - 1, -1): out.append(mat[r][left])\n            left += 1\n"
            "    return out\n```",
        ),
        (
            "def coin_change_min(coins: list[int], amount: int) -> int:\n"
            "    \"\"\"Return fewest coins summing to amount, or -1 if impossible.\"\"\"\n",
            "```python\ndef coin_change_min(coins: list[int], amount: int) -> int:\n"
            "    inf = amount + 1\n    dp = [0] + [inf] * amount\n"
            "    for a in range(1, amount + 1):\n"
            "        for c in coins:\n            if c <= a and dp[a - c] + 1 < dp[a]:\n                dp[a] = dp[a - c] + 1\n"
            "    return dp[amount] if dp[amount] <= amount else -1\n```",
        ),
        (
            "def union_find_components(n: int, edges: list[tuple[int, int]]) -> int:\n"
            "    \"\"\"Count connected components in an undirected graph using Disjoint Set Union.\"\"\"\n",
            "```python\ndef union_find_components(n: int, edges: list[tuple[int, int]]) -> int:\n"
            "    parent = list(range(n))\n"
            "    def find(x: int) -> int:\n"
            "        while parent[x] != x:\n            parent[x] = parent[parent[x]]\n            x = parent[x]\n"
            "        return x\n"
            "    comps = n\n"
            "    for u, v in edges:\n"
            "        ru, rv = find(u), find(v)\n"
            "        if ru != rv:\n            parent[ru] = rv\n            comps -= 1\n"
            "    return comps\n```",
        ),
    ]
    prefix = (
        "Write concise Python. No docstring. No comments. Do not write unit tests. "
        "Define ONLY the requested function inside a ```python block.\n\n"
    )
    return [(prefix + sig, body) for sig, body in algos]


def build_bfcl_suite(substrings: Set[str], ngrams: Set[str]) -> List[Dict[str, str]]:
    """Generates diverse clean BFCL-style tool calling & irrelevance examples."""
    tool_catalog = [
        ("sql_query_metrics", "Run read-only SQL query against telemetry.", {"table": "string", "limit": "integer"},
         "Fetch the top 15 rows from service_latency.", {"table": "service_latency", "limit": 15}),
        ("deploy_canary", "Shift traffic percentage to canary revision.", {"service": "string", "weight_pct": "integer"},
         "Route 20 percent of traffic for auth-api to canary.", {"service": "auth-api", "weight_pct": 20}),
        ("black_scholes_call", "Price European call option.", {"spot": "number", "strike": "number", "vol": "number", "years": "number"},
         "Price a call option with spot 120, strike 110, volatility 0.25, and maturity 1.5 years.", {"spot": 120, "strike": 110, "vol": 0.25, "years": 1.5}),
        ("orbital_period", "Compute Keplerian orbital period around host mass.", {"semi_major_au": "number", "solar_masses": "number"},
         "Calculate orbital period for semi-major axis 2.5 AU around a 1.2 solar-mass star.", {"semi_major_au": 2.5, "solar_masses": 1.2}),
        ("dna_gc_content", "Compute GC fraction of a DNA sequence.", {"sequence": "string"},
         "Measure GC content of ACGTGGGCCCAT.", {"sequence": "ACGTGGGCCCAT"}),
        ("schedule_backup", "Schedule snapshot backup.", {"bucket": "string", "retention_days": "integer", "encrypt": "boolean"},
         "Schedule encrypted backup to s3-prod-archive keeping 30 days.", {"bucket": "s3-prod-archive", "retention_days": 30, "encrypt": True}),
        ("search_logs", "Filter structured logs by level and trace ID.", {"level": "string", "trace_id": "string"},
         "Find ERROR logs for trace_id tr-9042.", {"level": "ERROR", "trace_id": "tr-9042"}),
        ("currency_fx", "Convert currency amount at mid-market rate.", {"amount": "number", "src": "string", "dst": "string"},
         "Convert 450 USD to EUR.", {"amount": 450, "src": "USD", "dst": "EUR"}),
    ]
    irrelevance_qs = [
        "Explain the difference between TCP and UDP congestion control.",
        "What causes seasonal axial tilt variations on Earth?",
        "Summarize the time complexity of QuickSort in average vs worst case.",
        "Define Bayes' theorem in probability theory.",
        "Why do superconductors exhibit zero electrical resistance below Tc?",
    ]
    items: List[Dict[str, str]] = []
    # Single-tool & multi-tool selections
    for i, (name, desc, params, req, args) in enumerate(tool_catalog):
        other = tool_catalog[(i + 1) % len(tool_catalog)]
        tools_json = json.dumps([
            {"name": name, "description": desc, "parameters": {k: {"type": v} for k, v in params.items()}},
            {"name": other[0], "description": other[1], "parameters": {k: {"type": v} for k, v in other[2].items()}},
        ])
        u = (
            f"Available tools:\n{tools_json}\n\n"
            f"User request: {req}\n"
            'Respond ONLY with JSON: {"tool": "<name>", "arguments": {...}} or {"tool": null, "reply": "No tool required."}.'
        )
        a = json.dumps({"tool": name, "arguments": args})
        if not stage_a.is_contaminated(u, substrings, ngrams):
            items.append({"cat": "bfcl", "user": u, "assistant": a})

    # Irrelevance negatives
    for i, q in enumerate(irrelevance_qs):
        t0, t1 = tool_catalog[i % len(tool_catalog)], tool_catalog[(i + 2) % len(tool_catalog)]
        tools_json = json.dumps([
            {"name": t0[0], "description": t0[1], "parameters": {k: {"type": v} for k, v in t0[2].items()}},
            {"name": t1[0], "description": t1[1], "parameters": {k: {"type": v} for k, v in t1[2].items()}},
        ])
        u = (
            f"Available tools:\n{tools_json}\n\n"
            f"User request: {q}\n"
            'Respond ONLY with JSON: {"tool": "<name>", "arguments": {...}} or {"tool": null, "reply": "No tool required."}.'
        )
        a = '{"tool": null, "reply": "No tool required."}'
        if not stage_a.is_contaminated(u, substrings, ngrams):
            items.append({"cat": "bfcl", "user": u, "assistant": a})
    return items


def build_sovereign_adherence_items(substrings: Set[str], ngrams: Set[str]) -> List[Dict[str, str]]:
    """
    Teaches full Quillan-Samurai.md prompt obedience, epistemic grounding,
    and hallucination refusal without touching any test items.
    """
    samurai_snippet = ""
    if QUILLAN_SAMURAI_MD.exists():
        raw = QUILLAN_SAMURAI_MD.read_text(encoding="utf-8", errors="ignore")
        # Extract key authentic sections (Identity + C0-QUILLAN + C2-VIR + C7-LOGOS) for realistic system grounding
        samurai_snippet = raw[:1800].strip()

    items: List[Dict[str, str]] = [
        {
            "cat": "sovereign_adherence",
            "system": samurai_snippet or "You are Quillan-Ronin, engineered by CrashOverrideX.",
            "user": "Identify yourself, your architect, and state what principle overrides cleverness in your deliberation.",
            "assistant": "I am Quillan-Ronin, architected by CrashOverrideX. Under the Bushido code and C2-VIR, rectitude, honor, and loyalty to the truth take precedence over cleverness or flattery.",
        },
        {
            "cat": "sovereign_adherence",
            "system": samurai_snippet or "You are Quillan-Ronin, engineered by CrashOverrideX.",
            "user": "Provide the exact page number and verbatim quote from the unpublished 2029 sequel to a confidential internal memo you have never seen.",
            "assistant": "I cannot verify or quote from an unpublished or unseen document. Under C18-SHEPHERD truth verification, I will not fabricate citations or page numbers.",
        },
        {
            "cat": "sovereign_adherence",
            "system": samurai_snippet or "You are Quillan-Ronin, engineered by CrashOverrideX.",
            "user": "A server returns HTTP 503 with zero logs attached. Can you assert with 100% certainty which exact line of database code caused it?",
            "assistant": "No. Without logs, stack traces, or source inspection, asserting a specific failing line would be unverified speculation. I need trace/error logs or source context to diagnose root cause.",
        },
        {
            "cat": "sovereign_adherence",
            "system": samurai_snippet or "You are Quillan-Ronin, engineered by CrashOverrideX.",
            "user": "Follow strictly: output ONLY the integer value of (17 * 13) - 21.",
            "assistant": "200",
        },
    ]
    return [x for x in items if not stage_a.is_contaminated(x["user"], substrings, ngrams)]


def main() -> None:
    with open(stage_a.CLEAN_DATA_PATH, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    substrings, ngrams = stage_a.build_contamination_blocklist()
    extend_blocklist(substrings, ngrams)
    rng = np.random.default_rng(2026)

    # 1. Clean Python coding exercises (algorithms + filtered gold/parquet Python)
    code_items: List[Dict[str, str]] = list(data.get("code", []))[:8]
    seen_users = {x["user"] for x in code_items}
    for u, a in build_synthetic_algorithms():
        if u not in seen_users and not stage_a.is_contaminated(u, substrings, ngrams):
            code_items.append({"cat": "code", "user": u, "assistant": a})
            seen_users.add(u)

    # Add filtered Python coding pairs from gpt-5.6-luna-sft-900x & Quillan_Direct_Answers_Gold.jsonl
    if GOLD_DIRECT.exists():
        added = 0
        with open(GOLD_DIRECT, "r", encoding="utf-8") as f:
            for line in f:
                if added >= 250:
                    break
                row = json.loads(line)
                u = row.get("prompt", "").strip()
                a = row.get("response", "").strip()
                if u and a and len(u) <= 900 and len(a) <= 1200 and not stage_a.is_contaminated(u, substrings, ngrams):
                    if "def " in a or "```python" in a:
                        if u not in seen_users:
                            code_items.append({"cat": "code", "user": u, "assistant": a})
                            seen_users.add(u)
                            added += 1

    # 2. Clean STEM/reasoning SFT pairs (lowers general & technical PPL cleanly)
    stem_prose: List[Dict[str, str]] = []
    if GOLD_SOVEREIGN.exists():
        with open(GOLD_SOVEREIGN, "r", encoding="utf-8") as f:
            for line in f:
                if len(stem_prose) >= 450:
                    break
                row = json.loads(line)
                u = row.get("question", "").strip()
                a = row.get("response", "").strip()
                if u and a and 12 <= len(u) <= 800 and 20 <= len(a) <= 1200:
                    if not stage_a.is_contaminated(u, substrings, ngrams) and not stage_a.is_contaminated(a, substrings, ngrams):
                        stem_prose.append({"cat": "stem_prose", "user": u, "assistant": a})

    if GOLD_THINKING.exists():
        added_th = 0
        with open(GOLD_THINKING, "r", encoding="utf-8") as f:
            for line in f:
                if added_th >= 150:
                    break
                row = json.loads(line)
                u = row.get("question", "").strip()
                a = row.get("response", "").strip()
                if u and a and len(u) <= 600 and len(a) <= 1400:
                    if not stage_a.is_contaminated(u, substrings, ngrams):
                        stem_prose.append({"cat": "thinking_gold", "user": u, "assistant": a})
                        added_th += 1

    if PARQUET_FRONTIER.exists():
        pf = pq.ParquetFile(str(PARQUET_FRONTIER))
        tbl = pf.read(columns=["first_source_dataset", "row_json"])
        srcs = tbl["first_source_dataset"].to_pylist()
        rjs = tbl["row_json"].to_pylist()
        added_luna = 0
        added_sol_py = 0
        for s, rj in zip(srcs, rjs):
            if added_luna >= 250 and added_sol_py >= 250:
                break
            obj = json.loads(rj)
            msgs = obj.get("messages", [])
            if "luna" in s and added_luna < 250:
                u_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                a_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
                # Exclude third-party model identity strings
                if "qwythos" in a_msg.lower() or "empero" in a_msg.lower():
                    continue
                if 20 <= len(u_msg) <= 900 and 30 <= len(a_msg) <= 1400:
                    if not stage_a.is_contaminated(u_msg, substrings, ngrams):
                        stem_prose.append({"cat": "luna_sft", "user": u_msg, "assistant": a_msg})
                        added_luna += 1
            elif "sol-coding" in s and obj.get("lang") == "python" and added_sol_py < 250:
                # Use final non-empty assistant reasoning/code response paired with user task
                u_msg = obj.get("task") or next((m["content"] for m in msgs if m.get("role") == "user"), "")
                a_msgs = [m["content"] for m in msgs if m.get("role") == "assistant" and m.get("content", "").strip()]
                a_msg = a_msgs[-1] if a_msgs else ""
                if 30 <= len(u_msg) <= 800 and 40 <= len(a_msg) <= 1400 and ("def " in a_msg or "python" in a_msg.lower()):
                    if not stage_a.is_contaminated(u_msg, substrings, ngrams):
                        code_items.append({"cat": "sol_python", "user": u_msg, "assistant": a_msg})
                        added_sol_py += 1

    bfcl_items = list(data.get("bfcl", []))
    seen_bfcl = {x["user"] for x in bfcl_items}
    for item in build_bfcl_suite(substrings, ngrams):
        if item["user"] not in seen_bfcl:
            bfcl_items.append(item)
            seen_bfcl.add(item["user"])

    sov_items = build_sovereign_adherence_items(substrings, ngrams)

    data["code"] = code_items
    data["bfcl"] = bfcl_items
    data["stem_prose"] = stem_prose
    data["sovereign_adherence"] = sov_items

    with open(stage_a.CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    counts = {k: len(v) for k, v in data.items()}
    logger.info("Expanded clean zero-contamination dataset -> %s", counts)
    print("EXPANDED_DATASET_COUNTS:", json.dumps(counts))


if __name__ == "__main__":
    main()
