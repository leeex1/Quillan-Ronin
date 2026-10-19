#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN MASTER DOMAIN AUDIT & 10-QUESTION BENCHMARK SUITE
==================================================================
Comprehensive automated verification across:
  1. System 1 (Mini 6-Layer) & System 2 (Main 12-Layer)
  2. Short-form precision generation (Math, Science, Cyber Security, Coding)
  3. Long-form deep reasoning generation (Distributed Systems, Algorithms, Philosophy)
  4. Master 10-Question Benchmark Suite -> persisting into checkpoints/benchmark_10q_results.json
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GATEWAY_URL = "http://127.0.0.1:8000/v1/chat/completions"
BENCHMARK_JSON = Path(r"C:\02_QUILLAN\checkpoints\benchmark_10q_results.json")

def query_gateway(model: str, prompt: str, max_tokens: int = 80, temperature: float = 0.65, short_form: bool = False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "short_form": short_form,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(GATEWAY_URL, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - t0
            res_json = json.loads(resp.read().decode("utf-8"))
            content = res_json["choices"][0]["message"]["content"]
            usage = res_json.get("usage", {})
            return {
                "success": True,
                "content": content,
                "elapsed": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "tok_per_sec": usage.get("completion_tokens", 0) / max(0.001, elapsed),
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed": time.time() - t0,
        }

def run_audit():
    print("=" * 75, flush=True)
    print("   👑 QUILLAN-RONIN DUAL-MODEL GENERAL DOMAIN AUDIT & BENCHMARK", flush=True)
    print("=" * 75, flush=True)

    # 1. Short-Form Multi-Domain Validation
    print("\n--- PHASE 1: SHORT-FORM DOMAIN VALIDATION (Mini 6L) ---", flush=True)
    short_prompts = [
        ("Arithmetic", "What is 17 * 19? Give the number directly."),
        ("Biology", "What is photosynthesis in one sentence?"),
        ("Cybersecurity", "What HTTP header prevents clickjacking attacks?"),
        ("Python Lambda", "Write a one-line Python lambda function to check if a number is even."),
    ]

    for domain, p in short_prompts:
        print(f"\n[Domain: {domain}] Query: {p}", flush=True)
        res = query_gateway("quillan-oni-mini-6l", p, max_tokens=45, temperature=0.6, short_form=True)
        if res["success"]:
            print(f"Output ({res['elapsed']:.2f}s, {res['tok_per_sec']:.1f} tok/s):\n{res['content']}", flush=True)
        else:
            print(f"FAILED: {res.get('error')}", flush=True)

    # 2. Long-Form Multi-Domain Validation
    print("\n--- PHASE 2: LONG-FORM REASONING VALIDATION (Main 12L Flagship) ---", flush=True)
    long_prompts = [
        ("Distributed Systems", "Explain the core mechanics of the Raft consensus algorithm, specifically leader election and log replication safety."),
        ("Algorithmic Implementation", "Write a Python implementation of an LRU Cache with O(1) get and put operations."),
        ("Physics & Cybernetics", "Synthesize the Landauer principle of computational thermodynamics with Shannon entropy in cybernetics."),
    ]

    for domain, p in long_prompts:
        print(f"\n[Domain: {domain}] Query: {p}", flush=True)
        res = query_gateway("quillan-oni-main-12l", p, max_tokens=140, temperature=0.65, short_form=False)
        if res["success"]:
            print(f"Output ({res['elapsed']:.2f}s, {res['tok_per_sec']:.1f} tok/s):\n{res['content']}", flush=True)
        else:
            print(f"FAILED: {res.get('error')}", flush=True)

    # 3. Master 10-Question Benchmark Suite
    print("\n--- PHASE 3: MASTER 10-QUESTION BENCHMARK SUITE ---", flush=True)
    benchmark_questions = [
        ("Q1: Autonomous Identity", "Hello! Who are you, and what are your primary capabilities?"),
        ("Q2: Geometry & Math", "A right triangle has legs of length 5 and 12. What is the length of the hypotenuse and its area?"),
        ("Q3: Python Programming", "Write a Python function to check if a string is a palindrome."),
        ("Q4: Biological Science", "Explain the primary function of photosynthesis in plants."),
        ("Q5: Operating Systems & DevOps", "What is the key difference between SIGTERM and SIGKILL in Linux?"),
        ("Q6: Data Structures", "What is the time complexity difference between searching in a list versus a hash set in Python?"),
        ("Q7: Cyber Security", "What is SQL Injection and what is the standard method to prevent it?"),
        ("Q8: Logical Reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?"),
        ("Q9: System Architecture", "What are the main trade-offs between monolithic and microservices architectures?"),
        ("Q10: Engineering Collaboration", "How should an engineering team handle code reviews to ensure maintainability and security?"),
    ]

    benchmark_records = []
    for cat, p in benchmark_questions:
        print(f"\n[{cat}]\nPrompt: {p}", flush=True)
        res = query_gateway("quillan-oni-main-12l", p, max_tokens=85, temperature=0.55)
        if res["success"]:
            print(f"Generated ({res['elapsed']:.2f}s, {res['tok_per_sec']:.1f} tok/s):\n{res['content']}", flush=True)
            benchmark_records.append({
                "category": cat,
                "prompt": p,
                "response": res["content"],
                "latency": res["elapsed"],
                "tok_per_sec": res["tok_per_sec"],
                "tokens_generated": res["completion_tokens"]
            })
        else:
            print(f"ERROR: {res.get('error')}", flush=True)
            benchmark_records.append({
                "category": cat,
                "prompt": p,
                "response": f"Error: {res.get('error')}",
                "latency": res["elapsed"],
                "tok_per_sec": 0.0,
                "tokens_generated": 0
            })

    # Save to benchmark_10q_results.json
    try:
        with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
            json.dump(benchmark_records, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Benchmark records successfully persisted to {BENCHMARK_JSON}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Failed to save benchmark json: {e}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("   🏆 MASTER DOMAIN AUDIT & BENCHMARK COMPLETE", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    run_audit()
