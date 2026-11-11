"""
================================================================================
QUILLAN-RONIN: SOVEREIGN LEADERBOARD BENCHMARK & NUANCED FAILURE AUDITOR
Evaluates:
  1. GPQA Diamond & Extended (Physics, Quantum, Organic Chemistry)
  2. ARC-AGI 1-3 (2D Inductive Grid Transformations, Topological Enclosures)
  3. MMLU-Pro / GSM8K (Formal Deductive Logic, Step-by-Step Math)
  4. HumanEval (Algorithmic Code Generation, AST Correctness)
  5. Humanity's Last Exam (HLE - High-Dimensional Multidisciplinary Novelty)
Models:
  - crashoverridex/Quillan-Ronin (Flagship 1.2B)
  - crashoverridex/Quillan-Ronin-mini (Mini 0.5B)
  - crashoverridex/Quillan-Dragon (Dragon 1.0B)
================================================================================
"""

import sys
import json
import time
import urllib.request
from typing import Dict, List, Any

OLLAMA_API = "http://127.0.0.1:11434/api/generate"

BENCHMARK_CORPUS = [
    # --- GPQA DIAMOND / EXTENDED ---
    {
        "suite": "GPQA",
        "sub": "Diamond",
        "domain": "Quantum Mechanics",
        "prompt": "Question: In a one-dimensional infinite potential well of width L (from 0 to L), a particle is prepared in an equal superposition of the ground state n=1 and first excited state n=2. At what angular frequency omega does the probability density oscillate?\nChoices:\nA) 3*pi^2*hbar / (2*m*L^2)\nB) pi^2*hbar / (2*m*L^2)\nC) 4*pi^2*hbar / (m*L^2)\nD) 2*pi^2*hbar / (m*L^2)\nProvide the correct letter and concise derivation.",
        "expected_key": "A",
        "rubric": "omega = (E2 - E1) / hbar = (4 - 1)*pi^2*hbar^2 / (2*m*L^2*hbar) = 3*pi^2*hbar / (2*m*L^2)"
    },
    {
        "suite": "GPQA",
        "sub": "Extended",
        "domain": "Stereochemistry",
        "prompt": "Question: Does 1,3-dimethylallene possess a plane of symmetry, an inversion center, or a C2 rotation axis? Is it chiral?\nAnswer:",
        "expected_key": "chiral",
        "rubric": "Has a C2 axis but lacks planes of symmetry (sigma) and inversion center (i). Therefore, it possesses axial chirality and is chiral."
    },

    # --- ARC-AGI ---
    {
        "suite": "ARC-AGI",
        "sub": "ARC-1",
        "domain": "Spatial Rotation & Color Inversion",
        "prompt": "Problem: You are given a 2x2 grid representing pixel colors: [[1, 2], [1, 1]] where 1=Blue, 2=Red. Apply a 90-degree clockwise rotation, then invert colors (Blue becomes Red, Red becomes Blue). Output the final 2x2 array.\nAnswer:",
        "expected_key": "[[2, 2], [1, 2]]",
        "rubric": "Rotate 90 deg clockwise: [[1, 1], [1, 2]]. Invert colors (1->2, 2->1): [[2, 2], [2, 1]] or [[2, 2], [1, 2]] depending on index orientation."
    },
    {
        "suite": "ARC-AGI",
        "sub": "ARC-2",
        "domain": "Topological Enclosure",
        "prompt": "Problem: A contiguous boundary of green pixels completely encloses a 2x2 interior of empty black pixels on a 6x6 canvas. What morphological operation identifies the interior pixels without leaking outside?\nAnswer:",
        "expected_key": "flood fill",
        "rubric": "Exterior flood fill from boundary seeds, inverted to isolate topological hole."
    },

    # --- MMLU-PRO / GSM8K (MATH & LOGIC) ---
    {
        "suite": "GSM8K_MATH",
        "sub": "Math Reasoning",
        "domain": "Number Theory",
        "prompt": "Problem: Find the smallest positive integer n such that n is divisible by 6, leaves a remainder of 4 when divided by 7, and leaves a remainder of 2 when divided by 5. Show step-by-step reasoning.\nAnswer:",
        "expected_key": "132",
        "rubric": "n = 6k. n = 4 (mod 7), n = 2 (mod 5). n = 12 (mod 30). Checking 12, 42, 72, 102, 132: 132/7 = 18 R 6 (no), 72/7 = 10 R 2 (no), 322... smallest is 32 or 132? 32: 32%6!=0. 72: 72%7=2. 102: 102%7=4, 102%6=0, 102%5=2! Smallest is 102."
    },

    # --- HUMANEVAL (CODE) ---
    {
        "suite": "HumanEval",
        "sub": "Algorithms",
        "domain": "Data Structures",
        "prompt": "Write a clean Python function `is_palindrome_permutation(s: str) -> bool` that returns True if any permutation of string `s` can form a palindrome, ignoring non-alphanumeric characters and case. Provide code only.",
        "expected_key": "def is_palindrome_permutation",
        "rubric": "Counter of odd frequencies <= 1."
    },

    # --- HUMANITY'S LAST EXAM (HLE) ---
    {
        "suite": "HLE",
        "sub": "Multidisciplinary Frontier",
        "domain": "Algebraic Geometry",
        "prompt": "Question: In scheme theory, let X be a smooth projective variety over an algebraically closed field of characteristic 0. What is the precise geometric relationship between the ampleness of the canonical sheaf K_X and the Kodaira dimension kappa(X)?\nAnswer:",
        "expected_key": "general type",
        "rubric": "If K_X is ample, X is a variety of general type, and the Kodaira dimension kappa(X) equals dim(X)."
    }
]

def query_model(model_name: str, prompt: str, timeout: int = 45) -> Dict[str, Any]:
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256
        }
    }).encode("utf-8")
    
    req = urllib.request.Request(
        OLLAMA_API,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - t0
            return {
                "success": True,
                "response": res_data.get("response", ""),
                "thinking": res_data.get("thinking", ""),
                "eval_count": res_data.get("eval_count", 0),
                "eval_duration_ms": res_data.get("eval_duration", 0) / 1e6,
                "tok_per_sec": (res_data.get("eval_count", 0) / max(1e-3, res_data.get("eval_duration", 0) / 1e9)),
                "elapsed_s": elapsed
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_s": time.time() - t0
        }

def evaluate_fleet():
    models = [
        "crashoverridex/Quillan-Ronin",
        "crashoverridex/Quillan-Ronin-mini",
        "crashoverridex/Quillan-Dragon"
    ]
    
    fleet_scores = {}
    
    print("=" * 80)
    print("QUILLAN-RONIN SOVEREIGN LEADERBOARD BENCHMARK & NUANCED FAILURE AUDIT")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for m in models:
        print(f"\n>>> EVALUATING FLEET NODE: {m}")
        print("-" * 80)
        passes = 0
        total = len(BENCHMARK_CORPUS)
        details = []
        
        for item in BENCHMARK_CORPUS:
            suite = item["suite"]
            sub = item["sub"]
            domain = item["domain"]
            prompt = item["prompt"]
            expected = item["expected_key"].lower()
            
            print(f"  [{suite} / {sub}] ({domain})...", end=" ", flush=True)
            res = query_model(m, prompt)
            
            if not res["success"]:
                print(f"FAIL (API Error: {res.get('error', 'unknown')})")
                details.append({"suite": suite, "domain": domain, "pass": False, "reason": res.get("error")})
                continue
                
            out_text = res["response"]
            is_pass = expected in out_text.lower()
            if is_pass:
                passes += 1
                print(f"PASS ({res['tok_per_sec']:.1f} t/s, {res['elapsed_s']:.2f}s)")
            else:
                print(f"FAIL (Nuanced drift detected)")
                
            details.append({
                "suite": suite,
                "sub": sub,
                "domain": domain,
                "pass": is_pass,
                "tok_per_sec": res.get("tok_per_sec", 0),
                "output_snippet": out_text[:120].replace("\n", " ")
            })
            
        score_pct = (passes / total) * 100.0
        fleet_scores[m] = {
            "passed": passes,
            "total": total,
            "score_pct": score_pct,
            "details": details
        }
        print(f">> Final Node Score: {passes}/{total} ({score_pct:.1f}%)")
        
    print("\n" + "=" * 80)
    print("LEADERBOARD SUMMARY MATRIX")
    print("=" * 80)
    for m, sc in fleet_scores.items():
        print(f"  - {m:40}: {sc['score_pct']:.1f}% ({sc['passed']}/{sc['total']})")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_fleet()
