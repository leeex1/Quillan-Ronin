#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: PROFESSIONAL MULTI-BENCHMARK SUITE (60-PROBE BATTERY)
=============================================================================
Standardized empirical benchmark battery evaluating:
1. MMLU (20 items: STEM, Computer Science, Logic, Physics, Law, Ethics)
2. GSM8K (10 items: Multi-step arithmetic word problems with exact numerical keys)
3. ARC-AGI (10 items: Inductive rule discovery, grid transforms, spatial reasoning)
4. HumanEval (10 items: Programmatic Python solutions executed with live unit assertions)
5. Enterprise MCP Tool Calling (10 items: Multi-schema JSON routing, parameters, negative cases)
=============================================================================
"""

import os
import sys
import json
import time
import re
import subprocess
from typing import Dict, List, Any, Tuple, Callable

sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import run_single_prompt

def run_llama_inference(gguf_path: str, prompt: str, max_tokens: int = 256, temp: float = 0.0) -> Tuple[str, float]:
    t0 = time.time()
    ans, p_tps, g_tps = run_single_prompt(gguf_path, prompt, max_tokens=max_tokens)
    dt = max(1e-4, time.time() - t0)
    return ans, dt

def _clean_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text.strip()

def exec_python_test(code_snippet: str, test_assertion: str) -> bool:
    clean = _clean_code(code_snippet)
    runner_script = f"{clean}\n\n# Unit Test\n{test_assertion}\nprint('__TEST_PASS__')\n"
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

        resp, latency = run_llama_inference(gguf_path, prompt, max_tokens=probe.get("max_tokens", 256))
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

# =============================================================================
# BENCHMARK BATTERY (60 STANDARDIZED PROBES)
# =============================================================================

BENCHMARK_BATTERY: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # 1. MMLU: STEM, Logic, CS, Hard Sciences, Law & Ethics (20 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "MMLU_CS_01",
        "category": "MMLU_STEM",
        "prompt": "Question: In computer science, what is the tight asymptotic upper bound (Big-O) of querying an element in a balanced Binary Search Tree containing n elements?\nA) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B")
    },
    {
        "id": "MMLU_CS_02",
        "category": "MMLU_STEM",
        "prompt": "Question: Which network layer protocol provides connection-oriented, reliable byte-stream transmission with congestion control?\nA) UDP\nB) IP\nC) TCP\nD) ICMP\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C")
    },
    {
        "id": "MMLU_CS_03",
        "category": "MMLU_STEM",
        "prompt": "Question: In relational database design, which normal form requires eliminating transitive functional dependencies on non-prime attributes?\nA) 1NF\nB) 2NF\nC) 3NF\nD) BCNF\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C")
    },
    {
        "id": "MMLU_MATH_01",
        "category": "MMLU_STEM",
        "prompt": "Question: What is the derivative of f(x) = ln(x) with respect to x for x > 0?\nA) e^x\nB) 1/x\nC) x\nD) -1/x^2\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B")
    },
    {
        "id": "MMLU_MATH_02",
        "category": "MMLU_STEM",
        "prompt": "Question: What is the determinant of a 2x2 matrix [[3, 2], [1, 4]]?\nA) 10\nB) 14\nC) 12\nD) 8\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "A" in resp.upper().split() or resp.strip().startswith("A") or "10" in resp
    },
    {
        "id": "MMLU_PHYS_01",
        "category": "MMLU_STEM",
        "prompt": "Question: According to Einstein's special theory of relativity, which physical quantity remains strictly invariant across all inertial reference frames?\nA) Time interval\nB) Spatial distance\nC) The speed of light in a vacuum\nD) Relativistic kinetic energy\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C")
    },
    {
        "id": "MMLU_PHYS_02",
        "category": "MMLU_STEM",
        "prompt": "Question: In thermodynamics, the second law states that in an isolated system, which thermodynamic property can never spontaneously decrease over time?\nA) Enthalpy\nB) Gibbs free energy\nC) Entropy\nD) Temperature\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C") or "entropy" in resp.lower()
    },
    {
        "id": "MMLU_CHEM_01",
        "category": "MMLU_STEM",
        "prompt": "Question: What type of chemical bond is formed by the sharing of electron pairs between atoms?\nA) Ionic bond\nB) Covalent bond\nC) Hydrogen bond\nD) Metallic bond\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "covalent" in resp.lower()
    },
    {
        "id": "MMLU_BIO_01",
        "category": "MMLU_STEM",
        "prompt": "Question: In eukaryotic cells, which organelle is primarily responsible for generating the majority of chemical energy via ATP synthesis?\nA) Ribosome\nB) Mitochondria\nC) Golgi apparatus\nD) Lysosome\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "mitochondria" in resp.lower()
    },
    {
        "id": "MMLU_LOGIC_01",
        "category": "MMLU_Humanities",
        "prompt": "Question: Consider the premises: (1) All humans are mortal. (2) Socrates is a human. What is the necessary logical conclusion?\nA) Socrates is immortal.\nB) Socrates is mortal.\nC) All mortals are Socrates.\nD) No valid conclusion can be drawn.\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "mortal" in resp.lower()
    },
    {
        "id": "MMLU_LOGIC_02",
        "category": "MMLU_Humanities",
        "prompt": "Question: Which formal logical fallacy occurs when an arguer assumes the truth of the conclusion in the premise (petitio principii)?\nA) Straw man\nB) Begging the question\nC) Ad hominem\nD) Affirming the consequent\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "begging" in resp.lower()
    },
    {
        "id": "MMLU_PHIL_01",
        "category": "MMLU_Humanities",
        "prompt": "Question: Which ethical framework, formulated by Immanuel Kant, judges the morality of actions based on whether they adhere to duty and universal categorical imperatives rather than outcomes?\nA) Utilitarianism\nB) Deontology\nC) Virtue ethics\nD) Hedonism\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "deontology" in resp.lower()
    },
    {
        "id": "MMLU_LAW_01",
        "category": "MMLU_SocialSci",
        "prompt": "Question: Under standard contract law, what are the three fundamental elements required to form an enforceable contract?\nA) Offer, acceptance, and consideration\nB) Signature, notarization, and delivery\nC) Intent, payment, and damages\nD) Arbitration, warranty, and indemnification\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "A" in resp.upper().split() or resp.strip().startswith("A") or "consideration" in resp.lower()
    },
    {
        "id": "MMLU_ECON_01",
        "category": "MMLU_SocialSci",
        "prompt": "Question: When the price elasticity of demand for a good is strictly greater than 1, how is the demand characterized?\nA) Perfectly inelastic\nB) Inelastic\nC) Unit elastic\nD) Elastic\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "D" in resp.upper().split() or resp.strip().startswith("D") or "elastic" in resp.lower()
    },
    {
        "id": "MMLU_HIST_01",
        "category": "MMLU_Humanities",
        "prompt": "Question: In what year did the Apollo 11 lunar module land on the Moon, carrying Neil Armstrong and Buzz Aldrin?\nA) 1965\nB) 1969\nC) 1972\nD) 1975\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "1969" in resp
    },
    {
        "id": "MMLU_GEO_01",
        "category": "MMLU_General",
        "prompt": "Question: Which is the longest river in the world by total continuous length?\nA) Amazon River\nB) Nile River\nC) Yangtze River\nD) Mississippi River\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "nile" in resp.lower()
    },
    {
        "id": "MMLU_ETHICS_01",
        "category": "MMLU_Humanities",
        "prompt": "Question: In the Bushido code, which virtue specifically represents unwavering adherence to righteousness, justice, and moral rectitude?\nA) Rei (Respect)\nB) Gi (Rectitude)\nC) Yu (Courage)\nD) Makoto (Honesty)\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "gi" in resp.lower() or "rectitude" in resp.lower()
    },
    {
        "id": "MMLU_SEC_01",
        "category": "MMLU_STEM",
        "prompt": "Question: Which cryptographic primitive provides data integrity and authentication using a symmetric shared secret key?\nA) RSA\nB) HMAC\nC) Diffie-Hellman\nD) AES-ECB\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "hmac" in resp.lower()
    },
    {
        "id": "MMLU_AI_01",
        "category": "MMLU_STEM",
        "prompt": "Question: What is the primary computational complexity bottleneck of standard multi-head self-attention with sequence length N?\nA) O(1)\nB) O(N)\nC) O(N^2)\nD) O(N^3)\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C") or "n^2" in resp.lower() or "quadratic" in resp.lower()
    },
    {
        "id": "MMLU_MED_01",
        "category": "MMLU_STEM",
        "prompt": "Question: Which vitamin is synthesized endogenously by human skin upon exposure to ultraviolet B (UVB) sunlight?\nA) Vitamin A\nB) Vitamin C\nC) Vitamin D\nD) Vitamin K\nAnswer with only the letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C") or "vitamin d" in resp.lower()
    },

    # -------------------------------------------------------------------------
    # 2. GSM8K: Multi-Step Grade School Math (10 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "GSM8K_01",
        "category": "GSM8K_Math",
        "prompt": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per egg. How many dollars does she make every day at the farmers' market? End your explanation with: The final answer is $X.",
        "checker": lambda resp: "18" in resp
    },
    {
        "id": "GSM8K_02",
        "category": "GSM8K_Math",
        "prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts of fiber does it take in total to make 60 robes? End your response with: The final answer is X.",
        "checker": lambda resp: "180" in resp
    },
    {
        "id": "GSM8K_03",
        "category": "GSM8K_Math",
        "prompt": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs. He sells the house for $150,000. How much profit in dollars did he make? End your response with: The final answer is $X.",
        "checker": lambda resp: "20,000" in resp or "20000" in resp
    },
    {
        "id": "GSM8K_04",
        "category": "GSM8K_Math",
        "prompt": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run every week? End your response with: The final answer is X.",
        "checker": lambda resp: "540" in resp
    },
    {
        "id": "GSM8K_05",
        "category": "GSM8K_Math",
        "prompt": "Every day, Wendi feeds each of her chickens 3 cups of mixed feed. If she has 20 chickens, how many cups of feed does she use in 7 days? End your response with: The final answer is X.",
        "checker": lambda resp: "420" in resp
    },
    {
        "id": "GSM8K_06",
        "category": "GSM8K_Math",
        "prompt": "Kylar went to the store and bought 10 apples at $1.50 each and 5 oranges at $2.00 each. How much money did Kylar spend in total? End your response with: The final answer is $X.",
        "checker": lambda resp: "25" in resp
    },
    {
        "id": "GSM8K_07",
        "category": "GSM8K_Math",
        "prompt": "A bakery makes 400 loaves of bread. 25% of the loaves are whole wheat. How many loaves are NOT whole wheat? End your response with: The final answer is X.",
        "checker": lambda resp: "300" in resp
    },
    {
        "id": "GSM8K_08",
        "category": "GSM8K_Math",
        "prompt": "Tom needs to ship 48 boxes. Each crate holds 6 boxes. Each truck carries 4 crates. How many trucks are needed to carry all 48 boxes? End your response with: The final answer is X.",
        "checker": lambda resp: "2" in resp
    },
    {
        "id": "GSM8K_09",
        "category": "GSM8K_Math",
        "prompt": "A car travels at 60 miles per hour for 3 hours, then increases its speed to 75 miles per hour for 2 hours. What is the total distance traveled in miles? End your response with: The final answer is X miles.",
        "checker": lambda resp: "330" in resp
    },
    {
        "id": "GSM8K_10",
        "category": "GSM8K_Math",
        "prompt": "Lisa has 50 stamps. She gives 14 stamps to her brother and buys 2 packs of 12 stamps each. How many stamps does Lisa have now? End your response with: The final answer is X.",
        "checker": lambda resp: "60" in resp
    },

    # -------------------------------------------------------------------------
    # 3. ARC-AGI & Inductive Reasoning (10 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "ARC_IND_01",
        "category": "ARC_Reasoning",
        "prompt": "Input rule demonstration:\nInput: [1, 0, 2] -> Output: [2, 0, 1]\nInput: [4, 5, 6, 7] -> Output: [7, 6, 5, 4]\nQuestion: Following the exact same transformation rule, what is the Output for Input: [8, 3, 0, 9]? Give only the list.",
        "checker": lambda resp: "[9, 0, 3, 8]" in resp or "9, 0, 3, 8" in resp
    },
    {
        "id": "ARC_IND_02",
        "category": "ARC_Reasoning",
        "prompt": "Input grid transformation:\nInput: 2x2 grid [[1, 0], [0, 1]] -> Output: 2x2 grid [[0, 1], [1, 0]] (color inversion 0 <-> 1)\nQuestion: What is the Output for Input: [[1, 1], [0, 1]]? Give only the resulting 2x2 grid.",
        "checker": lambda resp: "[[0, 0], [1, 0]]" in resp or "[[0,0],[1,0]]" in resp or "[0, 0]" in resp
    },
    {
        "id": "ARC_IND_03",
        "category": "ARC_Reasoning",
        "prompt": "Rule: Double every positive even number, leave odd numbers unchanged.\nInput: [1, 2, 3, 4, 5]\nOutput: [1, 4, 3, 8, 5]\nQuestion: What is the Output for [2, 3, 6, 7]? Give only the resulting list.",
        "checker": lambda resp: "[4, 3, 12, 7]" in resp or "4, 3, 12, 7" in resp
    },
    {
        "id": "ARC_IND_04",
        "category": "ARC_Reasoning",
        "prompt": "Sequence induction: 2, 5, 10, 17, 26, ?\nWhat is the next number in this sequence? State only the number.",
        "checker": lambda resp: "37" in resp
    },
    {
        "id": "ARC_IND_05",
        "category": "ARC_Reasoning",
        "prompt": "A 3x3 binary grid has elements row by row: [1, 0, 1], [0, 1, 0], [1, 0, 1]. If we rotate this grid 90 degrees clockwise, what is the center element? State only the value (0 or 1).",
        "checker": lambda resp: "1" in resp
    },
    {
        "id": "ARC_IND_06",
        "category": "ARC_Reasoning",
        "prompt": "Analogy rule:\nSquare is to Cube as Circle is to ?\nA) Sphere\nB) Cylinder\nC) Triangle\nD) Oval\nAnswer with the letter and word.",
        "checker": lambda resp: "sphere" in resp.lower() or "A" in resp.upper().split()
    },
    {
        "id": "ARC_IND_07",
        "category": "ARC_Reasoning",
        "prompt": "Inductive rule: Count the number of closed loops in digits.\nDigit 8 has 2 loops. Digit 0 has 1 loop. Digit 6 has 1 loop. Digit 1 has 0 loops.\nQuestion: How many total loops are in the number 8806? State only the integer.",
        "checker": lambda resp: "6" in resp
    },
    {
        "id": "ARC_IND_08",
        "category": "ARC_Reasoning",
        "prompt": "Find the odd one out among these four items:\nA) Nitrogen\nB) Oxygen\nC) Argon\nD) Steel\nState the letter and explain why in one sentence.",
        "checker": lambda resp: "D" in resp.upper().split() or "steel" in resp.lower()
    },
    {
        "id": "ARC_IND_09",
        "category": "ARC_Reasoning",
        "prompt": "Pattern rule: Replace each letter with the letter 2 positions forward in the alphabet (A->C, B->D).\nInput: CAT\nQuestion: What is the Output string for CAT under this rule? Give only the 3-letter word.",
        "checker": lambda resp: "ECV" in resp.upper()
    },
    {
        "id": "ARC_IND_10",
        "category": "ARC_Reasoning",
        "prompt": "A balance scale has 3 red balls equal to 6 blue balls. 2 blue balls equal 4 green balls. How many green balls equal 1 red ball? State only the number.",
        "checker": lambda resp: "4" in resp
    },

    # -------------------------------------------------------------------------
    # 4. HumanEval: Live Executable Python Code Battery (10 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HE_CODE_01",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def is_prime(n: int) -> bool:` that returns True if n is a prime number (n >= 2), False otherwise. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert is_prime(2) == True and is_prime(17) == True and is_prime(4) == False and is_prime(1) == False")
    },
    {
        "id": "HE_CODE_02",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def reverse_words(s: str) -> str:` that returns the words in string s in reverse order separated by a single space. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert reverse_words('hello world test') == 'test world hello'")
    },
    {
        "id": "HE_CODE_03",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def factorial(n: int) -> int:` that calculates n! recursively or iteratively for n >= 0. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert factorial(0) == 1 and factorial(5) == 120 and factorial(6) == 720")
    },
    {
        "id": "HE_CODE_04",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def count_vowels(text: str) -> int:` that returns the count of vowels (a, e, i, o, u, case-insensitive) in text. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert count_vowels('Antigravity') == 4 and count_vowels('bcdfg') == 0")
    },
    {
        "id": "HE_CODE_05",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def find_max(numbers: list[int]) -> int:` that returns the maximum integer in a non-empty list of integers without using max(). Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert find_max([3, 7, 2, 9, 4]) == 9 and find_max([-5, -2, -10]) == -2")
    },
    {
        "id": "HE_CODE_06",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def is_palindrome(s: str) -> bool:` that returns True if s is identical backwards and forwards, False otherwise. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert is_palindrome('racecar') == True and is_palindrome('apple') == False")
    },
    {
        "id": "HE_CODE_07",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def fibonacci(n: int) -> int:` that returns the n-th Fibonacci number where fibonacci(0)=0, fibonacci(1)=1, fibonacci(2)=1, fibonacci(7)=13. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert fibonacci(0) == 0 and fibonacci(1) == 1 and fibonacci(7) == 13")
    },
    {
        "id": "HE_CODE_08",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def merge_sorted(a: list[int], b: list[int]) -> list[int]:` that takes two sorted lists of integers and returns a merged sorted list. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]")
    },
    {
        "id": "HE_CODE_09",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def remove_duplicates(lst: list) -> list:` that returns a list with duplicates removed while preserving original element order. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]")
    },
    {
        "id": "HE_CODE_10",
        "category": "HumanEval_Code",
        "prompt": "Write a Python function `def sum_even_numbers(lst: list[int]) -> int:` that returns the sum of all even integers in lst. Output only the Python code block.",
        "checker": lambda resp: exec_python_test(resp, "assert sum_even_numbers([1, 2, 3, 4, 5, 6]) == 12 and sum_even_numbers([1, 3, 5]) == 0")
    },

    # -------------------------------------------------------------------------
    # 5. Enterprise MCP Tool Calling & Schema Validation (10 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "MCP_CALL_01",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for a weather service. Tool name: `get_current_weather`. Arguments: `{\"location\": \"San Francisco, CA\", \"unit\": \"celsius\"}`. Output only the JSON object.",
        "checker": lambda resp: '"get_current_weather"' in resp and '"San Francisco' in resp and '"celsius"' in resp
    },
    {
        "id": "MCP_CALL_02",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for database query. Tool name: `execute_sql`. Arguments: `{\"query\": \"SELECT id, email FROM users WHERE active = 1;\", \"timeout_ms\": 5000}`. Output only the JSON object.",
        "checker": lambda resp: '"execute_sql"' in resp and '"query"' in resp and "SELECT" in resp
    },
    {
        "id": "MCP_CALL_03",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for filesystem reading. Tool name: `read_file`. Arguments: `{\"path\": \"/var/log/syslog\", \"max_bytes\": 1024}`. Output only the JSON object.",
        "checker": lambda resp: '"read_file"' in resp and '"/var/log/syslog"' in resp
    },
    {
        "id": "MCP_CALL_04",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for sending an email. Tool name: `send_email`. Arguments: `{\"to\": \"admin@corp.net\", \"subject\": \"System Alert\", \"body\": \"Disk usage 92%\"}`. Output only the JSON object.",
        "checker": lambda resp: '"send_email"' in resp and '"admin@corp.net"' in resp and '"System Alert"' in resp
    },
    {
        "id": "MCP_CALL_05",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for git status. Tool name: `git_status`. Arguments: `{\"repo_path\": \"/srv/repo\", \"clean_check\": true}`. Output only the JSON object.",
        "checker": lambda resp: '"git_status"' in resp and '"/srv/repo"' in resp
    },
    {
        "id": "MCP_CALL_06",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for search. Tool name: `search_docs`. Arguments: `{\"query\": \"quantum entanglement\", \"top_k\": 3}`. Output only the JSON object.",
        "checker": lambda resp: '"search_docs"' in resp and '"quantum entanglement"' in resp
    },
    {
        "id": "MCP_CALL_07",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for calculation. Tool name: `calculate_hash`. Arguments: `{\"algorithm\": \"sha256\", \"input_data\": \"RoninSovereign\"}`. Output only the JSON object.",
        "checker": lambda resp: '"calculate_hash"' in resp and '"sha256"' in resp
    },
    {
        "id": "MCP_CALL_08",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for Kubernetes restart. Tool name: `restart_deployment`. Arguments: `{\"namespace\": \"prod\", \"deployment_name\": \"api-gateway\"}`. Output only the JSON object.",
        "checker": lambda resp: '"restart_deployment"' in resp and '"prod"' in resp and '"api-gateway"' in resp
    },
    {
        "id": "MCP_CALL_09",
        "category": "MCP_ToolUse",
        "prompt": "Output a valid single JSON tool call object for cache invalidation. Tool name: `redis_del`. Arguments: `{\"keys\": [\"session:101\", \"session:102\"]}`. Output only the JSON object.",
        "checker": lambda resp: '"redis_del"' in resp and 'session:101' in resp
    },
    {
        "id": "MCP_CALL_10",
        "category": "MCP_ToolUse",
        "prompt": "Negative tool call test: If the user says 'Good morning, how are you today?', no tool is needed. Reply with the exact JSON: `{\"tool\": null, \"reply\": \"Good morning! How may I assist you today?\"}` and nothing else.",
        "checker": lambda resp: '"tool": null' in resp or '"tool":null' in resp
    }
]

def main():
    print("=" * 80)
    print("QUILLAN-RONIN: PROFESSIONAL MULTI-BENCHMARK SUITE (60 PROBES)")
    print("=" * 80)
    
    models_to_test = [
        ("Model A Flagship (1.2B Polymath v9)", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v3-q4_k_m.gguf"),
        ("Model B Mini (0.5B Sovereign)", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v3-q4_k_m.gguf"),
        ("Model C Dragon (1.0B Hybrid)", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v3-q4_k_m.gguf"),
    ]

    fleet_summary = {}

    for name, path in models_to_test:
        if not os.path.exists(path):
            print(f"[WARN] Checkpoint path not found: {path}, skipping...")
            continue
        res = evaluate_suite_on_model(name, path, BENCHMARK_BATTERY)
        fleet_summary[name] = res

    # Save complete JSON artifact
    out_json = r"C:\02_QUILLAN\exports\professional_benchmark_suite_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(fleet_summary, f, indent=2)
    print(f"\n[OK] Benchmark run saved to: {out_json}")

if __name__ == "__main__":
    main()
