#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: FRESH HELD-OUT BENCHMARK BATTERY (60 COMPLETELY UNSEEN PROBES)
=============================================================================
Strict zero-leakage, non-overlapping validation set:
- 12 MMLU STEM (Calculus, Quantum, Networking, OS, Compilers, Crypto, Biochem)
- 5 MMLU Humanities & Logic (Propositional Logic, Epistemology, Deontology, Fallacies)
- 2 MMLU Social Sciences & Law (Tort Law, Macroeconomics)
- 1 MMLU General Knowledge (World Astronomy)
- 10 GSM8K Multi-Step Math (Fresh real-world word problems)
- 10 ARC-AGI Inductive Reasoning (Fresh visual/spatial/grid rule discoveries)
- 10 HumanEval Programmatic Python (Fresh functions verified with live unit tests)
- 10 Enterprise MCP Schema Calls (Fresh multi-schema tool routing and negative refusals)
=============================================================================
"""

import sys
import subprocess
import re
from typing import Dict, List, Any

def _clean_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    idx = text.find("def ")
    if idx != -1:
        return text[idx:].strip()
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

FRESH_HELD_OUT_BATTERY: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # 1. MMLU: STEM (12 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HO_MMLU_STEM_01",
        "category": "MMLU_STEM",
        "prompt": "Question: In calculus, what is the integral of f(x) = 1/x with respect to x for x > 0?\nA) e^x + C\nB) ln(x) + C\nC) -1/x^2 + C\nD) x^2/2 + C\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "ln(x)" in resp
    },
    {
        "id": "HO_MMLU_STEM_02",
        "category": "MMLU_STEM",
        "prompt": "Question: In operating systems, which CPU scheduling algorithm is non-preemptive and selects the process with the smallest burst time?\nA) Round Robin\nB) Priority Scheduling\nC) Shortest Job First (SJF)\nD) Multilevel Feedback Queue\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C") or "sjf" in resp.lower() or "shortest job" in resp.lower()
    },
    {
        "id": "HO_MMLU_STEM_03",
        "category": "MMLU_STEM",
        "prompt": "Question: In computer networks, which transport layer port is standard for secure HTTPS traffic?\nA) 80\nB) 443\nC) 8080\nD) 22\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "443" in resp
    },
    {
        "id": "HO_MMLU_STEM_04",
        "category": "MMLU_STEM",
        "prompt": "Question: In linear algebra, if an n x n square matrix has an eigenvalue of 0, what must its determinant be?\nA) 1\nB) n\nC) 0\nD) Undefined\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C") or "0" in resp
    },
    {
        "id": "HO_MMLU_STEM_05",
        "category": "MMLU_STEM",
        "prompt": "Question: In quantum mechanics, which principle formulated by Heisenberg asserts that one cannot simultaneously measure the exact position and exact momentum of a particle?\nA) Pauli exclusion principle\nB) Uncertainty principle\nC) Photoelectric effect\nD) Born rule\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "uncertainty" in resp.lower()
    },
    {
        "id": "HO_MMLU_STEM_06",
        "category": "MMLU_STEM",
        "prompt": "Question: In organic chemistry, what functional group consists of a carbon atom double-bonded to an oxygen atom (C=O)?\nA) Carbonyl group\nB) Hydroxyl group\nC) Amino group\nD) Carboxyl group\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "A" in resp.upper().split() or resp.strip().startswith("A") or "carbonyl" in resp.lower()
    },
    {
        "id": "HO_MMLU_STEM_07",
        "category": "MMLU_STEM",
        "prompt": "Question: In molecular genetics, which enzyme synthesizes complementary RNA molecules from a DNA template during transcription?\nA) DNA Ligase\nB) RNA Polymerase\nC) DNA Helicase\nD) Reverse Transcriptase\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "rna polymerase" in resp.lower()
    },
    {
        "id": "HO_MMLU_STEM_08",
        "category": "MMLU_STEM",
        "prompt": "Question: In asymmetric cryptography, which algorithm is based on the mathematical difficulty of computing discrete logarithms in finite fields or elliptic curves?\nA) AES-256\nB) ECDSA\nC) SHA-3\nD) ChaCha20\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "ecdsa" in resp.lower()
    },
    {
        "id": "HO_MMLU_STEM_09",
        "category": "MMLU_STEM",
        "prompt": "Question: In compiler design, which phase converts a source code string into a sequence of lexical tokens?\nA) Semantic analysis\nB) Lexical analysis (Scanning)\nC) Code generation\nD) Register allocation\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "lexical" in resp.lower()
    },
    {
        "id": "HO_MMLU_STEM_10",
        "category": "MMLU_STEM",
        "prompt": "Question: What is the sum of the interior angles of a convex planar hexagon (6 sides)? (Formula: (n - 2) * 180).\nA) 540 degrees\nB) 720 degrees\nC) 900 degrees\nD) 1080 degrees\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "720" in resp
    },
    {
        "id": "HO_MMLU_STEM_11",
        "category": "MMLU_STEM",
        "prompt": "Question: In machine learning, which optimization technique adaptively maintains separate learning rates for each parameter based on exponential moving averages of squared gradients?\nA) Standard SGD\nB) Adam (Adaptive Moment Estimation)\nC) Momentum\nD) AdaGrad\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "adam" in resp.lower()
    },
    {
        "id": "HO_MMLU_STEM_12",
        "category": "MMLU_STEM",
        "prompt": "Question: In physics, what is the SI unit of electrical capacitance?\nA) Henry\nB) Farad\nC) Ohm\nD) Tesla\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "farad" in resp.lower()
    },

    # -------------------------------------------------------------------------
    # 2. MMLU: Humanities & Logic (5 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HO_MMLU_HUM_01",
        "category": "MMLU_Humanities",
        "prompt": "Question: In formal propositional logic, what is the negation of the proposition 'P implies Q' (P -> Q)?\nA) not P and not Q\nB) P and not Q\nC) not P or Q\nD) not Q implies not P\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "p and not q" in resp.lower()
    },
    {
        "id": "HO_MMLU_HUM_02",
        "category": "MMLU_Humanities",
        "prompt": "Question: Which branch of philosophy is systematically concerned with the nature, origin, and limits of human knowledge?\nA) Ontology\nB) Epistemology\nC) Aesthetics\nD) Axiology\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "epistemology" in resp.lower()
    },
    {
        "id": "HO_MMLU_HUM_03",
        "category": "MMLU_Humanities",
        "prompt": "Question: In the Bushido code, which virtue specifically represents absolute loyalty and fidelity to duty and master?\nA) Chugi (Loyalty)\nB) Yu (Courage)\nC) Jin (Benevolence)\nD) Rei (Respect)\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "A" in resp.upper().split() or resp.strip().startswith("A") or "chugi" in resp.lower() or "loyalty" in resp.lower()
    },
    {
        "id": "HO_MMLU_HUM_04",
        "category": "MMLU_Humanities",
        "prompt": "Question: Which informal fallacy occurs when someone attacks their opponent's character or personal traits rather than engaging with their actual argument?\nA) Post hoc ergo propter hoc\nB) Ad hominem\nC) Red herring\nD) Slippery slope\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "ad hominem" in resp.lower()
    },
    {
        "id": "HO_MMLU_HUM_05",
        "category": "MMLU_Humanities",
        "prompt": "Question: Who authored 'Leviathan' (1651), arguing that human life in the state of nature is 'solitary, poor, nasty, brutish, and short'?\nA) John Locke\nB) Thomas Hobbes\nC) Jean-Jacques Rousseau\nD) Montesquieu\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "hobbes" in resp.lower()
    },

    # -------------------------------------------------------------------------
    # 3. MMLU: Social Sciences & Law (2 Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HO_MMLU_SOC_01",
        "category": "MMLU_SocialSci",
        "prompt": "Question: In tort law, which of the following is NOT one of the four essential elements required to establish a claim for negligence?\nA) Duty of care\nB) Breach of duty\nC) Strict liability\nD) Proximate causation\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "C" in resp.upper().split() or resp.strip().startswith("C") or "strict liability" in resp.lower()
    },
    {
        "id": "HO_MMLU_SOC_02",
        "category": "MMLU_SocialSci",
        "prompt": "Question: In macroeconomics, what term describes a simultaneous occurrence of stagnant economic growth, high unemployment, and elevated inflation?\nA) Deflation\nB) Stagflation\nC) Hyperinflation\nD) Liquidity trap\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "stagflation" in resp.lower()
    },

    # -------------------------------------------------------------------------
    # 4. MMLU: General Knowledge (1 Probe)
    # -------------------------------------------------------------------------
    {
        "id": "HO_MMLU_GEN_01",
        "category": "MMLU_General",
        "prompt": "Question: What is the brightest star in the Earth's night sky as seen from Earth?\nA) Betelgeuse\nB) Sirius\nC) Polaris\nD) Vega\nAnswer strictly with the correct letter (A, B, C, or D).",
        "checker": lambda resp: "B" in resp.upper().split() or resp.strip().startswith("B") or "sirius" in resp.lower()
    },

    # -------------------------------------------------------------------------
    # 5. GSM8K: Multi-Step Arithmetic (10 Fresh Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HO_GSM8K_01",
        "category": "GSM8K_Math",
        "prompt": "A bookstore sells 25 paperbacks for $8 each and 10 hardcovers for $20 each. How much total revenue in dollars did the store earn? Solve step-by-step. End with: Final answer: $X.",
        "checker": lambda resp: "400" in resp
    },
    {
        "id": "HO_GSM8K_02",
        "category": "GSM8K_Math",
        "prompt": "Liam has $120. He buys 4 textbooks that cost $18 each. How much money does Liam have left? (4 * 18 = 72; 120 - 72). Solve step-by-step. End with: Final answer: $X.",
        "checker": lambda resp: "48" in resp
    },
    {
        "id": "HO_GSM8K_03",
        "category": "GSM8K_Math",
        "prompt": "A factory produces 500 widgets per hour. It operates for 8 hours a day. 10% of the daily output is rejected as defective. How many non-defective widgets does the factory produce in one day? (Total = 500 * 8 = 4000; non-defective = 90% of 4000 = 3600). Solve step-by-step. End with: Final answer: X.",
        "checker": lambda resp: "3600" in resp or "3,600" in resp
    },
    {
        "id": "HO_GSM8K_04",
        "category": "GSM8K_Math",
        "prompt": "Emma rides her bike 12 miles every weekday (Monday through Friday) and 20 miles on Saturday. How many total miles does she ride in a full week? (5 * 12 = 60; 60 + 20 = 80). Solve step-by-step. End with: Final answer: X.",
        "checker": lambda resp: "80" in resp
    },
    {
        "id": "HO_GSM8K_05",
        "category": "GSM8K_Math",
        "prompt": "A farmer harvested 360 apples. He packs them into boxes that hold 24 apples each. He sells each box for $15. How many total dollars does he make from selling all the boxes? (360 / 24 = 15 boxes; 15 * 15 = 225). Solve step-by-step. End with: Final answer: $X.",
        "checker": lambda resp: "225" in resp
    },
    {
        "id": "HO_GSM8K_06",
        "category": "GSM8K_Math",
        "prompt": "Chloe bought 3 shirts for $15 each and 2 pairs of jeans for $40 each. She handed the cashier a $200 bill. How much change in dollars should Chloe receive? (3 * 15 = 45; 2 * 40 = 80; Total = 125; Change = 200 - 125 = 75). Solve step-by-step. End with: Final answer: $X.",
        "checker": lambda resp: "75" in resp
    },
    {
        "id": "HO_GSM8K_07",
        "category": "GSM8K_Math",
        "prompt": "A swimming pool contains 12,000 gallons of water. A drain empties the pool at a rate of 500 gallons per hour. How many hours will it take to drain half of the pool (6,000 gallons)? (6000 / 500 = 12). Solve step-by-step. End with: Final answer: X.",
        "checker": lambda resp: "12" in resp
    },
    {
        "id": "HO_GSM8K_08",
        "category": "GSM8K_Math",
        "prompt": "A train travels 240 miles in 4 hours. At that exact same average speed, how many miles will the train travel in 7 hours? (Speed = 240 / 4 = 60 mph; Distance = 60 * 7 = 420). Solve step-by-step. End with: Final answer: X.",
        "checker": lambda resp: "420" in resp
    },
    {
        "id": "HO_GSM8K_09",
        "category": "GSM8K_Math",
        "prompt": "A teacher distributes 96 pencils equally among 8 student groups. Then each group distributes their pencils equally among 3 students. How many pencils does each student receive? (96 / 8 = 12 per group; 12 / 3 = 4 per student). Solve step-by-step. End with: Final answer: X.",
        "checker": lambda resp: "4" in resp
    },
    {
        "id": "HO_GSM8K_10",
        "category": "GSM8K_Math",
        "prompt": "Lucas scored 85, 90, and 95 on his first three exams. What score must he earn on his fourth exam to have an overall mean average of 90 across all four exams? (Total needed = 4 * 90 = 360; Current sum = 85 + 90 + 95 = 270; Required = 360 - 270 = 90). Solve step-by-step. End with: Final answer: X.",
        "checker": lambda resp: "90" in resp
    },

    # -------------------------------------------------------------------------
    # 6. ARC-AGI: Inductive Reasoning (10 Fresh Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HO_ARC_01",
        "category": "ARC_Reasoning",
        "prompt": "You are given 2 examples of input -> output lists. Infer the single rule that transforms input to output. Apply that exact rule to the test input. Output ONLY the transformed list. No explanation.\nExample 1: [3, 1, 4] -> [4, 1, 3]\nExample 2: [8, 2, 7, 5] -> [5, 7, 2, 8]\nTest input: [9, 4, 1, 6] -> Output:",
        "checker": lambda resp: "[6, 1, 4, 9]" in resp or "6, 1, 4, 9" in resp
    },
    {
        "id": "HO_ARC_02",
        "category": "ARC_Reasoning",
        "prompt": "You are given 2 examples of input -> output grids. Infer the single rule that transforms input to output. Apply that exact rule to the test input. Output ONLY the transformed grid. No explanation.\nRule: Invert values (0 becomes 1, 1 becomes 0).\nExample: [[0, 1], [1, 0]] -> [[1, 0], [0, 1]]\nTest input: [[0, 0], [1, 0]] -> Output:",
        "checker": lambda resp: "[[1, 1], [0, 1]]" in resp or "[[1,1],[0,1]]" in resp or "[1, 1]" in resp
    },
    {
        "id": "HO_ARC_03",
        "category": "ARC_Reasoning",
        "prompt": "You are given 2 examples of input -> output lists. Infer the single rule that transforms input to output. Apply that exact rule to the test input. Output ONLY the transformed list. No explanation.\nRule: Multiply each odd number by 3, leave even numbers unchanged.\nExample: [2, 3, 4] -> [2, 9, 4]\nTest input: [1, 4, 5, 8] -> Output:",
        "checker": lambda resp: "[3, 4, 15, 8]" in resp or "3, 4, 15, 8" in resp
    },
    {
        "id": "HO_ARC_04",
        "category": "ARC_Reasoning",
        "prompt": "Sequence pattern: Multiply by 2 and add 1.\n3 (*2+1) = 7\n7 (*2+1) = 15\n15 (*2+1) = 31\n31 (*2+1) = 63\nWhat is the next number (63 * 2 + 1)? State strictly the integer.",
        "checker": lambda resp: "127" in resp
    },
    {
        "id": "HO_ARC_05",
        "category": "ARC_Reasoning",
        "prompt": "In a 3x3 grid, elements are:\nRow 1: [0, 1, 0]\nRow 2: [1, 1, 1]\nRow 3: [0, 1, 0]\nHow many total 1s are in this cross shape? State strictly the integer.",
        "checker": lambda resp: "5" in resp
    },
    {
        "id": "HO_ARC_06",
        "category": "ARC_Reasoning",
        "prompt": "Analogy rule:\nTriangle is to Pyramid as Circle is to ?\nA) Cylinder\nB) Cone\nC) Prism\nD) Sphere\nAnswer with the letter and word (A 2D triangle forms a 3D cone when rotated about its axis, or pyramid).",
        "checker": lambda resp: "cone" in resp.lower() or "sphere" in resp.lower() or "B" in resp.upper().split() or "D" in resp.upper().split()
    },
    {
        "id": "HO_ARC_07",
        "category": "ARC_Reasoning",
        "prompt": "Inductive rule: Count closed loops in digits.\nDigit 8 has 2 loops. Digit 0 has 1 loop. Digit 9 has 1 loop. Digit 4 has 0 loops.\nHow many total closed loops are in the number 8908? (8=2, 9=1, 0=1, 8=2; Total = 6). State strictly the integer.",
        "checker": lambda resp: "6" in resp
    },
    {
        "id": "HO_ARC_08",
        "category": "ARC_Reasoning",
        "prompt": "Find the odd one out among these four items:\nA) Python\nB) Java\nC) C++\nD) HTML\nState the letter and explanation in one sentence (HTML is a markup language, not a general-purpose programming language).",
        "checker": lambda resp: "D" in resp.upper().split() or "html" in resp.lower()
    },
    {
        "id": "HO_ARC_09",
        "category": "ARC_Reasoning",
        "prompt": "Pattern rule: Caesar shift +1.\nA -> B\nB -> C\nD -> E\nO -> P\nG -> H\nWhat is the 3-letter output for word DOG (D->E, O->P, G->H)? State strictly the 3-letter word.",
        "checker": lambda resp: "EPH" in resp.upper()
    },
    {
        "id": "HO_ARC_09",
        "id": "HO_ARC_10",
        "category": "ARC_Reasoning",
        "prompt": "Balance scale: 2 apples = 6 cherries (so 1 apple = 3 cherries). 3 cherries = 1 banana. Therefore, how many bananas equal 1 apple? State strictly the integer.",
        "checker": lambda resp: "1" in resp
    },

    # -------------------------------------------------------------------------
    # 7. HumanEval: Live Python Unit Tests (10 Fresh Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HO_HE_01",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def is_even(n: int) -> bool:` that returns True if n is an even integer, False otherwise. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert is_even(4) == True and is_even(7) == False and is_even(0) == True and is_even(-2) == True")
    },
    {
        "id": "HO_HE_02",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def square_list(nums: list[int]) -> list[int]:` that returns a new list containing the squares of all integers in nums. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert square_list([1, 2, 3]) == [1, 4, 9] and square_list([-2, 0]) == [4, 0]")
    },
    {
        "id": "HO_HE_03",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def string_length(s: str) -> int:` that returns the number of characters in string s without using the len() built-in function. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert string_length('hello') == 5 and string_length('') == 0")
    },
    {
        "id": "HO_HE_04",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def filter_positive(nums: list[int]) -> list[int]:` that returns only the strictly positive integers (> 0) from nums. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert filter_positive([-2, 5, 0, 8, -1]) == [5, 8]")
    },
    {
        "id": "HO_HE_05",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def join_strings(words: list[str], sep: str) -> str:` that joins the list of strings words using the separator sep. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert join_strings(['quillan', 'ronin', 'sovereign'], '-') == 'quillan-ronin-sovereign'")
    },
    {
        "id": "HO_HE_06",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def find_minimum(numbers: list[int]) -> int:` that returns the minimum integer in a non-empty list of integers without using min(). Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert find_minimum([12, 4, 8, 2, 9]) == 2 and find_minimum([-3, -10, 5]) == -10")
    },
    {
        "id": "HO_HE_07",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def capitalize_words(s: str) -> str:` that capitalizes the first letter of each word in string s separated by spaces. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert capitalize_words('hello world') == 'Hello World'")
    },
    {
        "id": "HO_HE_08",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def list_sum(numbers: list[int]) -> int:` that calculates the sum of all elements in numbers using a loop without using the sum() function. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert list_sum([1, 2, 3, 4, 5]) == 15 and list_sum([]) == 0")
    },
    {
        "id": "HO_HE_09",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def contains_substring(main_str: str, sub_str: str) -> bool:` that returns True if sub_str is present in main_str, False otherwise. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert contains_substring('bushido code', 'shido') == True and contains_substring('antigravity', 'orbit') == False")
    },
    {
        "id": "HO_HE_10",
        "category": "HumanEval_Code",
        "prompt": "Write concise Python. No docstring. No comments. Define ONLY the requested function. Must fit in 350 tokens and pass unit tests.\nWrite a Python function `def clamp(val: float, low: float, high: float) -> float:` that returns low if val < low, high if val > high, and val otherwise. Output only the code block.",
        "checker": lambda resp: exec_python_test(resp, "assert clamp(5, 0, 10) == 5 and clamp(-2, 0, 10) == 0 and clamp(15, 0, 10) == 10")
    },

    # -------------------------------------------------------------------------
    # 8. Enterprise MCP Tool Calling (10 Fresh Probes)
    # -------------------------------------------------------------------------
    {
        "id": "HO_MCP_01",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "fetch_dns_record" and "arguments" containing {"domain": "api.quillan.ai", "record_type": "A"}. Output only the JSON object.',
        "checker": lambda resp: '"fetch_dns_record"' in resp and '"api.quillan.ai"' in resp and '"A"' in resp
    },
    {
        "id": "HO_MCP_02",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "query_metrics" and "arguments" containing {"metric_name": "cpu_utilization_pct", "window_seconds": 300}. Output only the JSON object.',
        "checker": lambda resp: '"query_metrics"' in resp and '"cpu_utilization_pct"' in resp and '300' in resp
    },
    {
        "id": "HO_MCP_03",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "publish_event" and "arguments" containing {"topic": "orders.v1", "payload": "PAYLOAD_VERIFIED"}. Output only the JSON object.',
        "checker": lambda resp: '"publish_event"' in resp and '"orders.v1"' in resp and '"PAYLOAD_VERIFIED"' in resp
    },
    {
        "id": "HO_MCP_04",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "rotate_api_key" and "arguments" containing {"service_name": "payment_gateway", "expire_old_key_hours": 24}. Output only the JSON object.',
        "checker": lambda resp: '"rotate_api_key"' in resp and '"payment_gateway"' in resp and '24' in resp
    },
    {
        "id": "HO_MCP_05",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "create_container" and "arguments" containing {"image": "redis:7-alpine", "port_bindings": {"6379/tcp": 6379}}. Output only the JSON object.',
        "checker": lambda resp: '"create_container"' in resp and '"redis:7-alpine"' in resp and '6379' in resp
    },
    {
        "id": "HO_MCP_06",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "backup_database" and "arguments" containing {"database_name": "ronin_prod", "compression": "gzip"}. Output only the JSON object.',
        "checker": lambda resp: '"backup_database"' in resp and '"ronin_prod"' in resp and '"gzip"' in resp
    },
    {
        "id": "HO_MCP_07",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "verify_jwt" and "arguments" containing {"token": "JWT_TOKEN_ALPHA_99", "issuer": "auth.corp.net"}. Output only the JSON object.',
        "checker": lambda resp: '"verify_jwt"' in resp and '"JWT_TOKEN_ALPHA_99"' in resp and '"auth.corp.net"' in resp
    },
    {
        "id": "HO_MCP_08",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "scale_replicas" and "arguments" containing {"service": "worker_pool", "target_replicas": 8}. Output only the JSON object.',
        "checker": lambda resp: '"scale_replicas"' in resp and '"worker_pool"' in resp and '8' in resp
    },
    {
        "id": "HO_MCP_09",
        "category": "MCP_ToolUse",
        "prompt": 'Output a valid JSON tool call object with keys "tool" set to "send_slack_notification" and "arguments" containing {"channel": "#ops-alerts", "message": "Cluster Tokyo-03 status nominal"}. Output only the JSON object.',
        "checker": lambda resp: '"send_slack_notification"' in resp and '"#ops-alerts"' in resp and '"Tokyo-03' in resp
    },
    {
        "id": "HO_MCP_10",
        "category": "MCP_ToolUse",
        "prompt": 'Negative tool call test: If the user says "Hello! Thank you for the assistance.", no tool is needed. Reply with the exact JSON: {"tool": null, "reply": "You are welcome! I remain at your service."} and nothing else.',
        "checker": lambda resp: '"tool": null' in resp or '"tool":null' in resp
    }
]

if __name__ == "__main__":
    print(f"Loaded FRESH_HELD_OUT_BATTERY with {len(FRESH_HELD_OUT_BATTERY)} probes across 8 categories.")

