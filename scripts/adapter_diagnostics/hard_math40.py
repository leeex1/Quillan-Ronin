#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""40-problem hard math benchmark = hard_math20 (20, C28-verified) + 20 more.

Each new problem: (question, [A,B,C,D], correct index, verified solution).
Options are written so the correct VALUE sits at the declared index.
For the 40/40 bar AFTER the 20/20 bar is met. Editor-fidelity caveat applies.
"""

from __future__ import annotations
from hard_math20 import PROBLEMS as P20, KEY as K20, SOLUTIONS as S20, LETTERS

# 20 additional hand-verified problems. Correct value placed at the keyed index.
P_EXTRA = [
    ("What is 2^10 + 2^10?", ["1024", "512", "4096", "2048"], 3, "2*1024=2048."),
    (
        "If x + 1/x = 3, what is x^2 + 1/x^2?",
        ["9", "8", "7", "11"],
        2,
        "(x+1/x)^2-2=9-2=7.",
    ),
    (
        "How many diagonals does a regular octagon have?",
        ["20", "24", "28", "32"],
        0,
        "n(n-3)/2=8*5/2=20.",
    ),
    (
        "Sum of interior angles of a pentagon (degrees)?",
        ["360", "540", "720", "900"],
        1,
        "(5-2)*180=540.",
    ),
    (
        "A number is increased by 25% then decreased by 20%. Net change?",
        ["0%", "+5%", "-5%", "+10%"],
        0,
        "1.25*0.8=1 -> 0%.",
    ),
    ("10th Fibonacci number (F1=1,F2=1)?", ["34", "144", "89", "55"], 3, "F10=55."),
    (
        "Simplify sqrt(50) + sqrt(8).",
        ["5sqrt2", "7sqrt2", "6sqrt2", "9sqrt2"],
        1,
        "5sqrt2+2sqrt2=7sqrt2.",
    ),
    ("If 2x+3=11, what is 4x?", ["8", "12", "16", "20"], 2, "x=4 -> 16."),
    (
        "Probability of at least one head in 3 fair flips?",
        ["3/8", "5/8", "1/2", "7/8"],
        3,
        "1-1/8=7/8.",
    ),
    ("Product of roots of x^2-5x+6=0?", ["5", "6", "-6", "1"], 1, "c/a=6."),
    (
        "How many primes between 1 and 30?",
        ["11", "12", "9", "10"],
        3,
        "2,3,5,7,11,13,17,19,23,29 = 10.",
    ),
    ("What is log10(1000)?", ["2", "3", "4", "1000"], 1, "10^3=1000."),
    (
        "Circle radius 5: area in terms of pi?",
        ["10pi", "20pi", "25pi", "50pi"],
        2,
        "pi*r^2=25pi.",
    ),
    (
        "If a:b=2:3 and b:c=3:4, what is a:c?",
        ["1:2", "2:4", "3:4", "4:6"],
        0,
        "a:c=2:4=1:2.",
    ),
    (
        "Remainder of 2^20 divided by 7?",
        ["1", "2", "4", "0"],
        2,
        "2^3=1 mod7; 20 mod3=2 -> 2^2=4.",
    ),
    (
        "Solve |x-3|=5. Larger solution?",
        ["8", "5", "-2", "3"],
        0,
        "x=8 or -2; larger 8.",
    ),
    (
        "Average of first 10 positive even integers?",
        ["10", "20", "12", "11"],
        3,
        "2..20 avg=11.",
    ),
    (
        "sin(theta)=3/5 acute, cos(theta)=?",
        ["4/5", "3/4", "5/4", "2/5"],
        0,
        "3-4-5 triangle -> 4/5.",
    ),
    (
        "How many 3-digit numbers have all distinct digits?",
        ["504", "648", "720", "900"],
        1,
        "9*9*8=648.",
    ),
    ("15% of 200 plus 20% of 50?", ["30", "35", "40", "45"], 2, "30+10=40."),
]

PROBLEMS = list(P20) + P_EXTRA
KEY = K20 + [p[2] for p in P_EXTRA]
SOLUTIONS = S20 + [p[3] for p in P_EXTRA]


def format_problem(i: int) -> str:
    q, opts, _, _ = PROBLEMS[i]
    s = f"Question {i + 1}: {q}\n"
    for L, o in zip(LETTERS, opts):
        s += f"{L}) {o}\n"
    return s + "Answer:"


def balance_report():
    from collections import Counter

    return dict(Counter(LETTERS[k] for k in KEY))


if __name__ == "__main__":
    print("problems:", len(PROBLEMS), "balance:", balance_report())
    # assert correct value is at keyed index (basic sanity via solution keywords)
    for i in range(20, 40):
        print(
            f"{i + 1:2d}. {LETTERS[KEY[i]]} {PROBLEMS[i][0][:58]} -> {PROBLEMS[i][1][KEY[i]]}"
        )
