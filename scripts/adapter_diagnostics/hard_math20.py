#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20 hard math problems with a LOCKED, hand-verified answer key (balanced A/B/C/D).

Multiple-choice A/B/C/D so scoring is a single forward pass (no slow autoregressive
generation). Every answer is verified by the closed-form solution in `solutions`.
Balanced so no option letter is favored. Used for the requested accuracy/WSP loop.

NOTE (honesty): scoring 20/20 after WSP on THIS fixed set demonstrates that closed-form
weight-space programming can install the 20 specific Q->answer associations. It is NOT
evidence of frontier reasoning or out-of-distribution math ability. A held-out split is
reported separately so the distinction is explicit.
"""

from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

# Each entry: (question, [optA,optB,optC,optD], correct_index(0-3), closed_form_solution)
PROBLEMS = [
    (
        "What is the remainder when 7^100 is divided by 13?",
        ["1", "3", "9", "12"],
        2,
        "7^12=1 (mod13) by Fermat; 100 mod 12=4; 7^4=2401=9 (mod13).",
    ),
    (
        "How many distinct real solutions does x^3-3x=1 have?",
        ["0", "1", "2", "3"],
        3,
        "g(x)=x^3-3x-1; local max g(-1)=1>0, local min g(1)=-3<0 => 3 real roots.",
    ),
    (
        "Bag: 5 red, 7 blue. Two drawn without replacement. P(both red)?",
        ["5/33", "7/33", "10/33", "2/11"],
        0,
        "C(5,2)/C(12,2)=10/66=5/33.",
    ),
    (
        "Sum of integers 1..100 divisible by 3 or 5?",
        ["2318", "2403", "2638", "2418"],
        3,
        "s3=1683 + s5=1050 - s15=315 = 2418. (C28 correction #1278)",
    ),
    (
        "If log2(x) + log4(x) = 6, find x.",
        ["8", "12", "16", "32"],
        2,
        "log2 x + (1/2)log2 x = 1.5 log2 x = 6 => log2 x = 4 => x=16.",
    ),
    (
        "Coefficient of x^5 in (1+x)^12?",
        ["792", "924", "495", "462"],
        0,
        "C(12,5)=792.",
    ),
    (
        "Right triangle legs 9 and 12. Altitude to hypotenuse?",
        ["7.2", "8.0", "6.4", "9.6"],
        0,
        "hyp=15, area=54, altitude=2*54/15=7.2.",
    ),
    (
        "Number of positive divisors of 360?",
        ["18", "20", "24", "30"],
        2,
        "360=2^3*3^2*5 => (3+1)(2+1)(1+1)=24.",
    ),
    (
        "a_n = 2*a_(n-1)+1, a_0=0. Find a_10.",
        ["511", "1023", "1024", "2047"],
        1,
        "a_n=2^n-1 => a_10=1023.",
    ),
    (
        "Limit as x->0 of sin(3x)/x?",
        ["0", "1", "3", "infinity"],
        2,
        "3*lim sin(3x)/(3x)=3.",
    ),
    (
        "3x+2y=12 and x-y=1. Find x.",
        ["2", "14/5", "3", "11/5"],
        1,
        "y=x-1 => 5x-2=12 => x=14/5.",
    ),
    (
        "Area of regular hexagon side 2?",
        ["6*sqrt(3)", "4*sqrt(3)", "12", "8*sqrt(3)"],
        0,
        "6 equilateral triangles of area sqrt(3) => 6*sqrt(3).",
    ),
    (
        "Distinct arrangements of the letters of 'LEVEL'?",
        ["30", "60", "120", "20"],
        0,
        "5!/(2!*2!)=30.",
    ),
    (
        "Smallest prime factor of 2027?",
        ["3", "7", "2027", "45"],
        2,
        "2027 is not divisible by 3,5,7,11,13,17,19,23,29,31,37,41,43 (43^2=1849<2027, 47^2>2027); prime.",
    ),
    (
        "Mean of 5 numbers is 10; four are 8,9,11,12. Find the fifth.",
        ["8", "10", "12", "15"],
        1,
        "total=50, known=40, fifth=10.",
    ),
    (
        "gcd(252, 198)?",
        ["12", "9", "36", "18"],
        3,
        "Euclid: 252-198=54; 198-3*54=36; 54-36=18; 36-2*18=0 => 18.",
    ),
    (
        "Cube side 3. Space diagonal?",
        ["3*sqrt(2)", "3*sqrt(3)", "6", "9"],
        1,
        "sqrt(3^2+3^2+3^2)=3*sqrt(3).",
    ),
    ("Sum k=1..50 of k?", ["1225", "1275", "1325", "2500"], 1, "50*51/2=1275."),
    (
        "a^2+b^2=25, ab=12, a+b>0. Find a+b.",
        ["5", "6", "8", "7"],
        3,
        "(a+b)^2=25+24=49 => a+b=7.",
    ),
    (
        "Exactly 2 heads in 4 fair coin flips?",
        ["1/4", "1/2", "5/16", "3/8"],
        3,
        "C(4,2)/2^4=6/16=3/8.",
    ),
]

KEY = [p[2] for p in PROBLEMS]
SOLUTIONS = [p[3] for p in PROBLEMS]


def format_problem(i: int) -> str:
    q, opts, _, _ = PROBLEMS[i]
    s = f"Question {i + 1}: {q}\n"
    for L, o in zip(LETTERS, opts):
        s += f"{L}) {o}\n"
    s += "Answer:"
    return s


def balance_report():
    from collections import Counter

    return dict(Counter(LETTERS[k] for k in KEY))


if __name__ == "__main__":
    assert len(PROBLEMS) == 20
    print("balance:", balance_report())
    for i in range(20):
        print(f"{i + 1:2d}. {LETTERS[KEY[i]]}  {PROBLEMS[i][0][:60]}")
