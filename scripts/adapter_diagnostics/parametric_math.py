#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parametric math family: tests COMPUTATION, not memorization.

Fixed options A)20 B)30 C)40 D)50. Question: "What is a + b?" where a,b are
random integers in [1,25] chosen so a+b is exactly one of {20,30,40,50}.
The correct letter depends on computing a+b and bucketing it — a real 2-step
computation. Held-out uses (a,b) pairs never seen in calibration, so a lookup
table cannot pass. This is the honest test of 'installed intelligence via tensors'.
"""

from __future__ import annotations
import random

LETTERS = ["A", "B", "C", "D"]
BUCKETS = [20, 30, 40, 50]


def gen(rng: random.Random) -> tuple[str, int, tuple[int, int]]:
    # pick a bucket, then a,b in [1,25] summing to it
    bi = rng.randrange(4)
    target = BUCKETS[bi]
    a = rng.randint(max(1, target - 25), min(25, target - 1))
    b = target - a
    assert 1 <= a <= 25 and 1 <= b <= 25
    q = f"What is {a} + {b}?"
    opts = [f"{v}" for v in BUCKETS]  # fixed option order => letter = bucket index
    return q, opts, bi, (a, b)


def format_q(q, opts):
    return (
        f"Question: {q}\n"
        + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
        + "Answer:"
    )


def make_split(n, seed):
    rng = random.Random(seed)
    out = []
    seen = set()
    while len(out) < n:
        q, opts, key, pair = gen(rng)
        if pair in seen:
            continue
        seen.add(pair)
        out.append((format_q(q, opts), key, pair))
    return out
