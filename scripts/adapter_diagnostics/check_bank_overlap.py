#!/usr/bin/env python3
"""Check question-string overlap between domain_banks (accept gates) and
heldout_domain_large (confirmation bank). High overlap = contaminated."""

import sys
from pathlib import Path

DIAG = Path(r"C:\02_QUILLAN\scripts\adapter_diagnostics")
sys.path.insert(0, str(DIAG))
from domain_banks import BANKS as SMALL
from heldout_domain_large import BANKS as LARGE


def norm(s):
    return " ".join(s.lower().split())


small_q = {norm(q) for Q, _ in SMALL.values() for q, _, _ in Q}
total = 0
overlap = 0
for dom, (Q, K) in LARGE.items():
    for q, _, _ in Q:
        total += 1
        if norm(q) in small_q:
            overlap += 1
            print("OVERLAP:", q)
print(f"overlap {overlap}/{total}")
