#!/usr/bin/env python3
"""Apply C28's 6 option-order permutations to hard_math40.py (indices 20-39)."""

from pathlib import Path

p = Path(r"C:\02_QUILLAN\scripts\adapter_diagnostics\hard_math40.py")
s = p.read_text(encoding="utf-8")

# (old option-list line, new option-list line, old key line, new key) — exact substrings
edits = [
    (
        '["7", "8", "9", "11"],\n        0,',
        '["9", "8", "7", "11"],\n        2,',
    ),  # idx21
    ('["34", "55", "89", "144"], 1,', '["34", "144", "89", "55"], 3,'),  # idx25
    (
        '["3/8", "7/8", "1/2", "5/8"],\n        1,',
        '["3/8", "5/8", "1/2", "7/8"],\n        3,',
    ),  # idx28
    ('["11", "10", "9", "12"], 1,', '["11", "12", "9", "10"], 3,'),  # idx30
    ('["10", "11", "12", "20"], 1,', '["10", "20", "12", "11"], 3,'),  # idx36
]
for old, new in edits:
    if old in s:
        s = s.replace(old, new, 1)
        print("applied:", old[:40].replace("\n", " "))
    else:
        print("NOT FOUND:", old[:40].replace("\n", " "))

p.write_text(s, encoding="utf-8")
print("written")
