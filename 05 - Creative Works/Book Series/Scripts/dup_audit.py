"""Detect exact and near-duplicate prose lines in the Hybrid Throne manuscripts.

Outputs:
  1. Exact duplicate long lines (>= MIN_CHARS), excluding structural lines.
  2. Near-duplicate line pairs within a sliding window (token Jaccard >= THRESH).
  3. Repeated sentence-level n-gram boilerplate (e.g. same sentence twice).
"""

import re, sys, json, difflib
from collections import Counter, defaultdict

MIN_CHARS = 70
WINDOW = 40  # near-dup search window in lines (paragraphs are wrapped lines)
JACCARD_THRESH = 0.72


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str):
    return set(normalize(s).split())


def is_structural(line: str) -> bool:
    st = line.strip()
    return (
        not st
        or st.startswith(("#", "- ", "|", "---", "![", "> "))
        or bool(re.match(r"^(chapter|epilogue|coda|prologue)\b", st, re.I))
    )


def analyze(path: str):
    raw = open(path, encoding="utf-8-sig", errors="replace").read()
    lines = raw.splitlines()
    norm_lines = [normalize(l) for l in lines]

    # ---- exact duplicates (long lines only) ----
    counts = Counter()
    first_seen = {}
    for i, l in enumerate(norm_lines):
        if len(l) >= MIN_CHARS and not is_structural(lines[i]):
            counts[l] += 1
            first_seen.setdefault(l, i + 1)
    exact = [(l, c, first_seen[l]) for l, c in counts.items() if c > 1]
    exact.sort(key=lambda x: -x[1])

    # ---- near duplicates within window ----
    near = []
    idxs = [
        i
        for i, l in enumerate(lines)
        if len(norm_lines[i]) >= MIN_CHARS and not is_structural(l)
    ]
    for a in range(len(idxs)):
        i = idxs[a]
        ti = tokens(lines[i])
        for b in range(a + 1, len(idxs)):
            j = idxs[b]
            if j - i > WINDOW:
                break
            tj = tokens(lines[j])
            inter = len(ti & tj)
            if not inter:
                continue
            union = len(ti | tj)
            jac = inter / union
            if jac >= JACCARD_THRESH:
                near.append((i + 1, j + 1, round(jac, 3)))

    # ---- repeated sentences anywhere ----
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", raw))
    scount = Counter()
    for s in sentences:
        ns = normalize(s)
        if len(ns) >= 80:
            scount[ns] += 1
    sent_dups = [(s, c) for s, c in scount.items() if c > 1]
    sent_dups.sort(key=lambda x: -x[1])

    return exact, near, sent_dups, lines


if __name__ == "__main__":
    path = sys.argv[1]
    exact, near, sent_dups, lines = analyze(path)
    print(f"=== {path} ===")
    print(f"\n--- EXACT DUPLICATE LONG LINES ({len(exact)}) ---")
    for l, c, first in exact[:60]:
        print(f"[x{c} first@L{first}] {l[:150]}")
    print(f"\n--- NEAR-DUPLICATE PAIRS WITHIN {WINDOW} LINES ({len(near)}) ---")
    for i, j, jac in near[:80]:
        print(f"L{i} <-> L{j}  jac={jac}")
        print(f"   A: {lines[i - 1].strip()[:140]}")
        print(f"   B: {lines[j - 1].strip()[:140]}")
    print(f"\n--- REPEATED SENTENCES >=80 chars ({len(sent_dups)}) ---")
    for s, c in sent_dups[:60]:
        print(f"[x{c}] {s[:180]}")
