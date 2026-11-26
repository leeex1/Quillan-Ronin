#!/usr/bin/env python3
"""Find WHY rows are empty and WHICH native IDs were overwritten."""

from __future__ import annotations
import sys, json
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    text2q = json.loads((OUT / "quillan_vocab.json").read_text(encoding="utf-8"))
    q2text = {v: k for k, v in text2q.items()}
    meta = json.loads((OUT / "quillan_token_maps.json").read_text(encoding="utf-8"))
    d2q = meta["donor_to_quillan"]

    ck = torch.load(OUT / "quillan_embedding_compiled.pt", weights_only=True)
    E, cov = ck["embedding"], ck["coverage"]
    zero_q = torch.nonzero(E.norm(dim=-1) == 0).flatten().tolist()
    print(f"zero rows = {len(zero_q)}", flush=True)
    print("  sample empty slots (qid, text, donor-count):", flush=True)
    for q in zero_q[:15]:
        print(f"    {q}: {q2text.get(q, '?')!r}  donors={int(cov[q])}", flush=True)
    # classify empty-slot strings
    kinds = Counter()
    for q in zero_q:
        t = q2text.get(q, "")
        if t.startswith("<unmapped"):
            kinds["unmapped-placeholder"] += 1
        elif t == "":
            kinds["empty-string"] += 1
        elif t.isspace():
            kinds["whitespace"] += 1
        else:
            kinds["real-text"] += 1
    print(f"  empty-row kinds: {dict(kinds)}", flush=True)

    print(
        "\n=== overwritten native IDs (multiple native IDs -> same canonical slot) ===",
        flush=True,
    )
    for name, m in d2q.items():
        c = Counter(int(v) for v in m.values())
        dups = {q: n for q, n in c.items() if n > 1}
        print(
            f"  {name}: {len(dups)} slots hold >1 native ID, extra IDs lost = {sum(n - 1 for n in dups.values())}",
            flush=True,
        )
        for q, n in list(dups.items())[:5]:
            native_ids = [int(k) for k, v in m.items() if int(v) == q]
            print(
                f"    q={q} text={q2text.get(q, '?')!r} native_ids={native_ids[:6]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
