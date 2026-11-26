#!/usr/bin/env python3
"""COMPILE STEP 1 (CORRECT): canonical Quillan token-ID space from RAW token strings.

Uses tokenizer.get_vocab() token strings, which are 1:1 with IDs and preserve
byte identity. Decoded text collapsed thousands of distinct byte tokens onto
'\ufffd' and lost them. This version loses none.

No training, no gradient, no weight edits. Pure reindexing metadata.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = ROOT / "exports"

DONORS = {
    "qwen": r"E:\source_models\qwen_0.5b\tokenizer.json",
    "llama": r"E:\source_models\llama_1b\tokenizer.json",
    "bitnet": r"E:\source_models\bitnet_2b_4t\tokenizer.json",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json",
}


def main():
    from tokenizers import Tokenizer

    per_donor = {}
    print("=== raw token strings (1:1, byte-preserving) ===", flush=True)
    for name, path in DONORS.items():
        tok = Tokenizer.from_file(path)
        v = tok.get_vocab()  # token_string -> id
        ids = list(v.values())
        assert len(ids) == len(set(ids)), f"{name}: duplicate IDs!"
        per_donor[name] = {i: s for s, i in v.items()}
        print(
            f"  {name}: {len(v)} strings, distinct_ids={len(set(ids))}, collisions=0",
            flush=True,
        )

    print("\n=== canonical Quillan vocabulary ===", flush=True)
    strings = sorted({s for d in per_donor.values() for s in d.values()})
    text2q = {s: i for i, s in enumerate(strings)}
    Q = len(strings)
    print(f"  canonical slots = {Q}", flush=True)

    cover_count = {}
    for d in per_donor.values():
        for s in set(d.values()):
            cover_count[s] = cover_count.get(s, 0) + 1
    cover = {k: 0 for k in range(1, 5)}
    for c in cover_count.values():
        cover[min(c, 4)] += 1
    print(
        "  coverage: " + "  ".join(f"in{k}={cover[k]}" for k in range(1, 5)), flush=True
    )

    print("\n=== per-donor maps ===", flush=True)
    q2d = {}
    meta = {"canonical_vocab_size": Q, "donors": {}}
    for name, id2s in per_donor.items():
        arr = torch.full((Q,), -1, dtype=torch.int64)
        for did, s in id2s.items():
            arr[text2q[s]] = did
        q2d[name] = arr
        filled = int((arr >= 0).sum())
        meta["donors"][name] = {
            "tokenizer": DONORS[name],
            "native_ids": len(id2s),
            "slots_filled": filled,
            "coverage_pct": round(100.0 * filled / Q, 2),
        }
        print(
            f"  {name}: native={len(id2s)} filled={filled} ({100.0 * filled / Q:.1f}%)",
            flush=True,
        )

    print("\n=== examples ===", flush=True)
    for s in ["cat", "Ġthe", "Ġcat", "hello", "Ġ", "Ċ"]:
        if s in text2q:
            q = text2q[s]
            row = [f"{s!r}", f"Q={q}"]
            for name in DONORS:
                d = int(q2d[name][q])
                row.append(f"{name}={d}" if d >= 0 else f"{name}=MISSING")
            print("  " + "  ".join(row), flush=True)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "quillan_vocab.json", "w", encoding="utf-8") as f:
        json.dump(text2q, f, ensure_ascii=False)
    torch.save(q2d, OUT / "quillan_token_maps_q2d.pt")
    d2q = {n: {str(did): text2q[s] for did, s in per_donor[n].items()} for n in DONORS}
    with open(OUT / "quillan_token_maps.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "donor_to_quillan": d2q}, f, ensure_ascii=False)
    print("\nSAVED token map (raw-string basis)", flush=True)


if __name__ == "__main__":
    main()
