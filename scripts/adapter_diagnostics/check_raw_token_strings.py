#!/usr/bin/env python3
"""Verify raw token strings are 1:1 and comparable across all four tokenizers."""

from __future__ import annotations
from tokenizers import Tokenizer

PATHS = {
    "qwen": r"E:\source_models\qwen_0.5b\tokenizer.json",
    "llama": r"E:\source_models\llama_1b\tokenizer.json",
    "bitnet": r"E:\source_models\bitnet_2b_4t\tokenizer.json",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json",
}
for name, p in PATHS.items():
    tok = Tokenizer.from_file(p)
    v = tok.get_vocab()  # token_string -> id
    ids_to_str = {i: s for s, i in v.items()}
    print(
        f"{name}: vocab_keys={len(v)} distinct_ids={len(set(v.values()))} "
        f"id_range={min(v.values())}..{max(v.values())}"
    )
    for t in ["cat", "Ġthe", "Ġcat", "hello", "<0xF5>", "Ġ"]:
        print(f"    {t!r}: id={v.get(t, 'MISSING')}")
    # collision check: any id with 2 strings?
    print(f"    duplicate-id strings: {len(v) - len(set(v.values()))}")
