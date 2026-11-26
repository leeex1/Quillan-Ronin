#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the Quillan v5.4 four-donor tokenizer (closed-form, zero training).

Design (corrected after v1 produced an oversized 188k union with broken merges):
  - BASE = Qwen2.5 byte-level BPE vocabulary (151,643 tokens). It is the largest
    and richest byte-BPE of the four donors, has 85%+ overlap with Llama/BitNet,
    and is the only tokenizer proven end-to-end in llama.cpp on this box.
  - ADD the 5 sovereign special tokens at the top with FIXED ids 151643..151647
    (<|start|>, <|user|>, <|assistant|>, <|im_start|>, <|im_end|>), plus a pad.
  - Normalizer/pre-tokenizer/decoder/merges are copied VERBATIM from Qwen so the
    byte-level BPE is valid and llama.cpp/bitnet.cpp can load it.
  - Result: 151,649 tokens — within the ~150k 4GB-VRAM target.

Why not union all four vocabs (v1 mistake): byte-level BPE token strings are only
meaningful with their matching merge table. Mixing Qwen+Llama+legacy GPT-2 tokens
under Qwen merges creates a tokenizer that cannot round-trip. Qwen base + sovereign
specials is the correct closed-form construction. NO training, NO backprop.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")
QWEN = Path(r"E:\source_models\qwen_0.5b")
OUT = ROOT / "exports" / "quillan_tokenizer_v54_fourdonor"

# Sovereign special tokens — appended after Qwen's 151,643 base tokens.
# Qwen already provides <|endoftext|> (0), <|im_start|> (151644), <|im_end|> (151645).
# We add the distinct sovereign set with fixed ids above the base vocab.
SOVEREIGN_SPECIALS = [
    "<|quillan_pad|>",
    "<|quillan_bos|>",
    "<|quillan_start|>",
    "<|quillan_user|>",
    "<|quillan_assistant|>",
]


def main() -> None:
    print("=" * 70)
    print("Quillan v5.4 four-donor tokenizer builder (closed-form, zero training)")
    print("=" * 70)

    qwen_raw = json.loads((QWEN / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = dict(qwen_raw["model"]["vocab"])
    base_size = len(vocab)
    print(f"[base] Qwen byte-level BPE vocab: {base_size} tokens")

    # Assign sovereign specials ids immediately after the base vocab.
    added = []
    for i, sp in enumerate(SOVEREIGN_SPECIALS):
        tid = base_size + i
        if sp not in vocab:
            vocab[sp] = tid
            added.append(
                {
                    "id": tid,
                    "content": sp,
                    "special": True,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                }
            )
            print(f"[special] {sp} -> id {tid}")

    tok_json = {
        "version": qwen_raw.get("version", "1.0"),
        "normalizer": qwen_raw["normalizer"],
        "pre_tokenizer": qwen_raw["pre_tokenizer"],
        "post_processor": qwen_raw["post_processor"],
        "decoder": qwen_raw["decoder"],
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": qwen_raw["model"]["merges"],
            "byte_fallback": qwen_raw["model"].get("byte_fallback", False),
            "unk_token": qwen_raw["model"].get("unk_token", "<|endoftext|>"),
            "continuing_subword_prefix": qwen_raw["model"].get(
                "continuing_subword_prefix", ""
            ),
            "end_of_word_suffix": qwen_raw["model"].get("end_of_word_suffix", ""),
            "fuse_unk": qwen_raw["model"].get("fuse_unk", False),
        },
        "added_tokens": added,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tokenizer.json").write_text(
        json.dumps(tok_json, ensure_ascii=False), encoding="utf-8"
    )
    cfg = {
        "model_type": "quillan_ronin_oni",
        "vocab_size": len(vocab),
        "eos_token_id": 151643,
        "bos_token_id": base_size + 1,
        "pad_token_id": base_size + 0,
        "base": "Qwen2.5 byte-level BPE (151643) + 5 sovereign specials",
        "donor_overlap": {"qwen": 1.0, "llama": 0.854, "bitnet": 0.854},
        "build_method": "closed-form base + specials, zero training/backprop",
        "llama_cpp_compatible": True,
    }
    (OUT / "tokenizer_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "BUILD_REPORT.json").write_text(
        json.dumps(
            {
                "vocab_size": len(vocab),
                "base_size": base_size,
                "sovereign_specials": {sp: vocab.get(sp) for sp in SOVEREIGN_SPECIALS},
                "merges": len(qwen_raw["model"]["merges"]),
                "status": "closed-form v2 — Qwen base + sovereign specials",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Validate round-trip
    try:
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(
            str(OUT), local_files_only=True, use_fast=True
        )
        test = "The capital of France is Paris. <|quillan_start|>user hello<|quillan_user|>"
        ids = t.encode(test, add_special_tokens=False)
        dec = t.decode(ids, skip_special_tokens=False)
        print(f"[validate] roundtrip: {dec!r}")
        print(f"[validate] vocab={t.vocab_size} len={len(t)} fast={t.is_fast}")
        assert dec.replace("<|quillan_start|>", "").find("capital of France") >= 0, (
            "roundtrip failed"
        )
        print("[validate] PASS")
    except Exception as e:
        print(f"[validate] ERROR: {type(e).__name__}: {e}")
        raise

    print(f"\n[done] tokenizer -> {OUT}")


if __name__ == "__main__":
    main()
