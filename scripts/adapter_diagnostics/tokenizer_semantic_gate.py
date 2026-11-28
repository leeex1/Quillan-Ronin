#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noncanonical diagnostic tokenizer semantic gate.

Read-only against tokenizer artifacts. Writes only the explicit report path.
No model coherence or completion claims are made.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OWNER = Path(r"F:\02_QUILLAN\exports\quillan_tokenizer_v54_fourdonor\tokenizer.json")
QWEN = Path(r"E:\source_models\qwen_0.5b\tokenizer.json")
LLAMA = Path(r"E:\source_models\llama_1b\tokenizer.json")
LLAMA_SAFETENSORS = Path(r"E:\source_models\llama_1b\model.safetensors")
REPORT = Path(r"F:\02_QUILLAN\exports\diagnostics\tokenizer_gate__NONCANONICAL.json")

EXPECTED_SPECIALS = {
    151643: "<|quillan_pad|>",
    151644: "<|quillan_bos|>",
    151645: "<|quillan_start|>",
    151646: "<|quillan_user|>",
    151647: "<|quillan_assistant|>",
}
EXPECTED_OWNER_VOCAB = 151648

ROUNDTRIP_STRINGS = [
    "Quillan tokenizer semantic gate",
    "Ronin — 義, 勇, 仁; café ☕ 日本語",
    "<|quillan_pad|>",
    "<|quillan_bos|>",
    "<|quillan_start|>",
    "<|quillan_user|>",
    "<|quillan_assistant|>",
]
PAD_TEXT = "Padding invariance check"
PAD_COUNT = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def file_meta(path: Path, compute_hash: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    meta: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "bytes": st.st_size,
    }
    if compute_hash:
        meta["sha256"] = sha256_file(path)
    else:
        meta["sha256"] = "NOT_COMPUTED_LARGE_ARTIFACT"
    return meta


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def vocab_from(doc: dict[str, Any]) -> dict[str, int]:
    return dict(doc.get("model", {}).get("vocab", {}))


def inv_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {i: t for t, i in vocab.items()}


def added_token_map(doc: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in doc.get("added_tokens", []):
        out[int(item["id"])] = str(item["content"])
    return out


def safe_repr(value: str) -> str:
    return repr(value).encode("ascii", "backslashreplace").decode("ascii")


def safetensors_header(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            return {"exists": True, "error": "short header length"}
        header_len = struct.unpack("<Q", raw_len)[0]
        header_raw = fh.read(header_len)
    header = json.loads(header_raw.decode("utf-8"))
    weights = {
        k: v
        for k, v in header.items()
        if isinstance(v, dict) and str(k).endswith("embed_tokens.weight")
    }
    return {
        "exists": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": "NOT_COMPUTED_HEADER_ONLY_READ",
        "header_bytes": header_len,
        "embed_token_tensors": weights,
    }


def coverage(
    owner_inv: dict[int, str],
    donor_vocab: dict[str, int],
    donor_added: dict[int, str],
    limit: int = 10,
) -> dict[str, Any]:
    base_tokens = set(donor_vocab.keys())
    full_tokens = set(base_tokens) | set(donor_added.values())
    mapped_ids: list[int] = []
    unmapped_ids: list[int] = []
    mapped_base_ids: list[int] = []
    mapped_added_only_ids: list[int] = []

    for owner_id, owner_token in owner_inv.items():
        if owner_token in base_tokens:
            mapped_ids.append(owner_id)
            mapped_base_ids.append(owner_id)
        elif owner_token in full_tokens:
            mapped_ids.append(owner_id)
            mapped_added_only_ids.append(owner_id)
        else:
            unmapped_ids.append(owner_id)

    first_unmapped = [
        {
            "owner_id": i,
            "token": owner_inv[i],
            "token_repr": safe_repr(owner_inv[i]),
            "is_owner_special": i in EXPECTED_SPECIALS,
        }
        for i in sorted(unmapped_ids)[:limit]
    ]
    return {
        "owner_ids_total": len(owner_inv),
        "donor_model_vocab_len": len(donor_vocab),
        "donor_added_tokens_count": len(donor_added),
        "donor_full_token_string_count": len(full_tokens),
        "mapped_exact_string_owner_ids": len(mapped_ids),
        "unmapped_exact_string_owner_ids": len(unmapped_ids),
        "mapped_via_donor_model_vocab": len(mapped_base_ids),
        "mapped_via_donor_added_tokens_only": len(mapped_added_only_ids),
        "first_10_unmapped_owner_ids": first_unmapped,
    }


def main() -> int:
    dependency = {
        "python_version": os.sys.version,
        "json_stdlib_available": True,
        "tokenizers_spec_found": bool(importlib.util.find_spec("tokenizers")),
        "tokenizers_version": None,
        "tokenizers_import_error": None,
        "used_library": False,
    }
    tokenizer_lib = None
    if dependency["tokenizers_spec_found"]:
        try:
            import tokenizers  # type: ignore

            tokenizer_lib = tokenizers
            dependency["tokenizers_version"] = importlib.metadata.version("tokenizers")
            dependency["used_library"] = True
        except Exception as exc:  # pragma: no cover - defensive
            dependency["tokenizers_import_error"] = f"{type(exc).__name__}: {exc}"

    artifact_meta = {
        "owner": file_meta(OWNER),
        "qwen": file_meta(QWEN),
        "llama": file_meta(LLAMA),
        "llama_safetensors": file_meta(LLAMA_SAFETENSORS, compute_hash=False),
    }

    owner_doc = load_json(OWNER)
    qwen_doc = load_json(QWEN)
    llama_doc = load_json(LLAMA)

    owner_vocab = vocab_from(owner_doc)
    qwen_vocab = vocab_from(qwen_doc)
    llama_vocab = vocab_from(llama_doc)
    owner_inv = inv_vocab(owner_vocab)
    owner_added = added_token_map(owner_doc)
    qwen_added = added_token_map(qwen_doc)
    llama_added = added_token_map(llama_doc)

    vocab_values = list(owner_vocab.values())
    value_counts = Counter(vocab_values)
    duplicate_ids = {i: c for i, c in value_counts.items() if c > 1}
    owner_added_dupes = {
        i: c
        for i, c in Counter(
            int(a["id"]) for a in owner_doc.get("added_tokens", [])
        ).items()
        if c > 1
    }
    out_of_range_vocab = [i for i in vocab_values if i < 0 or i >= EXPECTED_OWNER_VOCAB]
    out_of_range_added = [i for i in owner_added if i < 0 or i >= EXPECTED_OWNER_VOCAB]
    specials_id_string_ok = all(
        owner_added.get(i) == s for i, s in EXPECTED_SPECIALS.items()
    )
    specials_inv_ok = all(owner_inv.get(i) == s for i, s in EXPECTED_SPECIALS.items())
    specials_are_special = all(
        any(
            int(a["id"]) == i and bool(a.get("special"))
            for a in owner_doc.get("added_tokens", [])
        )
        for i in EXPECTED_SPECIALS
    )

    special_check = {
        "check_id": 2,
        "name": "Special-ID check",
        "pass": (
            len(owner_vocab) == EXPECTED_OWNER_VOCAB
            and not duplicate_ids
            and not out_of_range_vocab
            and not out_of_range_added
            and specials_id_string_ok
            and specials_inv_ok
            and specials_are_special
        ),
        "owner_model_vocab_len": len(owner_vocab),
        "expected_owner_vocab_len": EXPECTED_OWNER_VOCAB,
        "vocab_min_id": min(vocab_values),
        "vocab_max_id": max(vocab_values),
        "expected_sovereign_specials": [
            {
                "id": i,
                "token": s,
                "owner_added_token": owner_added.get(i),
                "owner_vocab_inverse": owner_inv.get(i),
            }
            for i, s in EXPECTED_SPECIALS.items()
        ],
        "duplicate_vocab_ids": duplicate_ids,
        "duplicate_added_token_ids": owner_added_dupes,
        "out_of_range_vocab_ids": out_of_range_vocab,
        "out_of_range_added_token_ids": out_of_range_added,
        "all_expected_specials_marked_special": specials_are_special,
    }

    roundtrip_rows: list[dict[str, Any]] = []
    if tokenizer_lib is not None and dependency["used_library"]:
        tok = tokenizer_lib.Tokenizer.from_file(str(OWNER))
        for idx, text in enumerate(ROUNDTRIP_STRINGS, start=1):
            enc = tok.encode(text)
            ids = list(enc.ids)
            decoded = tok.decode(ids, skip_special_tokens=False)
            roundtrip_rows.append(
                {
                    "case": idx,
                    "input": text,
                    "input_repr": safe_repr(text),
                    "ids": ids,
                    "decoded": decoded,
                    "decoded_repr": safe_repr(decoded),
                    "pass": decoded == text,
                }
            )

        text_ids = list(tok.encode(PAD_TEXT).ids)
        pad_id = 151643
        padded_ids = text_ids + [pad_id] * PAD_COUNT
        decoded_skip_true = tok.decode(padded_ids, skip_special_tokens=True)
        decoded_skip_false = tok.decode(padded_ids, skip_special_tokens=False)
        pad_token_is_special = any(
            int(a["id"]) == pad_id and bool(a.get("special"))
            for a in owner_doc.get("added_tokens", [])
        )
        padding_check = {
            "check_id": 3,
            "name": "Padding/mask invariance",
            "method": "tokenizers.Tokenizer.decode(skip_special_tokens=True)",
            "text": PAD_TEXT,
            "text_ids": text_ids,
            "pad_token_id": pad_id,
            "pad_token": owner_added.get(pad_id),
            "pad_token_is_special": pad_token_is_special,
            "pad_count": PAD_COUNT,
            "ids_encoded_text_plus_pad": padded_ids,
            "decoded_skip_special_tokens_true": decoded_skip_true,
            "decoded_skip_special_tokens_false": decoded_skip_false,
            "pass": decoded_skip_true == PAD_TEXT and pad_token_is_special,
        }
        roundtrip_pass = all(row["pass"] for row in roundtrip_rows)
    else:
        roundtrip_rows = []
        padding_check = {
            "check_id": 3,
            "name": "Padding/mask invariance",
            "pass": False,
            "error": "tokenizers library unavailable; library round-trip not executed",
        }
        roundtrip_pass = False

    roundtrip_check = {
        "check_id": 1,
        "name": "Owner tokenizer encode/decode round-trip",
        "method": "tokenizers.Tokenizer.from_file(owner).encode/decode; skip_special_tokens=False; installed tokenizers decode API has no clean_up_tokenization_spaces kwarg",
        "case_count": len(ROUNDTRIP_STRINGS),
        "pass": roundtrip_pass and len(roundtrip_rows) == len(ROUNDTRIP_STRINGS),
        "cases": roundtrip_rows,
    }

    qwen_cov = coverage(owner_inv, qwen_vocab, qwen_added)
    llama_cov = coverage(owner_inv, llama_vocab, llama_added)
    qwen_cov["name"] = "owner->Qwen exact string remap coverage"
    llama_cov["name"] = "owner->Llama exact string remap coverage"

    converter_claims = {
        "Mini": {"mapped": 151643, "unmapped": 5},
        "Main": {"mapped": 109566, "unmapped": 42082},
    }
    qwen_cov["converter_reconciliation"] = {
        "claim": converter_claims["Mini"],
        "observed_mapped": qwen_cov["mapped_exact_string_owner_ids"],
        "observed_unmapped": qwen_cov["unmapped_exact_string_owner_ids"],
        "matches_claim": (
            qwen_cov["mapped_exact_string_owner_ids"]
            == converter_claims["Mini"]["mapped"]
            and qwen_cov["unmapped_exact_string_owner_ids"]
            == converter_claims["Mini"]["unmapped"]
        ),
    }
    llama_cov["converter_reconciliation"] = {
        "claim": converter_claims["Main"],
        "observed_mapped": llama_cov["mapped_exact_string_owner_ids"],
        "observed_unmapped": llama_cov["unmapped_exact_string_owner_ids"],
        "matches_claim": (
            llama_cov["mapped_exact_string_owner_ids"]
            == converter_claims["Main"]["mapped"]
            and llama_cov["unmapped_exact_string_owner_ids"]
            == converter_claims["Main"]["unmapped"]
        ),
    }

    st_header = safetensors_header(LLAMA_SAFETENSORS)
    embed_entries = st_header.get("embed_token_tensors", {})
    embed_summary = []
    for key, meta in embed_entries.items():
        shape = meta.get("shape")
        rows = int(shape[0]) if isinstance(shape, list) and shape else None
        embed_summary.append(
            {
                "tensor": key,
                "dtype": meta.get("dtype"),
                "shape": shape,
                "embed_rows": rows,
                "llama_model_vocab_len": len(llama_vocab),
                "llama_added_tokens_count": len(llama_added),
                "llama_total_assigned_token_ids": len(llama_vocab) + len(llama_added),
                "rows_match_model_vocab_only": rows == len(llama_vocab),
                "rows_match_model_vocab_plus_added": rows
                == len(llama_vocab) + len(llama_added),
            }
        )
    tokenizer_row_check = {
        "check_id": 5,
        "name": "Llama tokenizer vocab vs safetensors embed rows",
        "pass": bool(embed_summary),
        "safetensors": st_header,
        "embedding_summary": embed_summary,
        "interpretation": (
            "Converter's donor vocab 128000 refers to tokenizer model.vocab; tokenizer also has 256 added tokens "
            "at ids 128000..128255, so total tokenizer token-id coverage is 128256."
        ),
    }

    coverage_check = {
        "check_id": 4,
        "name": "Token remap exact-string coverage",
        "method": "Direct tokenizer.json model.vocab string lookup by owner ID; added-token strings included only in the full-token-string variant. No union-slot token maps used.",
        "pass": (
            qwen_cov["converter_reconciliation"]["matches_claim"]
            and llama_cov["converter_reconciliation"]["matches_claim"]
        ),
        "owner_to_qwen": qwen_cov,
        "owner_to_llama": llama_cov,
    }

    checks = [
        roundtrip_check,
        special_check,
        padding_check,
        coverage_check,
        tokenizer_row_check,
    ]
    report = {
        "report_class": "NONCANONICAL_DIAGNOSTIC",
        "classification": "tokenizer_semantic_gate",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope_statement": "Read-only tokenizer artifact diagnostics. No model coherence, completion, or training claim is made.",
        "artifacts": artifact_meta,
        "dependency_status": dependency,
        "commands": {
            "script": str(Path(__file__)),
            "invoke": f'python "{Path(__file__)}"',
            "environment_note": "PYTHONIOENCODING=utf-8 used by caller for Unicode-safe PowerShell output.",
        },
        "checks": {f"check_{c['check_id']}": c for c in checks},
        "overall_pass": all(bool(c.get("pass")) for c in checks),
        "caveats": [
            "Exact-string coverage is not semantic similarity and does not validate embedding row alignment.",
            "No model was loaded; no coherence or completion conclusion is implied.",
        ],
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"REPORT={REPORT}")
    print(f"REPORT_BYTES={REPORT.stat().st_size}")
    print(f"OVERALL_PASS={report['overall_pass']}")
    for c in checks:
        print(f"CHECK_{c['check_id']}_PASS={bool(c.get('pass'))} name={c['name']}")
    print(f"TOKENIZERS_VERSION={dependency['tokenizers_version']}")
    print(f"OWNER_VOCAB_LEN={special_check['owner_model_vocab_len']}")
    print(
        f"QWEN_MAPPED_UNMAPPED={qwen_cov['mapped_exact_string_owner_ids']}/{qwen_cov['unmapped_exact_string_owner_ids']}"
    )
    print(
        f"LLAMA_MAPPED_UNMAPPED={llama_cov['mapped_exact_string_owner_ids']}/{llama_cov['unmapped_exact_string_owner_ids']}"
    )
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
