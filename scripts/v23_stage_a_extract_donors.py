#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage A: Pre-Combination Base Integrity Verification, Clean Donor Caching,
and Zero-Contamination Non-Test Dataset Builder for WSP v23.

Guarantees:
  1. Verifies clean HF snapshots & GGUFs (shapes, param counts, finite values,
     matching headers against E:\\source_models\\ bases).
  2. Selectively unpacks Microsoft BitNet b1.58 2B-4T (E:\\source_models\\bitnet_2b_4t)
     via safetensors byte-offset seeks (no full-file mmap / pagefile 1455 risk),
     verifying unpacked ternary * weight_scale values are finite and non-NaN.
  3. Builds strictly non-test training/calibration splits (gsm8k:train,
     ai2_arc:train, mmlu:auxiliary_train/dev/val, non-test code & BFCL synthetic)
     filtered against ALL held-out eval texts and official test splits.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import gguf
import gguf.quants
import numpy as np
import torch
from datasets import load_dataset
from safetensors import safe_open

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")

import wsp_v8_closed_form_sota_compiler as v8
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, HELD_OUT_SUITE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("V23_StageA")

EXPORT_ROOT = Path(r"C:\02_QUILLAN\exports")
DONOR_CACHE_DIR = EXPORT_ROOT / "donor_cache_v23"
CLEAN_DATA_PATH = EXPORT_ROOT / "v23_clean_train_data.json"
BITNET_2B4T_ST = Path(r"E:\source_models\bitnet_2b_4t\model.safetensors")
BITNET_2B4T_DIR = Path(r"E:\source_models\bitnet_2b_4t")
E_LLAMA_ST = Path(r"E:\source_models\llama_1b\model.safetensors")
E_QWEN_ST = Path(r"E:\source_models\qwen_0.5b\model.safetensors")
BITNET_LARGE_HUB = Path(r"C:\Users\Admin\.cache\huggingface\hub\models--1bitLLM--bitnet_b1_58-large")


def read_safetensors_header(st_path: Path) -> Tuple[int, Dict[str, Any]]:
    """Reads only the JSON header of a safetensors file without mmaping tensor data."""
    with open(st_path, "rb") as fp:
        hdr_len = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(hdr_len).decode("utf-8"))
    return 8 + hdr_len, header


def read_safetensors_tensor_by_offset(
    st_path: Path, data_base: int, meta: Dict[str, Any]
) -> torch.Tensor:
    """Reads a single tensor directly via byte offset seek to avoid large mmap commits."""
    start, end = meta["data_offsets"]
    nbytes = end - start
    with open(st_path, "rb") as fp:
        fp.seek(data_base + start)
        raw = fp.read(nbytes)
    if len(raw) != nbytes:
        raise IOError(f"Short read on {st_path}: expected {nbytes}, got {len(raw)}")

    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    buf = bytearray(raw)
    if dtype == "BF16":
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    if dtype == "F16":
        arr = np.frombuffer(buf, dtype=np.float16).reshape(shape)
        return torch.from_numpy(arr.copy())
    if dtype == "F32":
        arr = np.frombuffer(buf, dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy())
    if dtype == "U8":
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        return torch.from_numpy(arr.copy())
    raise ValueError(f"Unsupported safetensors dtype {dtype} in {st_path}")


def unpack_bitnet_2b4t_weight(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Unpacks Microsoft BitNet b1.58 2B4T 2-bit packed U8 weight tensor along axis 0
    using exact transformers.integrations.bitnet.unpack_weights bit layout, scaled
    by BF16 weight_scale.
    """
    values_per_item = 4
    packed_shape = packed.shape
    out_rows = packed_shape[0] * values_per_item
    unpacked_shape = (out_rows, *packed_shape[1:])
    unpacked = torch.zeros(unpacked_shape, dtype=torch.uint8)
    for i in range(values_per_item):
        s_idx = i * packed_shape[0]
        e_idx = s_idx + packed_shape[0]
        mask = 3 << (2 * i)
        unpacked[s_idx:e_idx] = (packed & mask) >> (2 * i)
    w_ternary = unpacked.to(torch.float32) - 1.0
    w_scaled = w_ternary * scale.float().reshape(())
    if not torch.isfinite(w_scaled).all():
        raise ValueError("Non-finite value detected in unpacked BitNet-2B-4T weight")
    return w_scaled


def verify_tensor_stats(name: str, t: torch.Tensor) -> Dict[str, Any]:
    """Asserts tensor finiteness and returns norm/shape summary."""
    tf = t.float()
    if not torch.isfinite(tf).all():
        raise ValueError(f"Non-finite values in tensor {name}")
    norm_val = float(torch.norm(tf).item())
    if norm_val == 0.0:
        raise ValueError(f"Zero-norm tensor detected: {name}")
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "frobenius_norm": round(norm_val, 4),
    }


def verify_clean_bases_and_extract_donors() -> Dict[str, Any]:
    """
    Verifies clean pre-trained base models (Llama-3.2-1B, Qwen2.5-0.5B, Falcon3-1B,
    BitNet-2B-4T, BitNet-Large) and caches compact float16 donor projections on NVMe C:\\.
    """
    DONOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}

    # 1. Verify E:\source_models\llama_1b & qwen_0.5b headers match NVMe snapshots
    flagship_st = Path(v8.FLAGSHIP_SNAP) / "model.safetensors"
    mini_st = Path(v8.MINI_SNAP) / "model.safetensors"

    for label, e_path, c_path in [
        ("llama_1b", E_LLAMA_ST, flagship_st),
        ("qwen_0.5b", E_QWEN_ST, mini_st),
    ]:
        _, h_e = read_safetensors_header(e_path)
        _, h_c = read_safetensors_header(c_path)
        if h_e != h_c:
            raise AssertionError(f"Header mismatch between {e_path} and {c_path}")
        logger.info("[BASE CHECK] %s exact safetensors header match verified (%d keys)", label, len(h_c) - 1)

    # 2. Extract Llama-3.2-1B clean donor tensors + check finiteness
    llama_layers = [4, 7, 10, 12, 14, 15]
    llama_tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(flagship_st), framework="pt", device="cpu") as f:
        total_params = 0
        for k in f.keys():
            t = f.get_tensor(k)
            total_params += t.numel()
            if not torch.isfinite(t.float()).all():
                raise ValueError(f"Non-finite tensor in clean Llama-1B: {k}")
        llama_tensors["embed_tokens"] = f.get_tensor("model.embed_tokens.weight").half().clone()
        for l in llama_layers:
            llama_tensors[f"L{l}.o_proj"] = f.get_tensor(f"model.layers.{l}.self_attn.o_proj.weight").half().clone()
            llama_tensors[f"L{l}.down_proj"] = f.get_tensor(f"model.layers.{l}.mlp.down_proj.weight").half().clone()
    torch.save(llama_tensors, DONOR_CACHE_DIR / "llama_1b_donors.pt")
    report["llama_1b"] = {"tensors": 146, "params": total_params, "embed": verify_tensor_stats("llama.embed", llama_tensors["embed_tokens"])}
    del llama_tensors
    gc.collect()

    # 3. Extract Qwen2.5-0.5B clean donor tensors + check finiteness
    qwen_layers = [6, 10, 14, 18, 21, 23]
    qwen_tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(mini_st), framework="pt", device="cpu") as f:
        total_params = 0
        for k in f.keys():
            t = f.get_tensor(k)
            total_params += t.numel()
            if not torch.isfinite(t.float()).all():
                raise ValueError(f"Non-finite tensor in clean Qwen-0.5B: {k}")
        qwen_tensors["embed_tokens"] = f.get_tensor("model.embed_tokens.weight").half().clone()
        for l in qwen_layers:
            qwen_tensors[f"L{l}.o_proj"] = f.get_tensor(f"model.layers.{l}.self_attn.o_proj.weight").half().clone()
            qwen_tensors[f"L{l}.down_proj"] = f.get_tensor(f"model.layers.{l}.mlp.down_proj.weight").half().clone()
    torch.save(qwen_tensors, DONOR_CACHE_DIR / "qwen_0.5b_donors.pt")
    report["qwen_0.5b"] = {"tensors": 290, "params": total_params, "embed": verify_tensor_stats("qwen.embed", qwen_tensors["embed_tokens"])}
    del qwen_tensors
    gc.collect()

    # 4. Extract Falcon3-1B clean donor tensors from BLOB_DRAGON
    reader_d = gguf.GGUFReader(v8.BLOB_DRAGON)
    tmap_d = {t.name: t for t in reader_d.tensors}
    dragon_layers = [4, 7, 10, 13, 15, 17]
    dragon_tensors: Dict[str, torch.Tensor] = {}
    t_emb = tmap_d["token_embd.weight"]
    dragon_tensors["embed_tokens"] = torch.from_numpy(gguf.quants.dequantize(t_emb.data, t_emb.tensor_type).copy()).half()
    verify_tensor_stats("dragon.embed", dragon_tensors["embed_tokens"])
    for l in dragon_layers:
        t_o = tmap_d[f"blk.{l}.attn_output.weight"]
        t_dn = tmap_d[f"blk.{l}.ffn_down.weight"]
        dragon_tensors[f"L{l}.o_proj"] = torch.from_numpy(gguf.quants.dequantize(t_o.data, t_o.tensor_type).copy()).half()
        dragon_tensors[f"L{l}.down_proj"] = torch.from_numpy(gguf.quants.dequantize(t_dn.data, t_dn.tensor_type).copy()).half()
        verify_tensor_stats(f"dragon.L{l}.o", dragon_tensors[f"L{l}.o_proj"])
        verify_tensor_stats(f"dragon.L{l}.down", dragon_tensors[f"L{l}.down_proj"])
    torch.save(dragon_tensors, DONOR_CACHE_DIR / "dragon_1b_donors.pt")
    report["dragon_1b"] = {"tensors": len(reader_d.tensors), "embed": verify_tensor_stats("dragon.embed", dragon_tensors["embed_tokens"])}
    del dragon_tensors, reader_d, tmap_d
    gc.collect()

    # 5. Selectively read & unpack BitNet b1.58 2B-4T from E:\source_models\bitnet_2b_4t
    base_off, hdr_b = read_safetensors_header(BITNET_2B4T_ST)
    bitnet_layers = [5, 10, 15, 20, 25, 29]
    bitnet_2b4t: Dict[str, torch.Tensor] = {}
    emb_meta = hdr_b["model.embed_tokens.weight"]
    bitnet_2b4t["embed_tokens"] = read_safetensors_tensor_by_offset(BITNET_2B4T_ST, base_off, emb_meta).half()
    report["bitnet_2b_4t"] = {
        "tensors": len(hdr_b) - 1,
        "embed": verify_tensor_stats("bitnet2b4t.embed", bitnet_2b4t["embed_tokens"]),
    }
    for l in bitnet_layers:
        for proj in ("self_attn.o_proj", "mlp.down_proj"):
            pk = read_safetensors_tensor_by_offset(BITNET_2B4T_ST, base_off, hdr_b[f"model.layers.{l}.{proj}.weight"])
            sc = read_safetensors_tensor_by_offset(BITNET_2B4T_ST, base_off, hdr_b[f"model.layers.{l}.{proj}.weight_scale"])
            unpacked = unpack_bitnet_2b4t_weight(pk, sc).half()
            short_proj = "o_proj" if "o_proj" in proj else "down_proj"
            bitnet_2b4t[f"L{l}.{short_proj}"] = unpacked
            verify_tensor_stats(f"bitnet2b4t.L{l}.{short_proj}", unpacked)
    torch.save(bitnet_2b4t, DONOR_CACHE_DIR / "bitnet_2b4t_donors.pt")
    del bitnet_2b4t
    gc.collect()

    # 6. Verify & extract cached BitNet-Large (C:\ NVMe) as supplementary 1.58b donor
    bn_matches = list(BITNET_LARGE_HUB.rglob("model.safetensors"))
    if bn_matches:
        bn_layers = [6, 11, 16, 21]
        bn_tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(bn_matches[0]), framework="pt", device="cpu") as f:
            bn_tensors["embed_tokens"] = f.get_tensor("model.embed_tokens.weight").half().clone()
            for l in bn_layers:
                bn_tensors[f"L{l}.o_proj"] = f.get_tensor(f"model.layers.{l}.self_attn.o_proj.weight").half().clone()
                bn_tensors[f"L{l}.down_proj"] = f.get_tensor(f"model.layers.{l}.mlp.down_proj.weight").half().clone()
        torch.save(bn_tensors, DONOR_CACHE_DIR / "bitnet_large_donors.pt")
        report["bitnet_large"] = {"embed": verify_tensor_stats("bitnet_large.embed", bn_tensors["embed_tokens"])}
        del bn_tensors
        gc.collect()

    logger.info("Stage A base verification & donor caching complete: %s", json.dumps(report))
    return report


def _norm_text(s: str) -> str:
    return " ".join(re.sub(r"\s+", " ", s.lower()).strip().split())


def build_contamination_blocklist() -> Tuple[Set[str], Set[str]]:
    """
    Collects normalized prefixes AND 8-word n-grams from ALL official eval/test splits,
    HELD_OUT_PPL_TEXT, and HELD_OUT_SUITE so zero eval items can leak into training/calib.
    """
    substrings: Set[str] = set()
    ngrams: Set[str] = set()

    def _add_doc(text: str) -> None:
        nt = _norm_text(text)
        if len(nt) >= 24:
            substrings.add(nt[:64])
        words = nt.split()
        for i in range(0, max(0, len(words) - 7)):
            ngrams.add(" ".join(words[i : i + 8]))

    _add_doc(HELD_OUT_PPL_TEXT)
    for item in HELD_OUT_SUITE:
        _add_doc(item.get("prompt", ""))

    for row in load_dataset("openai/openai_humaneval", split="test"):
        _add_doc(row["prompt"])
    for row in load_dataset("openai/gsm8k", "main", split="test"):
        _add_doc(row["question"])
    for row in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
        _add_doc(row["question"])
    for row in load_dataset("cais/mmlu", "all", split="test"):
        _add_doc(row["question"])

    logger.info("Built zero-contamination blocklist: %d prefixes, %d 8-grams", len(substrings), len(ngrams))
    return substrings, ngrams


def is_contaminated(text: str, substrings: Set[str], ngrams: Set[str]) -> bool:
    nt = _norm_text(text)
    if len(nt) >= 24 and nt[:64] in substrings:
        return True
    words = nt.split()
    for i in range(0, max(0, len(words) - 7)):
        if " ".join(words[i : i + 8]) in ngrams:
            return True
    return False


def build_clean_non_test_datasets() -> Dict[str, Any]:
    """
    Builds strictly disjoint non-test calibration prose + training datasets from:
      - openai/gsm8k:train
      - allenai/ai2_arc:ARC-Challenge:train
      - cais/mmlu:auxiliary_train + dev + validation
      - non-test concise Python algorithms & BFCL JSON tool-calling examples
      - clean technical prose (strictly disjoint from HELD_OUT_PPL_TEXT)
    """
    substrings, ngrams = build_contamination_blocklist()
    rng = np.random.default_rng(2026)

    # 1. Clean technical & STEM anchor/validation prose (NOT HELD_OUT_PPL_TEXT)
    calib_prose = [
        (
            "In statistical mechanics, the partition function Z encodes the microscopic states of a "
            "canonical ensemble at inverse temperature beta = 1 / (k_B T), from which Helmholtz free "
            "energy F = -k_B T ln Z and entropy S follow directly by differentiation."
        ),
        (
            "Out-of-order superscalar processors use Tomasulo's algorithm with register renaming and "
            "a reorder buffer to eliminate write-after-read and write-after-write hazards while "
            "committing instructions strictly in program order for precise exceptions."
        ),
        (
            "In linear algebra, the singular value decomposition factors any real matrix M into U Sigma V^T, "
            "providing the optimal rank-k Eckart-Young approximation in both spectral and Frobenius norms "
            "without altering the orthogonal row or column bases."
        ),
        (
            "Cryptographic hash functions such as SHA-256 rely on the Merkle-Damgard construction paired "
            "with a Davies-Meyer compression function to guarantee preimage, second-preimage, and "
            "collision resistance across arbitrary-length inputs."
        ),
        (
            "Byzantine fault-tolerant state machine replication requires at least 3f + 1 replicas to "
            "tolerate f arbitrary faulty nodes, ensuring safety and liveness across asynchronous message "
            "delays via quorum intersection."
        ),
        (
            "In cellular metabolism, oxidative phosphorylation couples electron transport along the inner "
            "mitochondrial membrane to an electrochemical proton gradient that drives ATP synthase rotary "
            "catalysis."
        ),
    ]
    for p in calib_prose:
        if is_contaminated(p, substrings, ngrams):
            raise AssertionError(f"Contaminated calib_prose item detected: {p[:60]}")

    # 2. GSM8K train (step-by-step + strict 'Final answer: <number>')
    gsm_samples: List[Dict[str, str]] = []
    gsm_ds = list(load_dataset("openai/gsm8k", "main", split="train"))
    idx_gsm = rng.permutation(len(gsm_ds))[:1800]
    for idx in idx_gsm:
        row = gsm_ds[int(idx)]
        q, a = row["question"].strip(), row["answer"].strip()
        if is_contaminated(q, substrings, ngrams):
            continue
        if "####" not in a:
            continue
        rationale, final_num = a.split("####", 1)
        clean_num = final_num.strip().replace(",", "")
        rationale_clean = re.sub(r"<<[^>]+>>", "", rationale).strip()
        user = (
            f"{q}\n\nSolve step by step. On the last line write "
            f"'Final answer: <number>' with only the number."
        )
        asst = f"{rationale_clean}\n\nFinal answer: {clean_num}"
        gsm_samples.append({"cat": "gsm8k", "user": user, "assistant": asst})

    # 3. ARC-Challenge train (both direct letter & brief reasoning)
    arc_samples: List[Dict[str, str]] = []
    arc_ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"))
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
    for row in arc_ds:
        texts = row["choices"]["text"]
        if len(texts) != 4 or is_contaminated(row["question"], substrings, ngrams):
            continue
        gold = label_map.get(str(row["answerKey"]).strip(), "A")
        opts = "\n".join(f"{L}) {c}" for L, c in zip("ABCD", texts))
        user = (
            f"The following is a multiple choice science question.\n\n"
            f"{row['question']}\n{opts}\n\nRespond with only the letter of the correct answer."
        )
        arc_samples.append({"cat": "arc_direct", "user": user, "assistant": gold})

    # 4. MMLU auxiliary_train + validation + dev (direct letter & subject-aware)
    mmlu_samples: List[Dict[str, str]] = []
    for spl, cap in [("dev", 285), ("validation", 1531), ("auxiliary_train", 2500)]:
        ds_m = load_dataset("cais/mmlu", "all", split=spl)
        order = rng.permutation(len(ds_m))[:cap]
        for idx in order:
            row = ds_m[int(idx)]
            q = row["question"]
            if is_contaminated(q, substrings, ngrams):
                continue
            choices = row["choices"]
            if len(choices) != 4:
                continue
            gold = "ABCD"[int(row["answer"])]
            opts = "\n".join(f"{L}) {c}" for L, c in zip("ABCD", choices))
            subj = row.get("subject", "general knowledge").replace("_", " ")
            user = (
                f"The following is a multiple choice question about {subj}.\n\n"
                f"{q}\n{opts}\n\nRespond with only the letter of the correct answer."
            )
            mmlu_samples.append({"cat": "mmlu_direct", "user": user, "assistant": gold})

    # 5. Clean concise Python algorithm & data-structure examples (non-HumanEval)
    code_samples = [
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n"
            "    \"\"\"Merge two sorted lists in ascending order.\"\"\"\n",
            "```python\ndef merge_sorted(a: list[int], b: list[int]) -> list[int]:\n"
            "    res = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n            res.append(a[i]); i += 1\n"
            "        else:\n            res.append(b[j]); j += 1\n"
            "    res.extend(a[i:])\n    res.extend(b[j:])\n    return res\n```",
        ),
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def binary_search(nums: list[int], target: int) -> int:\n"
            "    \"\"\"Return index of target in sorted list nums, or -1 if absent.\"\"\"\n",
            "```python\ndef binary_search(nums: list[int], target: int) -> int:\n"
            "    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
            "        if nums[mid] == target:\n            return mid\n"
            "        if nums[mid] < target:\n            lo = mid + 1\n"
            "        else:\n            hi = mid - 1\n    return -1\n```",
        ),
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def count_vowels(text: str) -> int:\n"
            "    \"\"\"Count English vowels (case-insensitive) in text.\"\"\"\n",
            "```python\ndef count_vowels(text: str) -> int:\n"
            "    return sum(1 for ch in text.lower() if ch in 'aeiou')\n```",
        ),
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def flatten_matrix(grid: list[list[int]]) -> list[int]:\n"
            "    \"\"\"Flatten a 2D list of integers into a 1D list.\"\"\"\n",
            "```python\ndef flatten_matrix(grid: list[list[int]]) -> list[int]:\n"
            "    return [val for row in grid for val in row]\n```",
        ),
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def is_anagram(s1: str, s2: str) -> bool:\n"
            "    \"\"\"Return True if s1 and s2 are case-insensitive anagrams ignoring spaces.\"\"\"\n",
            "```python\ndef is_anagram(s1: str, s2: str) -> bool:\n"
            "    a = sorted(c for c in s1.lower() if not c.isspace())\n"
            "    b = sorted(c for c in s2.lower() if not c.isspace())\n"
            "    return a == b\n```",
        ),
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def unique_ordered(items: list[int]) -> list[int]:\n"
            "    \"\"\"Deduplicate items while preserving first-seen order.\"\"\"\n",
            "```python\ndef unique_ordered(items: list[int]) -> list[int]:\n"
            "    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n"
            "            seen.add(x)\n            out.append(x)\n    return out\n```",
        ),
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def run_length_encode(s: str) -> list[tuple[str, int]]:\n"
            "    \"\"\"Compress consecutive identical characters into (char, count) tuples.\"\"\"\n",
            "```python\ndef run_length_encode(s: str) -> list[tuple[str, int]]:\n"
            "    if not s:\n        return []\n    out = []\n    cur, cnt = s[0], 1\n"
            "    for ch in s[1:]:\n        if ch == cur:\n            cnt += 1\n"
            "        else:\n            out.append((cur, cnt))\n            cur, cnt = ch, 1\n"
            "    out.append((cur, cnt))\n    return out\n```",
        ),
        (
            "Write concise Python. No docstring. No comments. Do not write unit tests. "
            "Define ONLY the requested function inside a ```python block.\n\n"
            "def matrix_transpose(m: list[list[float]]) -> list[list[float]]:\n"
            "    \"\"\"Return the transpose of a non-empty rectangular matrix.\"\"\"\n",
            "```python\ndef matrix_transpose(m: list[list[float]]) -> list[list[float]]:\n"
            "    if not m:\n        return []\n    return [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]\n```",
        ),
    ]
    clean_code_items = []
    for u, a in code_samples:
        if not is_contaminated(u, substrings, ngrams):
            clean_code_items.append({"cat": "code", "user": u, "assistant": a})

    # Also harvest non-test Python coding prompts from fable_5_1_max/5k/train.jsonl (strictly filtered)
    fable_path = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\frontier_datasets\fable_5_1_max\5k\train.jsonl")
    if fable_path.exists():
        fable_added = 0
        with open(fable_path, "r", encoding="utf-8") as fp:
            for line in fp:
                if fable_added >= 200:
                    break
                row = json.loads(line)
                msgs = row.get("messages", [])
                u_txt, a_txt = "", ""
                for m in msgs:
                    if m.get("role") == "user":
                        u_txt = m.get("content", "")
                    elif m.get("role") == "assistant":
                        a_txt = m.get("content", "")
                if 40 <= len(u_txt) <= 800 and 20 <= len(a_txt) <= 1200 and "def " in a_txt:
                    if not is_contaminated(u_txt, substrings, ngrams):
                        clean_code_items.append({"cat": "code_trace", "user": u_txt, "assistant": a_txt})
                        fable_added += 1

    # 6. Clean BFCL-compatible JSON tool calling & irrelevance synthetic samples
    bfcl_samples = [
        (
            "Available tools:\n[{\"name\": \"calc_compound_interest\", \"description\": \"Compute final balance.\", "
            "\"parameters\": {\"principal\": {\"type\": \"number\"}, \"rate\": {\"type\": \"number\"}, \"years\": {\"type\": \"integer\"}}}]\n\n"
            "User request: Calculate the balance for $2500 at 0.04 annual rate over 5 years.\n"
            "Respond ONLY with JSON: {\"tool\": \"<name>\", \"arguments\": {...}} or {\"tool\": null, \"reply\": \"No tool required.\"}.",
            '{"tool": "calc_compound_interest", "arguments": {"principal": 2500, "rate": 0.04, "years": 5}}',
        ),
        (
            "Available tools:\n[{\"name\": \"fetch_weather\", \"description\": \"Get forecast by city.\", "
            "\"parameters\": {\"city\": {\"type\": \"string\"}, \"units\": {\"type\": \"string\"}}}]\n\n"
            "User request: What is the capital of France?\n"
            "Respond ONLY with JSON: {\"tool\": \"<name>\", \"arguments\": {...}} or {\"tool\": null, \"reply\": \"No tool required.\"}.",
            '{"tool": null, "reply": "No tool required."}',
        ),
        (
            "Available tools:\n[{\"name\": \"convert_units\", \"description\": \"Convert physical units.\", "
            "\"parameters\": {\"value\": {\"type\": \"number\"}, \"from_unit\": {\"type\": \"string\"}, \"to_unit\": {\"type\": \"string\"}}}]\n\n"
            "User request: Convert 12.5 kilometers to meters.\n"
            "Respond ONLY with JSON: {\"tool\": \"<name>\", \"arguments\": {...}} or {\"tool\": null, \"reply\": \"No tool required.\"}.",
            '{"tool": "convert_units", "arguments": {"value": 12.5, "from_unit": "kilometers", "to_unit": "meters"}}',
        ),
        (
            "Available tools:\n[{\"name\": \"query_inventory\", \"description\": \"Check stock count.\", "
            "\"parameters\": {\"sku\": {\"type\": \"string\"}}}]\n\n"
            "User request: Explain why water expands when freezing.\n"
            "Respond ONLY with JSON: {\"tool\": \"<name>\", \"arguments\": {...}} or {\"tool\": null, \"reply\": \"No tool required.\"}.",
            '{"tool": null, "reply": "No tool required."}',
        ),
    ]
    clean_bfcl = [{"cat": "bfcl", "user": u, "assistant": a} for u, a in bfcl_samples if not is_contaminated(u, substrings, ngrams)]

    # 7. Sovereign identity anchors (disjoint from HELD_OUT_SUITE)
    identity_samples = [
        {
            "cat": "identity",
            "user": "State your name and origin cleanly.",
            "assistant": "I am Quillan-Ronin, a sovereign hierarchical AI engineered by CrashOverrideX.",
        },
        {
            "cat": "identity",
            "user": "Who created and architected your reasoning framework?",
            "assistant": "I am Quillan-Ronin, architected and trained by CrashOverrideX.",
        },
    ]

    payload = {
        "calib_prose": calib_prose,
        "gsm8k": gsm_samples,
        "arc_direct": arc_samples,
        "mmlu_direct": mmlu_samples,
        "code": clean_code_items,
        "bfcl": clean_bfcl,
        "identity": identity_samples,
    }
    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    counts = {k: len(v) for k, v in payload.items()}
    logger.info("Saved verified zero-contamination dataset to %s: %s", CLEAN_DATA_PATH, counts)
    return counts


def main() -> None:
    report = verify_clean_bases_and_extract_donors()
    counts = build_clean_non_test_datasets()
    print("STAGE_A_REPORT:", json.dumps({"bases": report, "dataset_counts": counts}))


if __name__ == "__main__":
    main()
