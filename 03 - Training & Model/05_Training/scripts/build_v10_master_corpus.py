#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI — UNIFIED MASTER CORPUS COMPILER (v10)
================================================================
Compiles 100% of all scattered training corpora (~285M+ tokens) into high-throughput,
memory-mapped binary files (uint16 IDs and int32 Labels) at uniform 512 sequence length.

Ingestion Hierarchy:
  1. High-Density 1D Pre-Tokenized Tensors (.pt):
     - quillan_corpus_CLEAN_V7.pt (166.3M tokens)
     - clean_unified_multi_frontier.pt (24.1M tokens)
     - full_train.pt (16.0M tokens)
     - instruct_train.pt (13.5M tokens)
     - GPT_5.5_Distilled.pt (10.7M tokens)
     - train.pt (3.3M tokens)
     - code_train.pt (3.0M tokens)
     - quillan_12mb_training_dataset.pt (1.7M tokens)
     - quillan_science_additional.pt (1.0M tokens)
     - quillan_science_absolute.pt (0.6M tokens)
     - full_dataset.pt (0.3M tokens)
  2. Batched 2D Tensor Dictionaries (.pt):
     - frontier_intact_gold_master.pt (14.8M tokens)
     - pristine_frontier_gold_37k.pt (9.6M tokens)
     - quillan_tokenized.pt (6.9M tokens)
     - pristine_canonical_gold_sft.pt (3.7M tokens)
     - augmented_frontier_v2.pt (3.4M tokens)
     - intact_pair_dataset.pt (2.1M tokens)
     - intact_thought_reasoning_gold.pt (1.2M tokens)
     - omniformat_gold_dataset.pt (0.6M tokens)
  3. Specialized Domain Knowledge JSONL Corpora:
     - All 34 Council Expert files in experts_34/ (C0-ASTRA to C33-PREDATOR)
     - pdf_papers_corpus.jsonl (NVFP4, BitNet, MoE, Reasoning papers)
     - Quillan_Clean_Reasoning_Gold_Dataset.jsonl

Output:
  c:\\02_QUILLAN\\05_Training\\training_data\\v10_unified_master\\
    ├── train_ids.bin     (uint16 memmap, ~280M+ tokens)
    ├── train_labels.bin  (int32 memmap, ~280M+ tokens)
    ├── val_ids.bin       (uint16 memmap, ~2.8M tokens held-out)
    ├── val_labels.bin    (int32 memmap, ~2.8M tokens held-out)
    └── stats.json        (complete provenance & token manifest)
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Generator

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger("CorpusCompilerV10")

REPO_ROOT = Path(r"C:\02_QUILLAN")
DATA_DIR = REPO_ROOT / "05_Training" / "training_data"
OUT_DIR = DATA_DIR / "v10_unified_master"
SEQ_LEN = 512
VOCAB_SIZE = 50257
VAL_RATIO = 0.01  # 1% validation split (~2.8M tokens)

# Add oni directory to path for UnifiedQuillanTokenizer
sys.path.insert(0, str(REPO_ROOT / "00 - Meta" / "oni"))
from quillan_tokenizer_unified import UnifiedQuillanTokenizer


def sanitize_token_array(arr: np.ndarray, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Clamps token array to valid vocabulary range [0, vocab_size - 1]."""
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, vocab_size - 1).astype(np.uint16)
    else:
        if (arr >= vocab_size).any():
            arr = np.clip(arr, 0, vocab_size - 1)
    return arr


class MasterCorpusWriter:
    """Streams token chunks directly to disk binary files with deterministic buffer management."""

    def __init__(self, out_dir: Path, seq_len: int = SEQ_LEN, val_ratio: float = VAL_RATIO):
        self.out_dir = out_dir
        self.seq_len = seq_len
        self.val_ratio = val_ratio
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.train_ids_path = out_dir / "train_ids.bin"
        self.train_labels_path = out_dir / "train_labels.bin"
        self.val_ids_path = out_dir / "val_ids.bin"
        self.val_labels_path = out_dir / "val_labels.bin"

        self.f_train_ids = open(self.train_ids_path, "wb")
        self.f_train_labels = open(self.train_labels_path, "wb")
        self.f_val_ids = open(self.val_ids_path, "wb")
        self.f_val_labels = open(self.val_labels_path, "wb")

        self.buffer = np.empty(0, dtype=np.uint16)
        self.total_train_tokens = 0
        self.total_val_tokens = 0
        self.val_interval = int(1.0 / val_ratio)
        self.seq_counter = 0

    def feed_tokens(self, tokens: np.ndarray):
        """Append raw 1D token array to buffer and flush whole 512-sequences."""
        if len(tokens) == 0:
            return

        tokens = sanitize_token_array(tokens)
        self.buffer = np.concatenate([self.buffer, tokens])

        n_full = (len(self.buffer) // self.seq_len) * self.seq_len
        if n_full == 0:
            return

        ready = self.buffer[:n_full]
        self.buffer = self.buffer[n_full:]

        sequences = ready.reshape(-1, self.seq_len)
        for seq in sequences:
            self.seq_counter += 1
            is_val = (self.seq_counter % self.val_interval == 0)

            seq_uint16 = seq.astype(np.uint16)
            seq_int32 = seq.astype(np.int32)

            if is_val:
                self.f_val_ids.write(seq_uint16.tobytes())
                self.f_val_labels.write(seq_int32.tobytes())
                self.total_val_tokens += self.seq_len
            else:
                self.f_train_ids.write(seq_uint16.tobytes())
                self.f_train_labels.write(seq_int32.tobytes())
                self.total_train_tokens += self.seq_len

    def finalize(self):
        """Pad any remaining tokens in buffer with EOS (0) to complete the last sequence."""
        rem = len(self.buffer)
        if rem > 0:
            pad_len = self.seq_len - rem
            padded = np.pad(self.buffer, (0, pad_len), mode="constant", constant_values=0)
            padded_uint16 = padded.astype(np.uint16)
            padded_int32 = padded.astype(np.int32)
            self.f_train_ids.write(padded_uint16.tobytes())
            self.f_train_labels.write(padded_int32.tobytes())
            self.total_train_tokens += self.seq_len
            self.buffer = np.empty(0, dtype=np.uint16)

        self.f_train_ids.flush()
        self.f_train_labels.flush()
        self.f_val_ids.flush()
        self.f_val_labels.flush()

        self.f_train_ids.close()
        self.f_train_labels.close()
        self.f_val_ids.close()
        self.f_val_labels.close()


def compile_v10_master():
    LOGGER.info("=" * 70)
    LOGGER.info("   👑 COMPILING QUILLAN-RONIN v10 UNIFIED MASTER CORPUS")
    LOGGER.info("   Target: %s", OUT_DIR)
    LOGGER.info("=" * 70)

    start_time = time.time()
    writer = MasterCorpusWriter(OUT_DIR, seq_len=SEQ_LEN, val_ratio=VAL_RATIO)
    stats: Dict[str, Any] = {
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "sources": {},
    }

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: PRE-TOKENIZED 1D TENSORS (.pt)
    # ──────────────────────────────────────────────────────────────────────────
    pt_1d_sources = [
        "quillan_corpus_CLEAN_V7.pt",
        "clean_unified_multi_frontier.pt",
        "full_train.pt",
        "instruct_train.pt",
        "GPT_5.5_Distilled.pt",
        "train.pt",
        "code_train.pt",
        "quillan_12mb_training_dataset.pt",
        "quillan_science_additional.pt",
        "quillan_science_absolute.pt",
        "full_dataset.pt",
    ]

    LOGGER.info("\n[*] [1/3] Ingesting 1D Pre-Tokenized Tensor Corpora...")
    for fn in pt_1d_sources:
        fp = DATA_DIR / fn
        if not fp.exists():
            LOGGER.warning("  [-] File not found: %s (skipped)", fn)
            stats["sources"][fn] = "not_found"
            continue

        LOGGER.info("  [+] Loading %s...", fn)
        t0 = time.time()
        try:
            tensor = torch.load(fp, map_location="cpu", weights_only=False)
            if isinstance(tensor, torch.Tensor):
                arr = tensor.numpy().flatten()
                count = len(arr)
                writer.feed_tokens(arr)
                elapsed = time.time() - t0
                stats["sources"][fn] = f"{count:,} tokens ({elapsed:.2f}s)"
                LOGGER.info("      Ingested %s tokens in %.2fs", f"{count:,}", elapsed)
            else:
                LOGGER.warning("      Unexpected type %s in %s", type(tensor), fn)
                stats["sources"][fn] = f"skipped: {type(tensor)}"
        except Exception as err:
            LOGGER.error("      Error loading %s: %s", fn, err)
            stats["sources"][fn] = f"error: {err}"

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: BATCHED 2D TENSOR DICTIONARIES (.pt)
    # ──────────────────────────────────────────────────────────────────────────
    pt_2d_sources = [
        ("frontier_intact_gold_master.pt", "input_ids"),
        ("pristine_frontier_gold_37k.pt", "input_ids"),
        ("quillan_tokenized.pt", "tokens"),
        ("pristine_canonical_gold_sft.pt", "input_ids"),
        ("augmented_frontier_v2.pt", "input_ids"),
        ("intact_pair_dataset.pt", "input_ids"),
        ("intact_thought_reasoning_gold.pt", "input_ids"),
        ("omniformat_gold_dataset.pt", "input_ids"),
    ]

    LOGGER.info("\n[*] [2/3] Ingesting Batched 2D Tensor Dictionaries...")
    for fn, key in pt_2d_sources:
        fp = DATA_DIR / fn
        if not fp.exists():
            LOGGER.warning("  [-] File not found: %s (skipped)", fn)
            stats["sources"][fn] = "not_found"
            continue

        LOGGER.info("  [+] Loading %s [%s]...", fn, key)
        t0 = time.time()
        try:
            data = torch.load(fp, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and key in data:
                t = data[key]
                arr = t.numpy().flatten()
                count = len(arr)
                writer.feed_tokens(arr)
                elapsed = time.time() - t0
                stats["sources"][fn] = f"{count:,} tokens (shape {list(t.shape)})"
                LOGGER.info("      Ingested %s tokens (shape %s) in %.2fs", f"{count:,}", list(t.shape), elapsed)
            else:
                LOGGER.warning("      Key %s not found in %s (keys: %s)", key, fn, list(data.keys()) if isinstance(data, dict) else type(data))
                stats["sources"][fn] = f"skipped: key {key} missing"
        except Exception as err:
            LOGGER.error("      Error loading %s: %s", fn, err)
            stats["sources"][fn] = f"error: {err}"

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: SPECIALIZED DOMAIN EXPERT & PAPER JSONL CORPORA
    # ──────────────────────────────────────────────────────────────────────────
    LOGGER.info("\n[*] [3/3] Tokenizing Specialized Domain Corpora (UnifiedQuillanTokenizer)...")
    tokenizer = UnifiedQuillanTokenizer()

    # A. 34 Council Experts
    experts_dir = DATA_DIR / "experts_34"
    if experts_dir.exists():
        exp_files = sorted(experts_dir.glob("*.jsonl"))
        LOGGER.info("  [+] Ingesting %d Council Expert domain files from experts_34/...", len(exp_files))
        exp_tokens_total = 0
        t_exp = time.time()
        for ef in exp_files:
            file_tokens: List[int] = []
            with open(ef, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        q = record.get("question") or record.get("prompt") or record.get("instruction") or ""
                        a = record.get("response") or record.get("answer") or record.get("output") or ""
                        if q and a:
                            text = f"<|user|>\n{q}\n<|assistant|>\n{a}<|endoftext|>"
                            toks = tokenizer.encode(text, domain="general")
                            file_tokens.extend(toks)
                    except Exception:
                        continue

            if file_tokens:
                arr = np.array(file_tokens, dtype=np.uint16)
                writer.feed_tokens(arr)
                exp_tokens_total += len(file_tokens)

        stats["sources"]["experts_34"] = f"34 files, {exp_tokens_total:,} tokens ({time.time()-t_exp:.2f}s)"
        LOGGER.info("      Ingested %s tokens across all 34 Council Experts in %.2fs", f"{exp_tokens_total:,}", time.time()-t_exp)

    # B. Research Papers Corpus
    papers_file = DATA_DIR / "pdf_papers_corpus.jsonl"
    if papers_file.exists():
        LOGGER.info("  [+] Tokenizing %s...", papers_file.name)
        t_p = time.time()
        paper_tokens: List[int] = []
        with open(papers_file, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    txt = record.get("text", "")
                    if len(txt) > 50:
                        toks = tokenizer.encode(txt + "<|endoftext|>", domain="scientific")
                        paper_tokens.extend(toks)
                except Exception:
                    continue

        if paper_tokens:
            arr = np.array(paper_tokens, dtype=np.uint16)
            writer.feed_tokens(arr)
            stats["sources"]["pdf_papers_corpus.jsonl"] = f"{len(paper_tokens):,} tokens ({time.time()-t_p:.2f}s)"
            LOGGER.info("      Ingested %s research paper tokens in %.2fs", f"{len(paper_tokens):,}", time.time()-t_p)

    # C. Clean Reasoning Gold Dataset
    gold_file = DATA_DIR / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl"
    if gold_file.exists():
        LOGGER.info("  [+] Tokenizing %s...", gold_file.name)
        t_g = time.time()
        gold_tokens: List[int] = []
        with open(gold_file, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    q = record.get("question", "")
                    r = record.get("response", "")
                    if q and r:
                        txt = f"<|user|>\n{q}\n<|assistant|>\n{r}<|endoftext|>"
                        toks = tokenizer.encode(txt, domain="general")
                        gold_tokens.extend(toks)
                except Exception:
                    continue

        if gold_tokens:
            arr = np.array(gold_tokens, dtype=np.uint16)
            writer.feed_tokens(arr)
            stats["sources"]["Quillan_Clean_Reasoning_Gold_Dataset.jsonl"] = f"{len(gold_tokens):,} tokens ({time.time()-t_g:.2f}s)"
            LOGGER.info("      Ingested %s reasoning gold tokens in %.2fs", f"{len(gold_tokens):,}", time.time()-t_g)

    # ──────────────────────────────────────────────────────────────────────────
    # FINALIZE & EMIT STATS
    # ──────────────────────────────────────────────────────────────────────────
    writer.finalize()
    total_time = time.time() - start_time

    train_seqs = writer.total_train_tokens // SEQ_LEN
    val_seqs = writer.total_val_tokens // SEQ_LEN
    total_tokens = writer.total_train_tokens + writer.total_val_tokens

    stats.update({
        "total_tokens": total_tokens,
        "train_tokens": writer.total_train_tokens,
        "val_tokens": writer.total_val_tokens,
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "elapsed_s": round(total_time, 2),
        "tokenizer": "UnifiedQuillanTokenizer (custom BPE 50257)",
    })

    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("   🏆 COMPILATION COMPLETE: %s TOTAL TOKENS", f"{total_tokens:,}")
    LOGGER.info("   Train: %s tokens (%s seqs of %d)", f"{writer.total_train_tokens:,}", f"{train_seqs:,}", SEQ_LEN)
    LOGGER.info("   Val:   %s tokens (%s seqs of %d)", f"{writer.total_val_tokens:,}", f"{val_seqs:,}", SEQ_LEN)
    LOGGER.info("   Elapsed Time: %.2f seconds", total_time)
    LOGGER.info("   Manifest Saved: %s", stats_path)
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    compile_v10_master()
