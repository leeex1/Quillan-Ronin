#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI — PRISTINE GOLD DATASET CURATOR & PURIFIER
===================================================================
Applies rigorous statistical quality filters to the master corpus (v10_unified_master)
to eliminate:
  1. Pad / EOS token spikes (>50% pad/eos in a sequence).
  2. Degenerate token repetitions (>39% identical single token in a sequence).
  3. Low-entropy degenerative loops (Shannon entropy < 3.5 bits).
  4. Severe N-gram repetition loops (>50% 4-gram repetition).

Output:
  c:\\02_QUILLAN\\05_Training\\training_data\\v10_pristine_gold\\
    ├── train_ids.bin     (uint16 memmap, ~260M+ pristine tokens)
    ├── train_labels.bin  (int32 memmap, ~260M+ pristine tokens)
    ├── val_ids.bin       (uint16 memmap, ~2.6M pristine validation tokens)
    ├── val_labels.bin    (int32 memmap, ~2.6M pristine validation tokens)
    └── stats.json        (complete provenance, retention rate, defect breakdown)
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOGGER = logging.getLogger("PristineGoldCurator")

REPO_ROOT = Path(r"C:\02_QUILLAN")
SRC_DIR = REPO_ROOT / "05_Training" / "training_data" / "v10_unified_master"
OUT_DIR = REPO_ROOT / "05_Training" / "training_data" / "v10_pristine_gold"
SEQ_LEN = 512
VOCAB_SIZE = 50257
CHUNK_SIZE = 5000  # Process in chunks of 5,000 sequences


def evaluate_sequence(seq: np.ndarray) -> bool:
    """
    Evaluates sequence against multi-factor quality criteria.
    Returns True if sequence meets high quality standard, False otherwise.
    """
    # 1. Pad / EOS spike check
    pad_count = int(((seq == 0) | (seq == 1)).sum())
    if pad_count > 256:  # >50% pad
        return False

    # 2. Fast histogram
    counts = np.bincount(seq)
    if counts.max() > 200:  # >39% single identical token
        return False

    # 3. Shannon token entropy
    c_nonzero = counts[counts > 0]
    p = c_nonzero / float(SEQ_LEN)
    entropy = -float(np.sum(p * np.log2(p)))
    if entropy < 3.5:  # Degenerate low-entropy sequence
        return False

    # 4. 4-gram repetition (checked on marginal entropy < 5.5 bits)
    if entropy < 5.5:
        ngrams = [tuple(seq[i:i+4]) for i in range(SEQ_LEN - 3)]
        unique_ngrams = len(set(ngrams))
        rep_rate = 1.0 - (unique_ngrams / float(len(ngrams)))
        if rep_rate > 0.50:  # >50% repetition loop
            return False

    return True


def filter_split(split_name: str, src_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Filters a binary split (train or val) and writes out pristine binaries."""
    src_ids_path = src_dir / f"{split_name}_ids.bin"
    if not src_ids_path.exists():
        LOGGER.warning("Source split not found: %s", src_ids_path)
        return {}

    ids_mmap = np.memmap(src_ids_path, dtype=np.uint16, mode="r")
    total_seqs = len(ids_mmap) // SEQ_LEN
    LOGGER.info("[*] Filtering %s split: %s sequences (%s tokens)...", split_name, f"{total_seqs:,}", f"{len(ids_mmap):,}")

    out_ids_path = out_dir / f"{split_name}_ids.bin"
    out_labels_path = out_dir / f"{split_name}_labels.bin"

    f_ids = open(out_ids_path, "wb")
    f_labels = open(out_labels_path, "wb")

    accepted_seqs = 0
    rejected_seqs = 0
    t0 = time.time()

    for start_idx in range(0, total_seqs, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, total_seqs)
        chunk = ids_mmap[start_idx * SEQ_LEN : end_idx * SEQ_LEN].reshape(-1, SEQ_LEN)

        accepted_chunk = []
        for s in chunk:
            if evaluate_sequence(s):
                accepted_chunk.append(s)
                accepted_seqs += 1
            else:
                rejected_seqs += 1

        if accepted_chunk:
            acc_arr = np.array(accepted_chunk, dtype=np.uint16)
            f_ids.write(acc_arr.tobytes())
            f_labels.write(acc_arr.astype(np.int32).tobytes())

        if (end_idx % 50000 == 0) or (end_idx == total_seqs):
            pct = (end_idx / total_seqs) * 100.0
            cur_acc_pct = (accepted_seqs / max(1, accepted_seqs + rejected_seqs)) * 100.0
            LOGGER.info("    Progress: %5.1f%% (%s / %s seqs) | Accepted: %s (%.2f%%) | Rejected: %s",
                        pct, f"{end_idx:,}", f"{total_seqs:,}", f"{accepted_seqs:,}", cur_acc_pct, f"{rejected_seqs:,}")

    f_ids.flush()
    f_labels.flush()
    f_ids.close()
    f_labels.close()

    elapsed = time.time() - t0
    retention_rate = (accepted_seqs / total_seqs) * 100.0
    accepted_tokens = accepted_seqs * SEQ_LEN
    rejected_tokens = rejected_seqs * SEQ_LEN

    LOGGER.info("[+] %s split complete in %.2fs: %s tokens retained (%.2f%%), %s rejected.",
                split_name.upper(), elapsed, f"{accepted_tokens:,}", retention_rate, f"{rejected_tokens:,}")

    return {
        "total_source_seqs": total_seqs,
        "accepted_seqs": accepted_seqs,
        "rejected_seqs": rejected_seqs,
        "accepted_tokens": accepted_tokens,
        "rejected_tokens": rejected_tokens,
        "retention_rate_pct": round(retention_rate, 2),
        "elapsed_s": round(elapsed, 2)
    }


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("   👑 STARTING PRISTINE GOLD DATASET CURATION & PURIFICATION")
    LOGGER.info("   Source: %s", SRC_DIR)
    LOGGER.info("   Target: %s", OUT_DIR)
    LOGGER.info("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    train_stats = filter_split("train", SRC_DIR, OUT_DIR)
    val_stats = filter_split("val", SRC_DIR, OUT_DIR)

    total_tokens = train_stats.get("accepted_tokens", 0) + val_stats.get("accepted_tokens", 0)
    total_elapsed = time.time() - start_time

    manifest = {
        "status": "PRISTINE_GOLD_CERTIFIED",
        "description": "High-standard curated corpus with low-entropy, padding spikes, and n-gram loops excised.",
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "total_pristine_tokens": total_tokens,
        "train_pristine_tokens": train_stats.get("accepted_tokens", 0),
        "val_pristine_tokens": val_stats.get("accepted_tokens", 0),
        "train_pristine_seqs": train_stats.get("accepted_seqs", 0),
        "val_pristine_seqs": val_stats.get("accepted_seqs", 0),
        "total_rejected_tokens": train_stats.get("rejected_tokens", 0) + val_stats.get("rejected_tokens", 0),
        "overall_retention_rate_pct": round(
            total_tokens / max(1, (train_stats.get("total_source_seqs", 0) + val_stats.get("total_source_seqs", 0)) * SEQ_LEN) * 100.0, 2
        ),
        "filter_criteria": {
            "max_pad_ratio": 0.50,
            "max_single_token_ratio": 0.39,
            "min_shannon_entropy_bits": 3.5,
            "max_4gram_repetition_ratio": 0.50
        },
        "train_split": train_stats,
        "val_split": val_stats,
        "total_elapsed_s": round(total_elapsed, 2)
    }

    stats_path = OUT_DIR / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("   🏆 PRISTINE GOLD CORPUS READY: %s TOKENS", f"{total_tokens:,}")
    LOGGER.info("   Train: %s tokens (%s seqs of %d)", f"{manifest['train_pristine_tokens']:,}", f"{manifest['train_pristine_seqs']:,}", SEQ_LEN)
    LOGGER.info("   Val:   %s tokens (%s seqs of %d)", f"{manifest['val_pristine_tokens']:,}", f"{manifest['val_pristine_seqs']:,}", SEQ_LEN)
    LOGGER.info("   Retention Rate: %.2f%% (%s defective tokens purged)",
                manifest["overall_retention_rate_pct"], f"{manifest['total_rejected_tokens']:,}")
    LOGGER.info("   Total Time: %.2f seconds", total_elapsed)
    LOGGER.info("   Manifest Saved: %s", stats_path)
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
