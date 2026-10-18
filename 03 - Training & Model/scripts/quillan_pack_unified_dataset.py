#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN PRE-TOKENIZED DATASET PACKER
=============================================
Packs normalized canonical JSONL dataset into contiguous PyTorch tensors
loadable with maximum zero-copy memory throughput by the training pipeline:
  - Sequence shape: [N, max_seq_len]
  - Label masking: prompt tokens set to -100 (loss computed solely on reasoning & answer)
  - Strict weights_only=True deserialization compatibility
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import torch

REPO_ROOT: Final[Path] = Path(r"C:\02_QUILLAN")
MODEL_DIR: Final[Path] = REPO_ROOT / "03 - Training & Model"
DATA_DIR: Final[Path] = REPO_ROOT / "training_data"
CANONICAL_DIR: Final[Path] = DATA_DIR / "canonical_standardized"

for p in [str(REPO_ROOT), str(MODEL_DIR), str(REPO_ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_bpe_tokenizer import QuillanBPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER: Final[logging.Logger] = logging.getLogger("quillan_pack_unified_dataset")


def pack_dataset(
    jsonl_path: Path,
    output_pt_path: Path,
    max_seq_len: int = 512,
    max_samples: Optional[int] = None,
) -> Path:
    """Tokenizes and serializes canonical JSONL dataset into PyTorch tensor pack."""
    LOGGER.info("Initializing Quillan BPE Tokenizer...")
    tok = QuillanBPETokenizer()
    eos_id = 50256  # <|endoftext|>

    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    LOGGER.info("Reading canonical samples from %s (max_seq_len=%d)...", jsonl_path.name, max_seq_len)
    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            # Prompt + Reasoning + Reply
            prompt_header = f"<|start|>\n<|user|>\n{data.get('prompt', '').strip()}\n<|assistant|>\n"
            completion_body = f"<think>\n{data.get('thought', '').strip()}\n</think>\n{data.get('response', '').strip()}\n<|end|>"

            prompt_tokens = tok.encode(prompt_header)
            completion_tokens = tok.encode(completion_body)

            full_tokens = prompt_tokens + completion_tokens + [eos_id]

            # Truncate or pad to max_seq_len
            if len(full_tokens) > max_seq_len:
                full_tokens = full_tokens[:max_seq_len]

            pad_len = max_seq_len - len(full_tokens)
            padded_input_ids = full_tokens + [eos_id] * pad_len

            # Prompt loss masking: ignore_index = -100 for user prompt
            prompt_mask_len = min(len(prompt_tokens), max_seq_len)
            padded_labels = (
                [-100] * prompt_mask_len +
                full_tokens[prompt_mask_len:] +
                [-100] * pad_len
            )

            input_ids_list.append(padded_input_ids)
            labels_list.append(padded_labels)

            if (idx + 1) % 10000 == 0:
                LOGGER.info("Tokenized %d samples...", idx + 1)

    total_samples = len(input_ids_list)
    LOGGER.info("Tokenization complete! Total valid samples: %d", total_samples)

    LOGGER.info("Converting to PyTorch long tensors...")
    input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    output_pt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "num_samples": total_samples,
        "max_seq_len": max_seq_len,
        "format": "canonical_thinking_master",
    }

    LOGGER.info("Saving packed tensor dataset to %s...", output_pt_path)
    torch.save(payload, output_pt_path)

    file_size_mb = output_pt_path.stat().st_size / (1024 ** 2)
    LOGGER.info("Packed dataset successfully saved: %.2f MB across %d samples.", file_size_mb, total_samples)
    return output_pt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan Dataset Tensor Packer")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--seq-len", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    in_jsonl = CANONICAL_DIR / "quillan_unified_thinking_master.jsonl"
    out_pt = CANONICAL_DIR / "quillan_unified_thinking_master.pt"

    if in_jsonl.exists():
        pack_dataset(in_jsonl, out_pt, max_seq_len=args.seq_len, max_samples=args.max_samples)
    else:
        LOGGER.error("Source JSONL not found: %s. Run quillan_dataset_normalizer.py first.", in_jsonl)
