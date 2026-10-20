#!/usr/bin/env python3
"""
pack_master_frontier_bin.py
Zero-Copy Memory-Mapped Token Packer for Quillan-Ronin v5.4.0-ONI
Prepares the complete 319,097-sample master frontier reasoning dataset
into binary memmap files with prompt masking (-100 on prompts, loss only on reasoning/responses).

Outputs:
  - {output_dir}/train_ids.bin     (dtype uint16)
  - {output_dir}/train_labels.bin  (dtype int32)
  - {output_dir}/val_ids.bin       (dtype uint16)
  - {output_dir}/val_labels.bin    (dtype int32)
  - {output_dir}/dataset_meta.json
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Scripts dir on sys.path
SCRIPTS_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model\scripts")
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_tokenizer_unified import UnifiedQuillanTokenizer

INPUT_JSONL = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\quillan_master_frontier_all_334k.jsonl")
OUTPUT_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\data_bin")

SEQ_LEN = 512
VAL_RATIO = 0.02
EOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1

def main():
    start_time = time.time()
    print("=" * 72)
    print("QUILLAN-RONIN MASTER FRONTIER DATASET PACKER")
    print(f"Input JSONL:  {INPUT_JSONL}")
    print(f"Output Dir:   {OUTPUT_DIR}")
    print(f"Sequence Len: {SEQ_LEN}")
    print("=" * 72, flush=True)

    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = UnifiedQuillanTokenizer()
    print(f"[TOKENIZER] Loaded vocab size: {tokenizer.vocab_size:,}", flush=True)

    all_sequences_ids = []
    all_sequences_labels = []

    total_samples = 0
    total_tokens_generated = 0
    total_loss_tokens = 0

    print("\n[STREAMING] Tokenizing and packing with prompt masking (-100)...", flush=True)

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            messages = data.get("messages", [])
            if not messages and "conversations" in data:
                messages = [
                    {"role": "user" if c.get("from") in ("human", "user") else "assistant", "content": c.get("value", "")}
                    for c in data["conversations"]
                ]

            if not messages:
                continue

            # Build prompt tokens and completion tokens
            prompt_text = ""
            completion_text = ""

            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "")
                if role in ("system", "user"):
                    prompt_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                elif role == "assistant":
                    completion_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

            if not prompt_text or not completion_text:
                continue

            p_ids = tokenizer.encode(prompt_text)
            c_ids = tokenizer.encode(completion_text)

            combined_ids = p_ids + c_ids
            # Prompt mask: -100 for prompt tokens, token_id for completion tokens
            combined_labels = [-100] * len(p_ids) + c_ids

            total_samples += 1
            total_tokens_generated += len(combined_ids)
            total_loss_tokens += len(c_ids)

            # Chunk into SEQ_LEN windows
            for i in range(0, len(combined_ids), SEQ_LEN):
                chunk_ids = combined_ids[i : i + SEQ_LEN]
                chunk_labels = combined_labels[i : i + SEQ_LEN]

                # Pad to SEQ_LEN if shorter
                if len(chunk_ids) < SEQ_LEN:
                    pad_len = SEQ_LEN - len(chunk_ids)
                    chunk_ids = chunk_ids + [PAD_TOKEN_ID] * pad_len
                    chunk_labels = chunk_labels + [-100] * pad_len

                all_sequences_ids.append(np.array(chunk_ids, dtype=np.uint16))
                all_sequences_labels.append(np.array(chunk_labels, dtype=np.int32))

            if total_samples % 50000 == 0:
                elapsed = time.time() - start_time
                print(f"      ... processed {total_samples:,} samples | {len(all_sequences_ids):,} sequences ({elapsed:.1f}s)", flush=True)

    print(f"\n[ASSEMBLY] Total samples: {total_samples:,} | Total sequences of {SEQ_LEN}: {len(all_sequences_ids):,}", flush=True)

    seq_ids_mat = np.array(all_sequences_ids, dtype=np.uint16)
    seq_labels_mat = np.array(all_sequences_labels, dtype=np.int32)

    # Deterministic split
    num_seqs = len(seq_ids_mat)
    np.random.seed(42)
    indices = np.random.permutation(num_seqs)
    seq_ids_mat = seq_ids_mat[indices]
    seq_labels_mat = seq_labels_mat[indices]

    n_val = max(100, int(num_seqs * VAL_RATIO))
    n_train = num_seqs - n_val

    val_ids = seq_ids_mat[:n_val].flatten()
    val_labels = seq_labels_mat[:n_val].flatten()
    train_ids = seq_ids_mat[n_val:].flatten()
    train_labels = seq_labels_mat[n_val:].flatten()

    print(f"[WRITING] Writing binary files to {OUTPUT_DIR}...", flush=True)

    train_ids_path = OUTPUT_DIR / "train_ids.bin"
    train_labels_path = OUTPUT_DIR / "train_labels.bin"
    val_ids_path = OUTPUT_DIR / "val_ids.bin"
    val_labels_path = OUTPUT_DIR / "val_labels.bin"

    with open(train_ids_path, "wb") as f:
        f.write(train_ids.tobytes())
    with open(train_labels_path, "wb") as f:
        f.write(train_labels.tobytes())
    with open(val_ids_path, "wb") as f:
        f.write(val_ids.tobytes())
    with open(val_labels_path, "wb") as f:
        f.write(val_labels.tobytes())

    meta = {
        "dataset_name": "Quillan-Ronin-Master-Frontier-Unified",
        "total_dialogue_samples": total_samples,
        "total_tokens_raw": total_tokens_generated,
        "total_active_loss_tokens": total_loss_tokens,
        "sequence_length": SEQ_LEN,
        "total_sequences": num_seqs,
        "train_sequences": n_train,
        "val_sequences": n_val,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "vocab_size": tokenizer.vocab_size,
        "prompt_masking": "-100 applied to all user/system prompt tokens",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    meta_path = OUTPUT_DIR / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed_total = time.time() - start_time
    train_mb = (train_ids.nbytes + train_labels.nbytes) / (1024 ** 2)
    val_mb = (val_ids.nbytes + val_labels.nbytes) / (1024 ** 2)

    print("\n" + "=" * 72)
    print("BINARY MEMMAP DATASET READY FOR ZERO-OVERHEAD TRAINING")
    print(f"Total Train Sequences: {n_train:,} ({len(train_ids):,} tokens, {train_mb:.2f} MB)")
    print(f"Total Val Sequences:   {n_val:,} ({len(val_ids):,} tokens, {val_mb:.2f} MB)")
    print(f"Metadata File:         {meta_path.name}")
    print(f"Total Packing Time:    {elapsed_total:.1f}s ({elapsed_total / 60:.2f} mins)")
    print("=" * 72, flush=True)

if __name__ == "__main__":
    main()
