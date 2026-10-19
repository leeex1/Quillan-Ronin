#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 QUILLAN-RONIN v5.4.0 ONI - DATASET PREPARATION & TOKEN PACKING
================================================================================
Ingests markdown and text papers from Formal Papers / Knowledge Base / Full Corpus,
tokenizes them via UnifiedQuillanTokenizer (50,257 vocab), and serializes
zero-copy memory-mapped binary datasets (train_ids, train_labels, val_ids, val_labels).

Output files:
  - {output_dir}/train_ids.bin     (dtype uint16)
  - {output_dir}/train_labels.bin  (dtype int32)
  - {output_dir}/val_ids.bin       (dtype uint16)
  - {output_dir}/val_labels.bin    (dtype int32)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure local oni package imports correctly
REPO_ROOT = Path(__file__).resolve().parent.parent
ONI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ONI_DIR))

try:
    from quillan_tokenizer_unified import UnifiedQuillanTokenizer
except ImportError:
    # Fallback path if run from root
    sys.path.insert(0, str(REPO_ROOT / "oni"))
    from quillan_tokenizer_unified import UnifiedQuillanTokenizer


def discover_all_corpus_files(repo_root: Path) -> List[Path]:
    """Discovers markdown and text files across all 11 knowledge base domains."""
    subdirs = [
        "Formal Papers",
        "01_Knowledge_Base/Formal Papers",
        "01_Knowledge_Base/Book Series",
        "01_Knowledge_Base",
        "Book Series",
        "Quillan Knowledge files",
        "system prompts",
        "Software Engineer",
        "Audio Engineer",
        "Skills",
        "03_Skills",
        "00 - Meta",
        "docs",
    ]
    seen = set()
    files = []
    for s in subdirs:
        d = repo_root / s
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file() and f.suffix.lower() in [".md", ".txt", ".json"]:
                    if f.name.endswith(".bin") or ".git" in f.parts or "node_modules" in f.parts:
                        continue
                    try:
                        resolved = f.resolve()
                        if resolved not in seen:
                            seen.add(resolved)
                            files.append(f)
                    except Exception:
                        pass
    return files


def discover_input_dir(candidate: str | None = None) -> Path:
    """Finds the formal papers / knowledge documents directory."""
    if candidate and Path(candidate).is_dir():
        return Path(candidate).resolve()

    defaults = [
        REPO_ROOT / "01_Knowledge_Base" / "Formal Papers",
        REPO_ROOT / "Formal Papers",
        Path("/content/Quillan-Ronin/Formal Papers"),
        Path("/content/Quillan-Ronin/01_Knowledge_Base/Formal Papers"),
        Path("Formal Papers"),
        Path("01_Knowledge_Base/Formal Papers"),
    ]
    for d in defaults:
        if d.is_dir():
            return d.resolve()

    return REPO_ROOT


def extract_samples_from_file(file_path: Path, min_chars: int = 40) -> List[str]:
    """Reads a markdown/text file and segments it into cohesive text chunks."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[WARN] Failed to read {file_path.name}: {e}")
        return []

    # Clean Windows CRLF
    text = text.replace("\r\n", "\n")

    # Split into sections / paragraphs
    paragraphs = text.split("\n\n")
    samples = []
    current_chunk = []
    current_len = 0

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean or len(p_clean) < min_chars:
            continue
        
        # If adding this paragraph exceeds ~1200 chars, flush chunk
        if current_len + len(p_clean) > 1200 and current_chunk:
            samples.append("\n\n".join(current_chunk))
            current_chunk = [p_clean]
            current_len = len(p_clean)
        else:
            current_chunk.append(p_clean)
            current_len += len(p_clean)

    if current_chunk:
        samples.append("\n\n".join(current_chunk))

    return samples


def pack_corpus(
    input_dir: Path | None,
    output_dir: Path,
    seq_len: int = 512,
    val_ratio: float = 0.02,
    full_corpus: bool = False,
    vocab_size: int = 50257,
) -> Tuple[int, int, int]:
    """Tokenizes all text files and packs them into uint16 input / int32 label arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = UnifiedQuillanTokenizer()
    print("=" * 80)
    print(f"[QUILLAN PACKER] Initializing Corpus Packing (Vocab: {tokenizer.vocab_size:,})")
    print(f"  - Target Sequence Length: {seq_len}")
    print(f"  - Validation Ratio:       {val_ratio * 100:.1f}%")
    print(f"  - Output Directory:       {output_dir}")
    print("=" * 80)

    if full_corpus:
        md_files = discover_all_corpus_files(REPO_ROOT)
        print(f"[SCAN] Full-corpus discovery found {len(md_files):,} documents across 11 domains.")
    else:
        if input_dir is None:
            input_dir = discover_input_dir()
        md_files = sorted(
            [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in [".md", ".txt", ".json"]]
        )
        print(f"[SCAN] Found {len(md_files):,} candidate documents in {input_dir}.")

    if not md_files:
        raise ValueError("No documents found to pack.")

    all_samples = []
    for f in md_files:
        samples = extract_samples_from_file(f)
        all_samples.extend(samples)

    print(f"[EXTRACT] Extracted {len(all_samples):,} coherent text chunks.")
    print("[TOKENIZE] Encoding samples via UnifiedQuillanTokenizer...")
    
    t0 = time.perf_counter()
    all_tokens = []
    eos_id = getattr(tokenizer, "eos_token_id", getattr(tokenizer, "eot_token", 50256))
    for s in all_samples:
        tokens = tokenizer.encode(s)
        if tokens:
            all_tokens.extend(tokens)
            if eos_id is not None:
                all_tokens.append(eos_id)

    t_tok = time.perf_counter() - t0
    total_tokens = len(all_tokens)
    print(f"[TOKENIZE] Generated {total_tokens:,} total tokens in {t_tok:.2f}s ({total_tokens / max(1e-5, t_tok):,.0f} tok/s).")

    if total_tokens < seq_len:
        raise ValueError(f"Total tokens ({total_tokens}) is less than sequence length ({seq_len}).")

    # Sequence packing
    num_full_seqs = total_tokens // seq_len
    truncated_tokens = num_full_seqs * seq_len
    packed_tokens = all_tokens[:truncated_tokens]

    token_arr = np.array(packed_tokens, dtype=np.uint16)
    token_arr = np.clip(token_arr, 0, vocab_size - 1)
    seqs = token_arr.reshape(num_full_seqs, seq_len)
    
    # Shuffle sequences deterministically
    np.random.seed(42)
    indices = np.random.permutation(num_full_seqs)
    seqs = seqs[indices]

    n_val = max(1, int(num_full_seqs * val_ratio))
    val_seqs = seqs[:n_val]
    train_seqs = seqs[n_val:]

    train_arr = train_seqs.flatten()
    val_arr = val_seqs.flatten()

    train_ids_path = output_dir / "train_ids.bin"
    train_labels_path = output_dir / "train_labels.bin"
    val_ids_path = output_dir / "val_ids.bin"
    val_labels_path = output_dir / "val_labels.bin"

    with open(train_ids_path, "wb") as f:
        f.write(train_arr.tobytes())
    with open(train_labels_path, "wb") as f:
        f.write(train_arr.astype(np.int32).tobytes())

    with open(val_ids_path, "wb") as f:
        f.write(val_arr.tobytes())
    with open(val_labels_path, "wb") as f:
        f.write(val_arr.astype(np.int32).tobytes())

    n_train_seq = len(train_seqs)
    n_val_seq = len(val_seqs)

    print("[SUCCESS] Dataset successfully packed:")
    print(f"  - Train: {len(train_arr):,} tokens ({n_train_seq} sequences of {seq_len})")
    print(f"  - Val:   {len(val_arr):,} tokens ({n_val_seq} sequences of {seq_len})")
    print(f"  - Files: {train_ids_path.name}, {train_labels_path.name}, {val_ids_path.name}, {val_labels_path.name}")
    print("=" * 80)

    return total_tokens, n_train_seq, n_val_seq


def main():
    parser = argparse.ArgumentParser(
        description="Quillan-Ronin v5.4.0 ONI - Formal Papers & Full Corpus Dataset Preparation"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to directory containing .md / .txt papers (default: auto-detect 01_Knowledge_Base/Formal Papers)",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Pack all 11 knowledge domains across the repository (20.34M tokens)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ONI_DIR / "data"),
        help="Path to output directory for binary memmap datasets (default: oni/data)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for token packaging (default: 512)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
        help="Validation split ratio (default: 0.02 / 2%%)",
    )
    args = parser.parse_args()

    input_path = discover_input_dir(args.input_dir) if not args.full_corpus else None
    output_path = Path(args.output_dir).resolve()
    pack_corpus(
        input_path,
        output_path,
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
        full_corpus=args.full_corpus,
    )


if __name__ == "__main__":
    main()
