#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI — 343MB CORPUS PRE-TOKENIZER
=====================================================
Processes the 343MB `tokenizer_sample.jsonl` (research papers, code, architecture)
into 256-token continuous causal pre-training chunks for Stage 1 CPT.
"""
import sys
import json
import time
from pathlib import Path
import torch

REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"
PROJECTS_DIR = REPO_ROOT / "09 - Projects" / "projects"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for p in [str(REPO_ROOT), str(MODEL_DIR), str(PROJECTS_DIR / "oni"), str(SCRIPTS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_bpe_tokenizer import QuillanBPETokenizer

def build_corpus_pt(
    jsonl_path: Path,
    out_pt_path: Path,
    seq_len: int = 256,
    stride: int = 128,
    max_tokens: int = 25_000_000,
):
    print(f"[*] Initializing Quillan BPE Tokenizer...")
    tok = QuillanBPETokenizer()

    print(f"[*] Reading 343MB Corpus: {jsonl_path}...", flush=True)
    chunk_list = []
    start_time = time.time()
    total_text_chars = 0

    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if not text or len(text) < 100:
                    continue
                total_text_chars += len(text)
                
                # Tokenize document directly
                doc_tokens = tok.encode(text)
                doc_tokens.append(50256) # EOS

                # Direct zero-copy strided chunking
                if len(doc_tokens) >= seq_len:
                    for start in range(0, len(doc_tokens) - seq_len + 1, stride):
                        chunk_list.append(doc_tokens[start : start + seq_len])
                        if len(chunk_list) * seq_len >= max_tokens:
                            break

                if line_idx % 25 == 0:
                    elapsed = max(1.0, time.time() - start_time)
                    tok_count = len(chunk_list) * seq_len
                    print(f"    Doc {line_idx:,} | Chunks: {len(chunk_list):,} | Tokens: {tok_count:,} ({tok_count/elapsed:.0f} tok/s)", flush=True)

                if len(chunk_list) * seq_len >= max_tokens:
                    print(f"[*] Target token cap ({max_tokens:,}) reached.", flush=True)
                    break
            except Exception as e:
                pass

    total_tokens = len(chunk_list) * seq_len
    print(f"\n[+] Total Chunks Assembled: {len(chunk_list):,}")
    print(f"[+] Total Real Tokens:     {total_tokens:,} tokens (~{total_tokens/1e6:.1f}M)")

    print(f"[*] Packing into torch tensor...")
    tensor_chunks = torch.tensor(chunk_list, dtype=torch.int32)
    
    out_dict = {
        "input_ids": tensor_chunks,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
        "timestamp": time.time(),
        "source": "tokenizer_sample.jsonl (343MB Stage 1 Pre-Training Corpus)",
    }

    print(f"[*] Saving to {out_pt_path}...")
    torch.save(out_dict, out_pt_path)
    print(f"[SUCCESS] Stage 1 Continuous Pre-Training dataset ready: {out_pt_path.name} ({out_pt_path.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    src = Path(r"C:\02_QUILLAN\09 - Projects\projects\05_Training\scripts\tokenizer_sample.jsonl")
    dest = Path(r"C:\02_QUILLAN\training_data\quillan_pretrain_corpus_343mb.pt")
    build_corpus_pt(src, dest, seq_len=256, stride=128, max_tokens=10_000_000)
