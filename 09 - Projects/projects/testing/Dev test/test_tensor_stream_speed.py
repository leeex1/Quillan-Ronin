#!/usr/bin/env python3
"""
Diagnostic test for High-Speed Chunked Tensor Ingestion from 166M-token corpus.
"""
import torch, time
from pathlib import Path

pt_file = Path(r"C:\02_QUILLAN\training_data\quillan_corpus_CLEAN_V7.pt")
print(f"[*] Memory-mapping {pt_file.name}...", flush=True)
t0 = time.time()
corpus = torch.load(pt_file, map_location="cpu", weights_only=False)
print(f"[+] Loaded {corpus.shape[0]:,} tokens in {time.time()-t0:.2f}s!", flush=True)

# Test batch slicing speed
batch_size = 16
seq_len = 128
total_steps = 1000

t_slice = time.time()
for step in range(total_steps):
    # Random offsets across 166M tokens
    idx = torch.randint(0, corpus.shape[0] - seq_len - 1, (batch_size,))
    batch = torch.stack([corpus[i : i + seq_len] for i in idx])

dt = time.time() - t_slice
tokens_processed = batch_size * seq_len * total_steps
print(f"[+] Sliced {tokens_processed:,} tokens ({total_steps} batches) in {dt:.3f}s ({tokens_processed/dt:,.0f} tok/s)!", flush=True)
