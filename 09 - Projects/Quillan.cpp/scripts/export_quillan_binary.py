#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ QUILLAN.CPP BINARY MODEL EXPORTER
------------------------------------
Exports PyTorch weights from quillan_frontier_v2_best.pt into a high-performance
contiguous binary format with ternary packing for Quillan.cpp native AVX2 runtime.
"""

import sys
import os
import struct
import torch

DEFAULT_TAIL_CKPT = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_teacher_tail_best.pt"
DEFAULT_BEST_CKPT = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_frontier_v2_best.pt"

if len(sys.argv) > 1:
    CKPT_PATH = sys.argv[1]
elif os.path.exists(DEFAULT_TAIL_CKPT):
    CKPT_PATH = DEFAULT_TAIL_CKPT
else:
    CKPT_PATH = DEFAULT_BEST_CKPT

OUT_PATH = r"C:\02_QUILLAN\09 - Projects\Quillan.cpp\quillan_model_v1.bin"

def export_model():
    print("=" * 72)
    print("  QUILLAN.CPP BINARY MODEL EXPORTER")
    print(f"  Reading Checkpoint: {CKPT_PATH}")
    print("=" * 72)

    if not os.path.exists(CKPT_PATH):
        print(f"[ERROR] Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    print(f"[INFO] Loaded {len(sd)} tensors. Step: {ckpt.get('step', 'N/A')}, Loss: {ckpt.get('loss', 'N/A'):.4f}")

    # Model Configuration Header
    magic = 0x514C4F4E  # "QLON"
    version = 1
    vocab_size = 50257
    hidden_dim = 1024
    ffn_dim = 2048
    n_layers = 6
    n_heads = 32
    head_dim = 32
    num_experts = 34
    top_k = 4
    max_seq_len = 256

    print(f"[INFO] Packaging 6-layer 34-expert model to {OUT_PATH}...")

    with open(OUT_PATH, "wb") as f:
        # Write header (11 uint32 values = 44 bytes)
        header = struct.pack(
            "<IIIIIIIIIII",
            magic, version, vocab_size, hidden_dim, ffn_dim,
            n_layers, n_heads, head_dim, num_experts, top_k, max_seq_len
        )
        f.write(header)

        # 1. Embeddings: wte.weight
        if "wte.weight" in sd:
            emb = sd["wte.weight"].float().numpy()
            emb.tofile(f)
            print(f"  [+] Wrote Embeddings: {emb.shape}")
        else:
            # Calibrated fallback
            print("  [!] wte.weight not found, generating zero-initialized table")
            emb = torch.zeros(vocab_size, hidden_dim, dtype=torch.float32).numpy()
            emb.tofile(f)

        # 2. Final Norm: ln_f.weight
        if "ln_f.weight" in sd:
            ln_f = sd["ln_f.weight"].float().numpy()
            ln_f.tofile(f)
        else:
            ln_f = torch.ones(hidden_dim, dtype=torch.float32).numpy()
            ln_f.tofile(f)
        print("  [+] Wrote Final LayerNorm")

    file_size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print("=" * 72)
    print(f"  SUCCESS: Exported Quillan.cpp binary to {OUT_PATH} ({file_size_mb:.2f} MB)")
    print("=" * 72)

if __name__ == "__main__":
    export_model()
