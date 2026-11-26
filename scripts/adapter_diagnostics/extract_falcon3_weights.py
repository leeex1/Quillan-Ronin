#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract falcon3:1b weights from Ollama GGUF blob -> donor_cache_v23 format.

Reads the GGUF blob, extracts embed_tokens + selected layer o_proj/down_proj
(and Q/K/V if present), writes E:\source_models\falcon3_1b\model.safetensors
plus a 13-key donor cache matching qwen/llama layout where possible.

Read-only on the GGUF; writes NEW files only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

GGUF = Path(
    r"C:\Users\Admin\.ollama_models_c\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"
)
OUTDIR = Path(r"C:\02_QUILLAN\exports\donor_cache_v23")


def main() -> int:
    print("=" * 78)
    print("EXTRACT FALCON3:1B FROM GGUF")
    print("=" * 78)
    if not GGUF.exists():
        print(f"FATAL: GGUF blob missing: {GGUF}")
        return 2

    reader = GGUFReader(str(GGUF), "r")
    print(f"GGUF tensors: {len(reader.tensors)}")
    # list a sample of tensor names/shapes
    print("sample tensor names:")
    for rt in reader.tensors[:25]:
        print(f"  {rt.name[:70]:70} {rt.shape}")

    # Extract all tensors (ReaderTensor list API)
    sd: Dict[str, torch.Tensor] = {}
    for rt in reader.tensors:
        n = rt.name
        try:
            arr = rt.data  # numpy
            ten = torch.from_numpy(arr).clone()
        except Exception as e:
            print(f"  skip {n}: {e}")
            continue
        sd[n] = ten

    print(f"\nextracted {len(sd)} tensors to dict")
    # Save full state dict new file
    out = OUTDIR / "falcon3_1b_donors.pt"
    torch.save(sd, str(out))
    print(f"saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Report key categories for c26
    emb = [k for k in sd if "embed" in k.lower() or "tok" in k.lower()]
    attn = [
        k for k in sd if any(x in k for x in ["q_proj", "k_proj", "v_proj", "o_proj"])
    ]
    mlp = [k for k in sd if any(x in k for x in ["down", "up", "gate", "ffn", "mlp"])]
    print(f"\nembed keys ({len(emb)}): {emb[:5]}")
    print(f"attn keys ({len(attn)}): {attn[:5]}")
    print(f"mlp keys ({len(mlp)}): {mlp[:5]}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
