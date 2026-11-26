#!/usr/bin/env python3
"""Print per-tensor stats for each donor to expose the scale clash and the
naming-scheme mismatch (HF-style vs GGUF-style)."""

from __future__ import annotations
from pathlib import Path
import torch

D = Path(r"C:\02_QUILLAN\exports\donor_cache_v23")


def load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(p, map_location="cpu")


def stats(t):
    t = t.float()
    return f"shape={tuple(t.shape)} norm={t.norm().item():.3f} mean={t.mean().item():+.4f} std={t.std().item():.4f} absmax={t.abs().max().item():.3f}"


def main():
    for f in [
        "qwen_0.5b_donors.pt",
        "llama_1b_donors.pt",
        "bitnet_2b4t_donors.pt",
        "falcon3_1b_donors.pt",
    ]:
        d = load(D / f)
        print(f"\n===== {f} ({len(d)} tensors) =====", flush=True)
        for k in sorted(d.keys()):
            if torch.is_tensor(d[k]):
                print(f"  {k:35s} {stats(d[k])}", flush=True)


if __name__ == "__main__":
    main()
