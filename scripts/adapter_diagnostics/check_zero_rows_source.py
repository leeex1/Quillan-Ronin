#!/usr/bin/env python3
"""Are the 2324 zero compiled rows zero in the SOURCE donor too?"""

from __future__ import annotations
import sys, json
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

FILES = {
    "qwen": r"E:\source_models\qwen_0.5b\model.safetensors",
    "llama": r"E:\source_models\llama_1b\model.safetensors",
    "bitnet": r"E:\source_models\bitnet_2b_4t\model.safetensors",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model.safetensors",
}


def load_emb(p):
    from safetensors import safe_open

    with safe_open(p, framework="pt", device="cpu") as f:
        for k in ("model.embed_tokens.weight", "embed_tokens.weight"):
            if k in f.keys():
                return f.get_tensor(k).float()
        for k in f.keys():
            t = f.get_tensor(k)
            if t.ndim == 2 and t.shape[0] > 10000:
                return t.float()


def main():
    q2d = torch.load(OUT / "quillan_token_maps_q2d.pt", weights_only=True)
    ck = torch.load(OUT / "quillan_embedding_compiled.pt", weights_only=True)
    E, cov = ck["embedding"], ck["coverage"]
    zq = torch.nonzero(E.norm(dim=-1) == 0).flatten().tolist()
    print(f"zero compiled rows={len(zq)}", flush=True)
    donors = Counter()
    src_norm = []
    Ecache = {}
    for q in zq:
        for n in FILES:
            nid = int(q2d[n][q])
            if nid < 0:
                continue
            donors[n] += 1
            if n not in Ecache:
                Ecache[n] = load_emb(FILES[n])
            src_norm.append(Ecache[n][nid].norm().item())
    print(f"  which donor covers these: {dict(donors)}", flush=True)
    print(
        f"  source row norms: min={min(src_norm):.6f} max={max(src_norm):.6f} mean={sum(src_norm) / len(src_norm):.6f}",
        flush=True,
    )
    print(
        f"  source rows that are EXACTLY zero: {sum(1 for x in src_norm if x == 0)} / {len(src_norm)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
