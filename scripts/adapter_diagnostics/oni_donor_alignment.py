#!/usr/bin/env python3
"""Step 1 of 4->2->1 knowledge transfer into Quillan's OWN ONI body.

Compute donor -> ONI alignment matrices from shared token rows.
Owner-locked 2026-11-26: vocab 151648, context 2097152.
No donor backbone, no hifi/lofi checkpoint anywhere in this file.
"""

from __future__ import annotations
import gc, sys
from pathlib import Path
import torch

ROOT = Path(r"F:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "03 - Training & Model" / "scripts"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)

DONORS = {
    "qwen": r"F:\quillan_cache\donors\qwen.safetensors",
    "llama": r"F:\quillan_cache\donors\llama.safetensors",
    "bitnet": r"F:\quillan_cache\donors\bitnet.safetensors",
    "falcon": r"F:\quillan_cache\donors\falcon.safetensors",
}
TOKEN_MAPS = r"F:\02_QUILLAN\exports\quillan_token_maps_q2d.pt"
TARGET_EMB = r"F:\02_QUILLAN\exports\quillan_embedding_compiled.safetensors"
OUT = r"F:\02_QUILLAN\exports\oni_donor_alignment.pt"
VOCAB = 151648  # owner-locked
TARGET_DIM = 1152  # mini_6l ONI hidden; main uses slice/pad from this table
MAX_SHARED = 30000  # cap rows per donor for RAM
CHUNK = 8192  # rows read per safetensors slice


def first_key(path: str) -> str:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return next(iter(f.keys()))


def read_rows(path: str, rows: torch.Tensor) -> torch.Tensor:
    """Gather scattered rows without loading the full tensor into RAM."""
    from safetensors import safe_open

    key = first_key(path)
    out = torch.empty((rows.numel(), 0))
    with safe_open(path, framework="pt", device="cpu") as f:
        sl = f.get_slice(key)
        nrows, dim = sl.get_shape()
        out = torch.empty((rows.numel(), dim), dtype=torch.float32)
        valid = (rows >= 0) & (rows < nrows)
        safe_rows = torch.where(valid, rows, 0)
        order, inverse = torch.sort(safe_rows)
        for start in range(0, order.numel(), CHUNK):
            idx = order[start : start + CHUNK]
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            block = sl[lo:hi]  # (hi-lo, dim)
            local = idx - lo
            vals = block[local]  # (chunk, dim)
            pos = torch.arange(start, start + idx.numel())
            out[inverse[pos]] = vals.float()
        out[~valid] = 0.0
    return out


def main():
    from safetensors import safe_open

    # ONI target embedding: ONLY the locked 151648 addressable rows.
    with safe_open(TARGET_EMB, framework="pt", device="cpu") as f:
        sl = f.get_slice("model.embed_tokens.weight")
        E_full = sl[0:VOCAB].float()  # (151648, 2048)
    E_t = E_full[:, :TARGET_DIM].contiguous()  # deterministic projection to ONI hidden
    del E_full
    print(f"ONI target embedding (locked rows) {tuple(E_t.shape)}", flush=True)

    q2d = torch.load(TOKEN_MAPS, weights_only=True, map_location="cpu")
    maps = {}
    for name, path in DONORS.items():
        dm = q2d[name].long() if isinstance(q2d, dict) else q2d[name].long()
        dm = dm[:VOCAB]
        slots = torch.arange(VOCAB)
        did = torch.where((dm >= 0), dm, -1)
        ok = (did >= 0).nonzero().flatten()
        if ok.numel() > MAX_SHARED:
            ok = ok[torch.linspace(0, ok.numel() - 1, MAX_SHARED).long()]
        X = read_rows(path, did[ok])  # donor rows, gathered in chunks
        Y = E_t[ok]
        G = X.T @ X + 1e-2 * torch.eye(X.shape[1]) * X.abs().mean()
        P = torch.linalg.solve(G, X.T @ Y)  # donor_dim -> 1152
        cos = torch.nn.functional.cosine_similarity(X @ P, Y, dim=1).mean().item()
        maps[name] = {"P": P, "shared": int(ok.numel()), "cos": cos}
        print(
            f"  {name:7s} shared={ok.numel():6d} cos={cos:.4f} P={tuple(P.shape)}",
            flush=True,
        )
        # incremental save: no heavy donor can lose earlier results
        torch.save(
            {
                "target_dim": TARGET_DIM,
                "vocab": VOCAB,
                "target_embedding": E_t,
                "maps": maps,
                "note": "donor->ONI alignment on locked rows; step 1 of 4->2->1",
            },
            OUT,
        )
        del X, Y, G, P
        gc.collect()
    print(f"SAVED {OUT}", flush=True)


if __name__ == "__main__":
    main()
