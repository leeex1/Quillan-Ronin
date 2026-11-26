#!/usr/bin/env python3
"""Replace the random MoE router with a DETERMINISTIC closed-form router.

The bank maps experts 0-7=qwen, 8-15=llama, 16-23=dragon/falcon, 24-31=bitnet,
32-33=consensus. Each donor has a tokenizer; shared token strings let us map a
token to each donor's embedding. We build a router signature per expert group:
  sig[expert] = mean over shared tokens of (host_embed[token] aligned via donor P)
Then router(h) = h @ sig.T routes each hidden state to the donor whose
representation it is closest to. No training, no backprop.

Since the FFN experts live in the HOST frame (bank was full-LS aligned), the
simplest deterministic routing is by the HOST embedding of the token itself:
donor groups become distinguishable by frequency-band / token-class signatures.
We use a principled proxy: K-means over the host embedding table into 34 clusters,
each cluster assigned to a donor group round-robin. Router rows = cluster centers.
This gives deterministic, input-dependent routing (tokens near a center fire its
donor) with no randomness and no training.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model

torch.set_num_threads(2)
SRC = ROOT / "exports" / "quillan-mini-hifi-donorbank.safetensors"
OUT = ROOT / "exports" / "quillan-mini-hifi-detrouter.safetensors"


def kmeans_centers(X, k, iters=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    # sample rows as init
    idx = torch.randperm(X.shape[0], generator=g)[:k]
    C = X[idx].clone().float()
    for _ in range(iters):
        d = torch.cdist(X.float(), C)  # (N,k)
        assign = d.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = X[m].float().mean(0)
    return C


def main():
    m = load_model(SRC)
    E = m.embed_tokens.unpack_float().float()  # (vocab, 896)
    # sample a broad slice for tractable kmeans
    X = E[:50000]
    C = kmeans_centers(X, 34, iters=15, seed=20261124)
    C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-8) * E.norm(dim=1).mean()
    # assign clusters to donor groups (0-7 qwen, 8-15 llama, 16-23 falcon, 24-31 bitnet, 32-33 consensus)
    with torch.no_grad():
        for li in (11, 22):
            m.layers[li].council_moe.router.weight.copy_(
                C.to(m.layers[li].council_moe.router.weight.dtype)
            )
    m.save_packed_safetensors(OUT)
    print(
        f"SAVED {OUT} router centers={tuple(C.shape)} norm={C.norm().item():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
