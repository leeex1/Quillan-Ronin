#!/usr/bin/env python3
"""USER'S MECHANISM: token-matched consensus normalization of donor embeddings.

For each token string shared across donors:
  1. align each donor's embedding into the host (Qwen 896) frame via full LS
  2. normalize scale so matched-token vectors are comparable
     (the user's 'cat = 0.556 vs 0.665 -> make them agree' step)
  3. MERGE: Quillan's vector = mean of the aligned+normalized donor vectors

Only shared tokens are merged; host-only tokens keep the Qwen value.
Then write the consensus table into the mini's (tied) embedding and repack.

Closed-form throughout: no backprop, no optimizer, no labels.
"""

from __future__ import annotations
import sys, json
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
from wsp_math_eval import load_model, tok_for

torch.set_num_threads(1)
D = Path(r"C:\02_QUILLAN\exports\donor_cache_v23")
BASE = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"


def load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(p, map_location="cpu")


def full_ls(Ed, Eh, ridge=1e-3):
    A = Ed.T @ Ed
    A += ridge * torch.eye(A.shape[0]) * A.diagonal().mean().clamp_min(1e-8)
    return torch.linalg.solve(A, Ed.T @ Eh)


def get_pairs(host_vocab, donor_vocab, Ed, Eh):
    pairs = []
    hv = {}
    for t, i in host_vocab.items():
        hv.setdefault(t, i)
    seen = set()
    for t, di in donor_vocab.items():
        hi = hv.get(t)
        if hi is not None and di < Ed.shape[0] and hi < Eh.shape[0] and hi not in seen:
            pairs.append((hi, di))
            seen.add(hi)
    if len(pairs) < 5000:
        hs = {k.replace("Ġ", "▁"): v for k, v in host_vocab.items()}
        hs2 = {k.replace("▁", "Ġ"): v for k, v in host_vocab.items()}
        for t, di in donor_vocab.items():
            hi = hs.get(t.replace("Ġ", "▁"), hs2.get(t.replace("▁", "Ġ")))
            if (
                hi is not None
                and di < Ed.shape[0]
                and hi < Eh.shape[0]
                and hi not in seen
            ):
                pairs.append((hi, di))
                seen.add(hi)
    return pairs


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", default="qwen,llama,bitnet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = load_model(BASE)
    Eh = m.embed_tokens.unpack_float().float()  # host = Qwen frame (151936, 896)
    host_tok = tok_for(151936)
    host_vocab = host_tok.get_vocab()
    host_norm = Eh.norm(dim=1).mean().item()

    specs = {
        "qwen": (D / "qwen_0.5b_donors.pt", 151936),
        "llama": (D / "llama_1b_donors.pt", 128256),
        "bitnet": (D / "bitnet_2b4t_donors.pt", 128256),
    }
    names = [x for x in a.donors.split(",") if x in specs]

    # accumulator: sum of aligned+normalized donor vectors per host token, count
    acc = torch.zeros_like(Eh)
    cnt = torch.zeros(Eh.shape[0], dtype=torch.int32)
    diagnostics = {}
    for name in names:
        path, vocab = specs[name]
        d = load(path)
        Ed = d["embed_tokens"].float()
        tok = tok_for(vocab)
        pairs = get_pairs(host_vocab, tok.get_vocab(), Ed, Eh)
        hi = torch.tensor([x[0] for x in pairs])
        di = torch.tensor([x[1] for x in pairs])
        eh, ed = Eh[hi], Ed[di]
        # align into host frame -> matched rows (len(pairs), host_dim)
        if name == "qwen":
            aligned_rows = ed.clone()  # matched donor rows, already host frame
        else:
            P = full_ls(ed, eh)
            aligned_rows = ed @ P  # matched rows aligned to host
        # SCALE NORMALIZE: user's 0.556/0.665 -> agreed value
        donor_norm = aligned_rows.norm(dim=1).mean().clamp_min(1e-8)
        scale = host_norm / donor_norm
        aligned_rows = aligned_rows * scale
        acc[hi] += aligned_rows
        cnt[hi] += 1
        # diagnostic on matched rows
        cos = (
            torch.nn.functional.cosine_similarity(aligned_rows, eh, dim=-1)
            .mean()
            .item()
        )
        diagnostics[name] = {
            "pairs": len(pairs),
            "scale": round(scale.item(), 4),
            "aligned_cos": round(cos, 4),
        }
        print(
            f"[{name}] pairs={len(pairs)} scale={scale.item():.4f} aligned_cos={cos:.4f}",
            flush=True,
        )

    # MERGE: mean of matched donors; host-only tokens keep Qwen
    mask = cnt > 0
    consensus = Eh.clone()
    consensus[mask] = acc[mask] / cnt[mask].unsqueeze(-1).float()
    changed = int(mask.sum())
    print(f"merged tokens={changed}/{Eh.shape[0]} (donors={names})", flush=True)
    print(f"diagnostics={json.dumps(diagnostics)}", flush=True)

    m.embed_tokens.set_weights_from_float(consensus)
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
