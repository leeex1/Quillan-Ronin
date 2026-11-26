#!/usr/bin/env python3
"""Test the user's hypothesis: fusion failed because donors weren't aligned
properly. Current bank uses ORTHOGONAL Procrustes (cos 0.21-0.37). Test a FULL
least-squares linear alignment P = (Ed'Ed)^-1 Ed'Eh (closed form, no training),
which can rescale/reorient donor space into the host frame.

If align-cos jumps from ~0.2-0.37 to >0.7, the earlier fusion failure was an
alignment-method bug and a proper manual tensor alignment is viable.
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
import v23_common as v23
from wsp_math_eval import tok_for

torch.set_num_threads(1)


def pair_cos(a, b):
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float((a * b).sum(-1).mean())


def full_ls(Ed, Eh, ridge=1e-3):
    # P: (d_donor, d_host) minimizing ||Ed P - Eh||
    Ed, Eh = Ed.float(), Eh.float()
    A = Ed.T @ Ed
    A += ridge * torch.eye(A.shape[0]) * A.diagonal().mean().clamp_min(1e-8)
    return torch.linalg.solve(A, Ed.T @ Eh)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    D = Path(r"C:\02_QUILLAN\exports\donor_cache_v23")
    qwen_tok = tok_for(151936)
    # host embeddings from qwen donor cache (896)
    hd = torch.load(D / "qwen_0.5b_donors.pt", map_location="cpu", weights_only=True)
    Eh = hd["embed_tokens"].float()
    host_vocab = qwen_tok.get_vocab()

    donors = {
        "qwen": (D / "qwen_0.5b_donors.pt", qwen_tok, 151936),
        "llama": (D / "llama_1b_donors.pt", None, 128256),
        "bitnet": (D / "bitnet_2b4t_donors.pt", None, 128256),
    }
    for name, (path, tok, vocab) in donors.items():
        d = torch.load(path, map_location="cpu", weights_only=True)
        Ed = d["embed_tokens"].float()
        if tok is None:
            tok = tok_for(vocab)
        donor_vocab = tok.get_vocab()
        # shared token pairs
        pairs = []
        hv = {}
        for t, i in host_vocab.items():
            hv.setdefault(t, i)
        for t, di in donor_vocab.items():
            hi = hv.get(t)
            if hi is not None and di < Ed.shape[0] and hi < Eh.shape[0]:
                pairs.append((hi, di))
        if len(pairs) < 512:
            # normalization fallback
            hs = {k.replace("Ġ", "▁"): v for k, v in host_vocab.items()}
            for t, di in donor_vocab.items():
                hi = hs.get(t.replace("Ġ", "▁"))
                if hi is not None and di < Ed.shape[0] and hi < Eh.shape[0]:
                    pairs.append((hi, di))
        pairs = pairs[:20000]
        hi = torch.tensor([x[0] for x in pairs])
        di = torch.tensor([x[1] for x in pairs])
        eh, ed = Eh[hi], Ed[di]
        # orthogonal (current method)
        p_orth, _ = v23.compute_shared_vocab_procrustes(
            host_vocab, donor_vocab, Eh, Ed, max_pairs=20000
        )
        cos_orth = pair_cos(ed @ p_orth, eh)
        # full least squares (proposed)
        p_ls = full_ls(ed, eh)
        cos_ls = pair_cos(ed @ p_ls, eh)
        print(
            f"{name:8s} pairs={len(pairs):6d} Ed={tuple(Ed.shape)} "
            f"orth_cos={cos_orth:.4f}  fullLS_cos={cos_ls:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
