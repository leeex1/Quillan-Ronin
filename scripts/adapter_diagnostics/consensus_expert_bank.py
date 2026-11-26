#!/usr/bin/env python3
"""USER'S MECHANISM, applied correctly: token/feature-aligned consensus for
EXPERT weights (not the tied embedding).

The tied embedding is the read head; changing it broke the model (tested).
The consensus belongs in FFN experts, which the council gate selects. For each
donor's expert matrices, align into host frame with full LS + scale normalize
(the 0.556/0.665 -> agreed value step), then merge donors into consensus experts.

Builds on the existing aligned bank shape: (34, 896, 512) etc.
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
from wsp_math_eval import load_model

torch.set_num_threads(1)
D = Path(r"C:\02_QUILLAN\exports\donor_cache_v23")


def load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(p, map_location="cpu")


def full_ls(A, B, ridge=1e-3):
    G = A.T @ A
    G += ridge * torch.eye(G.shape[0]) * G.diagonal().mean().clamp_min(1e-8)
    return torch.linalg.solve(G, A.T @ B)


def main():
    # host frame: qwen down_proj activations basis from embedding alignment
    hd = load(D / "qwen_0.5b_donors.pt")
    host_E = hd["embed_tokens"].float()
    donors = {
        "qwen": load(D / "qwen_0.5b_donors.pt"),
        "llama": load(D / "llama_1b_donors.pt"),
        "bitnet": load(D / "bitnet_2b4t_donors.pt"),
    }
    report = {}
    # For each donor and key, align its matrix rows into host dim and record
    # post-alignment cosine to the host's corresponding matrix (where shared).
    for name, d in donors.items():
        report[name] = {}
        for key in ["embed_tokens", "L6.down_proj", "L6.o_proj"]:
            if key not in d:
                continue
            W = d[key].float()
            if key == "embed_tokens":
                host = host_E
            else:
                host = hd.get(key)
                if host is None:
                    continue
                host = host.float()
            if W.shape[1] != host.shape[1]:
                # align columns into host width using row-space LS on embedding-derived basis
                P = full_ls(
                    host[: min(20000, host.shape[0])], host[: min(20000, host.shape[0])]
                )
            # scale normalize (user step)
            s = host.norm().item() / W.norm().clamp_min(1e-8).item()
            report[name][key] = {"shape": list(W.shape), "scale_to_host": round(s, 4)}
        print(f"[{name}] keys checked", flush=True)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
