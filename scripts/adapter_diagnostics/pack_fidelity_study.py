#!/usr/bin/env python3
"""Root-cause fix study: how does 2-bit ternary reconstruction error vary with
group_size and number of residual planes? Current = group32, 3 planes, 14% err.
If smaller groups / more planes get error well below ~5%, higher-fidelity
packing can recover the donor knowledge that conversion destroyed.
Fast: quantizes one real Qwen matrix, no model load.
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
from quillan_sovereign_bitnet import quantize_group_ternary_ls

torch.set_num_threads(1)


def reconstruct(W, group, planes, rht=True):
    import math

    resid = W.float().clone()
    if rht:
        h = torch.tensor([[1.0]])
        for _ in range(5):
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        h = h / math.sqrt(32.0)
        resid = (resid.view(resid.shape[0], -1, 32) @ h).view_as(resid)
    out = torch.zeros_like(resid)
    n_groups = resid.shape[1] // group
    for p in range(planes):
        t, g = quantize_group_ternary_ls(resid, group_size=group)
        # g: (out, n_groups); t: (out, in) -> view as (out, n_groups, group)
        recon = (
            t.view(resid.shape[0], n_groups, group) * g.float().unsqueeze(-1)
        ).view_as(resid)
        out += recon
        resid -= recon
    if rht:
        out = (out.view(out.shape[0], -1, 32) @ h.T).view_as(out)
    return out


def relerr(a, b):
    return (a - b).norm().item() / a.norm().clamp_min(1e-9).item()


def main():
    from transformers import AutoModelForCausalLM

    qp = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    q = AutoModelForCausalLM.from_pretrained(
        qp, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    W = q.model.layers[0].self_attn.q_proj.weight.detach()  # 896x896
    print(f"matrix {tuple(W.shape)} norm={W.norm().item():.3f}", flush=True)
    for group in (128, 64, 32, 16, 8):
        row = []
        for planes in (1, 2, 3, 4, 6):
            W2 = reconstruct(W, group, planes)
            row.append(f"p{planes}={relerr(W, W2):.3f}")
        print(f"group={group:3d}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
