#!/usr/bin/env python3
"""Port the clean ffn=512 4-donor bank directly into council packed buffers (no pooling)."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import (
    quantize_group_ternary_ls,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)
from safetensors.torch import safe_open, save_file

BANK = (
    ROOT / "exports" / "donor_experts_ls.safetensors"
)  # [34,896,512]/[34,512,896] F16
SRC = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
OUT = ROOT / "exports" / "quillan-mini-council-ls.safetensors"
GROUP = 32
LAYERS = (11, 22)


def pack(w):  # w [E,out,in]
    qs, gs = [], []
    for e in range(w.shape[0]):
        t, g = quantize_group_ternary_ls(w[e], group_size=GROUP)
        p, _ = pack_ternary_2bit(t)
        qs.append(p)
        gs.append(g.to(torch.float16))
    return torch.stack(qs).contiguous(), torch.stack(gs).contiguous()


def main():
    with safe_open(str(BANK), framework="pt", device="cpu") as f:
        # bank orientation matches council: w1 [E,896,512]? check
        b1 = f.get_tensor("moe.w1").float()
        bg = f.get_tensor("moe.wgate").float()
        b2 = f.get_tensor("moe.w2").float()
    print("bank shapes", tuple(b1.shape), tuple(bg.shape), tuple(b2.shape), flush=True)
    # council expects w1_q gamma [E,512,28] -> w1 logical [E,512,896]; bank w1 [E,896,512] -> transpose
    w1 = b1.transpose(1, 2)
    wg = bg.transpose(1, 2)
    w2 = b2.transpose(1, 2)  # -> [E,896,512] for gamma [E,896,16]
    patches = {}
    for name, w in [("w1", w1), ("wgate", wg), ("w2", w2)]:
        q, g = pack(w)
        patches[name] = (q, g)
        print(
            f" {name}: q={tuple(q.shape)} gamma={tuple(g.shape)} absm={float(w.abs().mean()):.4f}",
            flush=True,
        )
    with safe_open(str(SRC), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = dict(f.metadata() or {})
        tensors = {}
        for k in keys:
            t = f.get_tensor(k)
            for li in LAYERS:
                pre = f"layers.{li}.council_moe."
                for name, (q, g) in patches.items():
                    if k == pre + name + "_q":
                        assert tuple(t.shape) == tuple(q.shape), (
                            f"{k} {tuple(t.shape)} vs {tuple(q.shape)}"
                        )
                        t = q.clone()
                    elif k == pre + name + "_gamma":
                        assert tuple(t.shape) == tuple(g.shape), (
                            f"{k} {tuple(t.shape)} vs {tuple(g.shape)}"
                        )
                        t = g.clone()
            tensors[k] = t.contiguous()
    save_file(tensors, str(OUT), metadata=meta)
    print(f"[done] {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)", flush=True)
    # verify unpacked expert weights non-trivial
    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        q = f.get_tensor("layers.11.council_moe.w1_q")
        g = f.get_tensor("layers.11.council_moe.w1_gamma")
    t = unpack_ternary_2bit(q[0], (512, 28, 32), 512 * 896)
    w = t * g[0].float().unsqueeze(-1)
    print(
        f"VERIFY unpacked w1: absm={float(w.abs().mean()):.4f} max={float(w.abs().max()):.3f} nonzero={(w != 0).sum().item()}/{w.numel()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

