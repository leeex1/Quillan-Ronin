#!/usr/bin/env python3
"""Port the complete 4-donor bank (ffn=4864) into the packed council MoE format (ffn=512).
Closed-form: mean-pool ffn 4864->512, ternarize with group_size=32, pack to uint8 buffers.
Writes a NEW checkpoint (never overwrites). Zero backprop.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import quantize_group_ternary_ls, pack_ternary_2bit
from safetensors.torch import safe_open, save_file

BANK = ROOT / "exports" / "donor_experts_aligned.safetensors"
SRC = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
OUT = ROOT / "exports" / "quillan-mini-council4.safetensors"
N_E = 34
SRC_FFN = 4864
DST_FFN = 512
HIDDEN = 896
GROUP = 32


def pool_ffn(w, dim):
    # w [E, a, b]; pool the ffn axis (dim=1 for w1/wgate [E,HIDDEN,FFN], dim=2 for w2 [E,FFN,HIDDEN])
    n = w.shape[dim]
    # reshape to [..., DST_FFN, k] and mean
    k = n // DST_FFN
    w = w.narrow(dim, 0, k * DST_FFN)
    sh = list(w.shape)
    sh[dim] = DST_FFN
    sh.insert(dim + 1, k)
    return w.reshape(sh).mean(dim=dim + 1)


def pack(w):
    # w [E, out, in] float -> (q uint8 [E,packed_len], gamma fp16 [E,out,in//GROUP])
    qs, gs = [], []
    for e in range(w.shape[0]):
        t, g = quantize_group_ternary_ls(w[e], group_size=GROUP)  # 2D [out,in]
        packed, _ = pack_ternary_2bit(t)
        qs.append(packed)
        gs.append(g.to(torch.float16))
    return torch.stack(qs).contiguous(), torch.stack(gs).contiguous()


def main():
    with safe_open(str(BANK), framework="pt", device="cpu") as f:
        b1 = f.get_tensor("moe.w1").float()  # [E,896,4864]
        bg = f.get_tensor("moe.wgate").float()
        b2 = f.get_tensor("moe.w2").float()  # [E,4864,896]
    # council w1/wgate want [E,512,896] (out=ffn,in=hidden); dense w1 is [E,896,4864] -> transpose ffn/hidden then pool ffn
    w1 = pool_ffn(b1.transpose(1, 2), 1)  # [E,4864,896] -> [E,512,896]
    wg = pool_ffn(bg.transpose(1, 2), 1)
    w2 = pool_ffn(b2, 1).transpose(
        1, 2
    )  # [E,4864,896]->[E,512,896]->[E,896,512] (w2 gamma is [E,896,16])
    print(
        "pooled shapes:", tuple(w1.shape), tuple(wg.shape), tuple(w2.shape), flush=True
    )
    # NOTE: w2 in council is [E,FFN,HIDDEN]=[E,512,896] per w2_q logical (896*512). Verify against ckpt below.
    patches = {}
    for name, w in [("w1", w1), ("wgate", wg), ("w2", w2)]:
        q, g = pack(w)
        patches[name] = (q, g)
        print(f"  {name}: q={tuple(q.shape)} gamma={tuple(g.shape)}", flush=True)

    with safe_open(str(SRC), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        meta = dict(f.metadata() or {})
        tensors = {}
        for k in keys:
            t = f.get_tensor(k)
            for li in (11, 22):
                pre = f"layers.{li}.council_moe."
                for name, (q, g) in patches.items():
                    if k == pre + name + "_q":
                        assert tuple(t.shape) == tuple(q.shape), (
                            f"{k}: {tuple(t.shape)} vs {tuple(q.shape)}"
                        )
                        t = q.clone()
                    elif k == pre + name + "_gamma":
                        assert tuple(t.shape) == tuple(g.shape), (
                            f"{k}: {tuple(t.shape)} vs {tuple(g.shape)}"
                        )
                        t = g.clone()
            tensors[k] = t.contiguous()
    save_file(tensors, str(OUT), metadata=meta)
    print(f"[done] {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
