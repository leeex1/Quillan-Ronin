#!/usr/bin/env python3
"""TIES-style sign-aware merge of Llama (host) + Falcon3 into the flagship.

Naive averaging collapsed (9/40) because conflicting weights cancel. TIES:
  1. delta_i = W_donor_i - W_base_for_donor_i   (task vector)
     Here no shared base exists, so we use a symmetric form: for 2 donors,
     treat each donor's weight as a "task vector" from the other's perspective.
  2. Sign election: keep weights where donors AGREE in sign; for disagreements,
     keep the larger-magnitude one (the stronger trait) rather than averaging.
  3. Trim: drop the lowest-magnitude 20% (noise), renormalize survivors.
This preserves unique traits (strong donor-specific weights) and merges similar
ones (agreeing weights averaged), matching the user's spec.

Closed-form, no backprop, no optimizer.
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
from safetensors import safe_open

torch.set_num_threads(2)
FALCON = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model_bf16.safetensors"
HOST = ROOT / "exports" / "quillan-flagship-hifi-g4-exact.safetensors"  # faithful Llama
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


def ties_merge(a, b, trim=0.2):
    """Sign-aware merge of two equally-shaped matrices."""
    sa = torch.sign(a)
    sb = torch.sign(b)
    maga = a.abs()
    magb = b.abs()
    agree = sa == sb
    # where they agree: average. where disagree: keep the larger magnitude one.
    out = torch.where(agree, (a + b) / 2, torch.where(maga >= magb, a, b))
    # trim lowest-magnitude trim fraction toward zero (noise suppression)
    flat = out.abs().flatten()
    k = int(trim * flat.numel())
    if k > 0:
        thresh = torch.kthvalue(flat, k + 1).values
        out = torch.where(out.abs() < thresh, torch.zeros_like(out), out)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim", type=float, default=0.2)
    ap.add_argument("--layers", type=int, default=16)
    a = ap.parse_args()
    host = load_model(HOST)
    falcon_layers = [round(i * 17 / (a.layers - 1)) for i in range(a.layers)]
    print(f"falcon layer map: {falcon_layers}", flush=True)
    with safe_open(FALCON, framework="pt", device="cpu") as ff:
        n = 0
        for li, layer in enumerate(host.layers):
            fl = falcon_layers[li]
            for sub in ATTN + FFN:
                mm = getattr(layer, sub, None)
                if mm is None or not hasattr(mm, "unpack_effective_weight"):
                    continue
                W_h = mm.unpack_effective_weight().float()
                fk = (
                    f"model.layers.{fl}."
                    + ("self_attn." if sub in ATTN else "mlp.")
                    + sub
                    + ".weight"
                )
                if fk not in ff.keys():
                    continue
                W_f = ff.get_tensor(fk).float()
                if tuple(W_f.shape) != tuple(W_h.shape):
                    if (
                        sub in ("k_proj", "v_proj")
                        and W_f.shape[0] == 1024
                        and W_h.shape[0] == 512
                        and W_f.shape[1] == W_h.shape[1]
                    ):
                        W_f = W_f.view(512, 2, W_f.shape[1]).mean(dim=1)
                    else:
                        continue
                W = ties_merge(W_h, W_f, a.trim)
                mm.set_weights_from_float(W)
                n += 1
            if li % 4 == 0:
                print(f"  merged layer {li}", flush=True)
    out = ROOT / "exports" / a.out
    host.save_packed_safetensors(out)
    print(
        f"SAVED {out} merged={n} trim={a.trim} size={out.stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
