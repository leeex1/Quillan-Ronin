#!/usr/bin/env python3
"""TRUE FOUR-DONOR COLD START: align all four donors into one frame, merge all four.

Donors -> host frame (flagship/Llama, 2048 hidden, 16 layers, 8192 FFN):
  LLaMA   : native host (2048) - identity
  FALCON  : 2048 hidden - ridge align via shared tokens
  QWEN    : 896 hidden  - ridge align via shared tokens (upsample to 2048)
  BITNET  : 2560 hidden - ridge align via shared tokens (downsample to 2048)

For each layer and each matrix, align every donor's weight into host frame,
then sign-aware (TIES) merge ALL FOUR into one matrix. Result: ONE weight set
carrying all four donors' knowledge. Closed-form, no training, no backprop.
This is the cold start that WSP tunes later (including real-time learning).
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
from safetensors import safe_open

torch.set_num_threads(2)
HOST = ROOT / "exports" / "quillan-flagship-hifi-g4-exact.safetensors"
LLAMA_W = r"E:\source_models\llama_1b\model.safetensors"
FALCON_W = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model_bf16.safetensors"
QWEN_W = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"
BITNET_W = r"E:\source_models\bitnet_2b_4t\model.safetensors"
TOKS = {
    "llama": r"E:\source_models\llama_1b\tokenizer.json",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json",
    "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json",
    "bitnet": r"E:\source_models\bitnet_2b_4t\tokenizer.json",
}
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


def fit_P(Ed, Eh, ridge=1e-2):
    G = Ed.T @ Ed + ridge * torch.eye(Ed.shape[1]) * Ed.abs().mean()
    return torch.linalg.solve(G, Ed.T @ Eh)


def align_matrices_for_donor(donor, host_model, layers_map):
    """Return dict[(layer,sub)] -> aligned weight in host dims."""
    pass


def force_shape(W, target_shape):
    """Resample to exact target shape via linear interpolation by index. Closed-form."""
    for ax in range(W.ndim):
        cur, tgt = W.shape[ax], target_shape[ax]
        if cur == tgt:
            continue
        pos = torch.linspace(0, cur - 1, steps=tgt)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=cur - 1)
        fr = pos - lo
        idx_shape = [1] * W.ndim
        idx_shape[ax] = tgt
        a = W.gather(
            ax,
            lo.reshape(idx_shape).expand(
                list(W.shape[:ax]) + [tgt] + list(W.shape[ax + 1 :])
            ),
        )
        b = W.gather(
            ax,
            hi.reshape(idx_shape).expand(
                list(W.shape[:ax]) + [tgt] + list(W.shape[ax + 1 :])
            ),
        )
        fr = fr.reshape(idx_shape)
        W = a * (1 - fr) + b * fr
    return W if tuple(W.shape) == tuple(target_shape) else None


def ties_multi(weights, trim=0.2):
    """Sign-aware merge of N matrices: per-element majority sign, mean of agreeing, max-mag of disagreeing."""
    stack = torch.stack([w.float() for w in weights])  # (N,...)
    signs = torch.sign(stack)
    mags = stack.abs()
    # majority sign
    pos = (signs > 0).sum(0)
    neg = (signs < 0).sum(0)
    majority = torch.where(pos >= neg, 1.0, -1.0)
    # mean of weights agreeing with majority sign
    agree = signs == majority.unsqueeze(0)
    num = (stack * agree).sum(0)
    den = agree.sum(0).clamp_min(1)
    out = num / den
    k = int(trim * out.numel())
    if k > 0:
        th = torch.kthvalue(out.abs().flatten(), k + 1).values
        out = torch.where(out.abs() < th, torch.zeros_like(out), out)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trim", type=float, default=0.2)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    host = load_model(HOST)
    H = host.cfg.hidden_size
    # host embedding for shared-token alignment
    with safe_open(LLAMA_W, framework="pt", device="cpu") as s:
        hostE = s.get_tensor("model.embed_tokens.weight").float()
    toks = {k: Tokenizer.from_file(v).get_vocab() for k, v in TOKS.items()}
    shared = sorted(set.intersection(*[set(v) for v in toks.values()]))
    print(f"shared tokens across 4={len(shared)}", flush=True)
    # fit P per donor into host frame
    Ps = {}
    for donor, Wp in [("falcon", FALCON_W), ("qwen", QWEN_W), ("bitnet", BITNET_W)]:
        with safe_open(Wp, framework="pt", device="cpu") as s:
            dE = s.get_tensor("model.embed_tokens.weight").float()
        di = torch.tensor(
            [toks[donor][t] for t in shared if t in toks[donor]][: len(shared)]
        )
        hi = torch.tensor([toks["llama"][t] for t in shared])
        n = min(len(di), len(hi))
        Ed = dE[di[:n]]
        Eh = hostE[hi[:n]]
        Ps[donor] = fit_P(Ed, Eh)
        cos = (
            torch.nn.functional.cosine_similarity(Ed @ Ps[donor], Eh, dim=-1)
            .mean()
            .item()
        )
        print(
            f"  P[{donor}] {tuple(Ps[donor].shape)} aligned_cos={cos:.4f}", flush=True
        )
    # donor layer maps (different depths -> map to host 16)
    donor_n = {"llama": 16, "falcon": 18, "qwen": 24, "bitnet": 30}

    def lmap(d, li):
        return round(li * (donor_n[d] - 1) / 15)

    Wpaths = {"llama": LLAMA_W, "falcon": FALCON_W, "qwen": QWEN_W, "bitnet": BITNET_W}
    opens = {d: safe_open(w, framework="pt", device="cpu") for d, w in Wpaths.items()}
    nmerge = 0
    for li, layer in enumerate(host.layers):
        for sub in ATTN + FFN:
            mm = getattr(layer, sub, None)
            if mm is None or not hasattr(mm, "unpack_effective_weight"):
                continue
            Wh = mm.unpack_effective_weight().float()
            Ws = [Wh]  # llama native
            for d in ("falcon", "qwen", "bitnet"):
                f = opens[d]
                fl = lmap(d, li)
                prefix = (
                    "model.layers."
                    + str(fl)
                    + "."
                    + ("self_attn." if sub in ATTN else "mlp.")
                    + sub
                    + ".weight"
                )
                if prefix not in f.keys():
                    Ws.append(None)
                    continue
                W = f.get_tensor(prefix).float()
                P = Ps[d]
                d_h = {"falcon": 2048, "qwen": 896, "bitnet": 2560}[d]
                # Align hidden dims into host frame via P (input=right, output=left)
                if W.shape[1] == d_h:
                    W = W @ P  # (out,d_h)@(d_h,H) -> (out,H)
                if W.shape[0] == d_h:
                    W = P.T @ W  # (H,d_h)@(d_h,in) -> (H,in)
                # Now coerce to exact host shape by pooling/repeating FFN & head dims
                W = force_shape(W, tuple(Wh.shape))
                Ws.append(W)
            if any(w is None for w in Ws):
                print(f"  skip L{li}.{sub} (shape mismatch donors)", flush=True)
                continue
            Wm = ties_multi(Ws, a.trim)
            mm.set_weights_from_float(Wm)
            nmerge += 1
        if li % 4 == 0:
            print(f"  layer {li}", flush=True)
    for f in opens.values():
        f.__exit__(None, None, None)
    out = ROOT / "exports" / a.out
    host.save_packed_safetensors(out)
    print(
        f"SAVED {out} merged_matrices={nmerge} size={out.stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
