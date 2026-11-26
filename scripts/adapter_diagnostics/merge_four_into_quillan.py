#!/usr/bin/env python3
"""CODE THE TENSORS OUT: all four donors into ONE Quillan base.

Mechanism (WSP paper Sec 2/4/5 + board #1139), closed-form, no training:
  For every layer and every projection, each donor's weight is aligned into the
  host (Llama/flagship) frame by similarity transform, the delta vs the host base
  is t-SVD'd to rank 16, and the low-rank factors are written into a DISJOINT
  rank-16 block of the host's rank-64 LoRA A/B slots:
      effective W += B @ A = sum over donors of (donor delta)
  Donor blocks: qwen=0..15, llama=16..31, falcon=32..47, bitnet=48..63.
  ALL FOUR contribute real weight tensors. No averaging, no router, no gate.

Alignment: P maps donor hidden -> host hidden (row convention).
  W_aligned = P.T @ W @ pinv(P).T   (hidden dims)
  ffn dims (4864/8192) centre-cropped/padded (no token correspondence).
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model
from tokenizers import Tokenizer
from safetensors import safe_open

torch.set_num_threads(2)
BASE = OUT / "quillan-flagship-hifi-g4-exact.safetensors"
LLAMA_W = r"E:\source_models\llama_1b\model.safetensors"
DONORS = {
    "qwen": (
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors",
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json",
        896,
        4864,
    ),
    "llama": (LLAMA_W, r"E:\source_models\llama_1b\tokenizer.json", 2048, 8192),
    "falcon": (
        r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model_bf16.safetensors",
        r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json",
        2048,
        8192,
    ),
    "bitnet": (
        r"E:\source_models\bitnet_2b_4t\model.safetensors",
        r"E:\source_models\bitnet_2b_4t\tokenizer.json",
        2560,
        6912,
    ),
}
ORDER = ["qwen", "llama", "falcon", "bitnet"]
SUBS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BLOCK = 16


def pinv_row(P):
    # P [d_d, d_h], full row rank; pinv = P.T @ inv(P P.T)
    return P.T @ torch.linalg.inv(P @ P.T)


def crop_pad(t, dim, target):
    cur = t.shape[dim]
    if cur == target:
        return t
    sl = [slice(None)] * t.ndim
    if cur > target:
        st = (cur - target) // 2
        sl[dim] = slice(st, st + target)
        return t[tuple(sl)]
    pad = [0] * (t.ndim * 2)
    st = (target - cur) // 2
    pad[(t.ndim - 1 - dim) * 2 + 1] = target - cur - st
    return torch.nn.functional.pad(t, pad)


def align_weight(W, P, in_is_ffn, out_is_ffn, host_ffn=8192):
    """W [out_d, in_d] -> [out_h, in_h]. Similarity on hidden axes, crop/pad on ffn."""
    Pinv = pinv_row(P)  # [d_h, d_d]
    # hidden transform: out axis uses P (donor->host); in axis uses pinv(P).T
    Wt = W.float()
    # output side
    if not out_is_ffn:
        Wt = P.T @ Wt  # [d_h, in_d]
    else:
        Wt = crop_pad(Wt, 0, host_ffn)
    # input side
    if not in_is_ffn:
        Wt = Wt @ Pinv.T  # [..., d_h]
    else:
        Wt = crop_pad(Wt, 1, host_ffn)
    return Wt


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank-block", type=int, default=16)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    host = load_model(BASE)
    L = len(host.layers)
    with safe_open(LLAMA_W, framework="pt", device="cpu") as f:
        hostE = f.get_tensor("model.embed_tokens.weight").float()
    toks = {n: Tokenizer.from_file(t).get_vocab() for n, (_, t, _, _) in DONORS.items()}
    shared = sorted(set.intersection(*[set(v) for v in toks.values()]))
    print(
        f"shared={len(shared)} layers={L} block={a.rank_block} strength={a.strength} dry={a.dry}",
        flush=True,
    )
    hi = torch.tensor([toks["llama"][s] for s in shared])
    P = {}
    for n, (wp, tp, dh, df) in DONORS.items():
        with safe_open(wp, framework="pt", device="cpu") as f:
            dE = f.get_tensor("model.embed_tokens.weight").float()
        di = torch.tensor([toks[n][s] for s in shared])
        G = dE[di].T @ dE[di] + 1e-2 * torch.eye(dE.shape[1]) * dE.abs().mean()
        P[n] = torch.linalg.solve(G, dE[di].T @ hostE[hi])
        print(f"  P[{n}]={tuple(P[n].shape)}", flush=True)

    # collect host base weights + target shapes
    hostW = {}
    for li in range(L):
        for sub in SUBS:
            mod = getattr(host.layers[li], sub, None)
            if mod is None or not hasattr(mod, "unpack_effective_weight"):
                continue
            hostW[(li, sub)] = mod.unpack_effective_weight().float()

    if a.dry:
        print("\n=== DRY RUN: shape check for every donor/projection ===", flush=True)
        ok = 0
        bad = 0
        for n, (wp, tp, dh, df) in DONORS.items():
            dl_scale = (L - 1) / 14  # donor layers -> host layers roughly
            with safe_open(wp, framework="pt", device="cpu") as f:
                dkeys = set(f.keys())
                for li in range(L):
                    dl = min(
                        round(
                            li
                            * (
                                15
                                if n == "llama"
                                else (
                                    23 if n == "qwen" else (17 if n == "falcon" else 29)
                                )
                            )
                            / max(L - 1, 1)
                        ),
                        (
                            15
                            if n == "llama"
                            else (23 if n == "qwen" else (17 if n == "falcon" else 29))
                        ),
                    )
                    for sub in SUBS:
                        dk = f"model.layers.{dl}.{('self_attn' if sub in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp')}.{sub}.weight"
                        if dk not in dkeys or (li, sub) not in hostW:
                            continue
                        W = f.get_tensor(dk)
                        try:
                            Wa = align_weight(
                                W,
                                P[n],
                                sub in ("gate_proj", "up_proj", "down_proj"),
                                sub in ("gate_proj", "up_proj", "down_proj"),
                            )
                            tgt = hostW[(li, sub)].shape
                            match = tuple(Wa.shape) == tuple(tgt)
                            if match:
                                ok += 1
                            else:
                                bad += 1
                                if bad <= 12:
                                    print(
                                        f"  MISMATCH {n} L{li}/{dl} {sub}: {tuple(Wa.shape)} vs {tuple(tgt)}",
                                        flush=True,
                                    )
                        except Exception as e:
                            bad += 1
                            if bad <= 12:
                                print(
                                    f"  ERR {n} L{li}/{dl} {sub}: {str(e)[:80]}",
                                    flush=True,
                                )
        print(f"DRY RESULT: aligned_exact={ok} mismatched={bad}", flush=True)
        return

    print("\n=== WRITE all four donors into rank-64 LoRA slots ===", flush=True)
    # accumulator per (layer,sub) delta
    accum = {k: torch.zeros_like(v) for k, v in hostW.items()}
    counts = {k: 0 for k in hostW}
    for bi, n in enumerate(ORDER):
        wp, tp, dh, df = DONORS[n]
        nlay = (
            15 if n == "llama" else 23 if n == "qwen" else 17 if n == "falcon" else 29
        )
        print(f"  donor {n} -> block {bi * BLOCK}:{(bi + 1) * BLOCK}", flush=True)
        with safe_open(wp, framework="pt", device="cpu") as f:
            dkeys = set(f.keys())
            for li in range(L):
                dl = round(li * nlay / max(L - 1, 1))
                for sub in SUBS:
                    dk = f"model.layers.{dl}.{('self_attn' if sub in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp')}.{sub}.weight"
                    k = (li, sub)
                    if dk not in dkeys or k not in hostW:
                        continue
                    try:
                        Wa = align_weight(
                            f.get_tensor(dk),
                            P[n],
                            sub in ("gate_proj", "up_proj", "down_proj"),
                            sub in ("gate_proj", "up_proj", "down_proj"),
                        )
                    except Exception:
                        continue
                    if tuple(Wa.shape) != tuple(hostW[k].shape):
                        continue
                    accum[k] += Wa - hostW[k]
                    counts[k] += 1
        print(
            f"    accumulated {sum(1 for v in counts.values() if v > 0)} matrices",
            flush=True,
        )

    written = 0
    for (li, sub), Delta in accum.items():
        if counts[(li, sub)] == 0:
            continue
        mod = getattr(host.layers[li], sub, None)
        Delta = Delta * strength
        r = min(a.rank_block, mod.lora_A.shape[0], *Delta.shape)
        U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)
        A = (Vh[:r]).contiguous()
        Bv = (U[:, :r] * S[:r]).contiguous()
        with torch.no_grad():
            mod.lora_A.zero_()
            mod.lora_B.zero_()
            mod.lora_A[:r].copy_(A.to(mod.lora_A.dtype))
            mod.lora_B[:, :r].copy_(Bv.to(mod.lora_B.dtype))
            if hasattr(mod, "_wc"):
                mod._wc = None
        written += 1
    print(f"  wrote {written} matrices with all-4 deltas", flush=True)
    host.save_packed_safetensors(OUT / a.out)
    print(f"SAVED {OUT / a.out}", flush=True)


if __name__ == "__main__":
    main()
