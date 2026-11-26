#!/usr/bin/env python3
"""THREE SEPARATE FULL MERGES, each producing its own complete checkpoint:

  MERGE 1: llama + qwen    -> merge_base1.safetensors   (full model)
  MERGE 2: bitnet + falcon3 -> merge_base2.safetensors   (full model)
  MERGE 3: merge_base1 + merge_base2 -> quillan base    (full model)

Each merge writes the paired donors' aligned delta into the Quillan base's
rank-64 LoRA slots and saves a COMPLETE standalone checkpoint. Merge 3 reads
both saved checkpoints, combines their LoRA deltas, and saves the final base.

Closed-form, no training. Flagship (Llama 2048) frame.
"""

from __future__ import annotations
import sys
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
LLAMA = r"F:\quillan_cache\donors\llama.safetensors"
QWEN = r"F:\quillan_cache\donors\qwen.safetensors"
BITNET = r"F:\quillan_cache\donors\bitnet.safetensors"
FALCON = r"F:\quillan_cache\donors\falcon.safetensors"
TOK = {
    "llama": r"E:\source_models\llama_1b\tokenizer.json",
    "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json",
    "bitnet": r"E:\source_models\bitnet_2b_4t\tokenizer.json",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json",
}
W = {"llama": LLAMA, "qwen": QWEN, "bitnet": BITNET, "falcon": FALCON}
NLAY = {"llama": 15, "qwen": 23, "bitnet": 29, "falcon": 17}
SUBS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def pinv_row(P):
    return P.T @ torch.linalg.inv(P @ P.T)


def crop_pad(t, dim, tgt):
    if t.shape[dim] == tgt:
        return t
    cur = t.shape[dim]
    if cur > tgt:
        sl = [slice(None)] * t.ndim
        st = (cur - tgt) // 2
        sl[dim] = slice(st, st + tgt)
        return t[tuple(sl)]
    p = [0] * (2 * t.ndim)
    p[(t.ndim - 1 - dim) * 2 + 1] = tgt - cur
    return torch.nn.functional.pad(t, p)


def align(Wt, P, is_ffn, Pinv=None):
    if Pinv is None:
        Pinv = pinv_row(P)
    Wt = Wt.float()
    if not is_ffn:
        Wt = P.T @ Wt
    else:
        Wt = crop_pad(Wt, 0, 8192)
    if not is_ffn:
        Wt = Wt @ Pinv.T
    else:
        Wt = crop_pad(Wt, 1, 8192)
    return Wt


def build_Ps():
    with safe_open(LLAMA, framework="pt", device="cpu") as f:
        hE = f.get_tensor("model.embed_tokens.weight").float()
    toks = {n: Tokenizer.from_file(t).get_vocab() for n, t in TOK.items()}
    shared = sorted(set.intersection(*[set(v) for v in toks.values()]))
    hi = torch.tensor([toks["llama"][s] for s in shared])
    P = {}
    for n, wp in W.items():
        with safe_open(wp, framework="pt", device="cpu") as f:
            dE = f.get_tensor("model.embed_tokens.weight").float()
        di = torch.tensor([toks[n][s] for s in shared])
        G = dE[di].T @ dE[di] + 1e-2 * torch.eye(dE.shape[1]) * dE.abs().mean()
        P[n] = torch.linalg.solve(G, dE[di].T @ hE[hi])
    return P, len(shared)


def pair_delta(hostW, P, donors):
    accum = {k: torch.zeros_like(v) for k, v in hostW.items()}
    cnt = {k: 0 for k in hostW}
    for dn in donors:
        with safe_open(W[dn], framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for k, hW in hostW.items():
                li, sub = k
                dl = round(li * NLAY[dn] / 15)
                grp = "self_attn" if sub in ATTN else "mlp"
                dk = f"model.layers.{dl}.{grp}.{sub}.weight"
                if dk not in keys:
                    continue
                try:
                    Wa = align(f.get_tensor(dk), P[dn], sub not in ATTN)
                except Exception:
                    continue
                if tuple(Wa.shape) != tuple(hW.shape):
                    continue
                accum[k] += Wa - hW
                cnt[k] += 1
    return {k: accum[k] / max(cnt[k], 1) for k in hostW if cnt[k] > 0}


def write_delta_into_model(m, delta, strength, rank):
    written = 0
    for k, Delta in delta.items():
        mod = getattr(m.layers[k[0]], k[1], None)
        if mod is None:
            continue
        Delta = Delta * strength
        r = min(rank, mod.lora_A.shape[0], *Delta.shape)
        U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)
        A = Vh[:r].contiguous()
        Bv = (U[:, :r] * S[:r]).contiguous()
        with torch.no_grad():
            mod.lora_A.zero_()
            mod.lora_B.zero_()
            mod.lora_A[:r].copy_(A.to(mod.lora_A.dtype))
            mod.lora_B[:, :r].copy_(Bv.to(mod.lora_B.dtype))
            if hasattr(mod, "_wc"):
                mod._wc = None
        written += 1
    return written


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    P, nsh = build_Ps()
    print(
        f"shared={nsh} P shapes=" + ", ".join(f"{n}:{tuple(P[n].shape)}" for n in P),
        flush=True,
    )

    # host base weights from fresh base
    m0 = load_model(BASE)
    hostW = {}
    for li in range(len(m0.layers)):
        for sub in SUBS:
            mod = getattr(m0.layers[li], sub, None)
            if mod is not None and hasattr(mod, "unpack_effective_weight"):
                hostW[(li, sub)] = mod.unpack_effective_weight().float()
    del m0

    # MERGE 1
    print("\n=== MERGE 1: llama + qwen -> merge_base1 ===", flush=True)
    d1 = pair_delta(hostW, P, ["llama", "qwen"])
    m = load_model(BASE)
    n = write_delta_into_model(m, d1, a.strength, a.rank)
    m.save_packed_safetensors(OUT / "merge_base1.safetensors")
    print(f"  wrote {n} matrices -> {OUT / 'merge_base1.safetensors'}", flush=True)
    del m

    # MERGE 2
    print("\n=== MERGE 2: bitnet + falcon3 -> merge_base2 ===", flush=True)
    d2 = pair_delta(hostW, P, ["bitnet", "falcon"])
    m = load_model(BASE)
    n = write_delta_into_model(m, d2, a.strength, a.rank)
    m.save_packed_safetensors(OUT / "merge_base2.safetensors")
    print(f"  wrote {n} matrices -> {OUT / 'merge_base2.safetensors'}", flush=True)
    del m

    # MERGE 3: load the two saved bases, combine their LoRA deltas, save Quillan
    print("\n=== MERGE 3: merge_base1 + merge_base2 -> QUILLAN BASE ===", flush=True)
    m1 = load_model(OUT / "merge_base1.safetensors")
    m2 = load_model(OUT / "merge_base2.safetensors")
    written = 0
    for li in range(len(m1.layers)):
        for sub in SUBS:
            a1 = getattr(m1.layers[li], sub, None)
            a2 = getattr(m2.layers[li], sub, None)
            if a1 is None or a2 is None or not hasattr(a1, "lora_A"):
                continue
            # combined delta = B1@A1 + B2@A2, re-factor to rank
            D = (a1.lora_B.float() @ a1.lora_A.float()) + (
                a2.lora_B.float() @ a2.lora_A.float()
            )
            r = min(a1.lora_A.shape[0], *D.shape)
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            a1.lora_A.zero_()
            a1.lora_B.zero_()
            a1.lora_A[:r].copy_((Vh[:r]).to(a1.lora_A.dtype))
            a1.lora_B[:, :r].copy_((U[:, :r] * S[:r]).to(a1.lora_B.dtype))
            if hasattr(a1, "_wc"):
                a1._wc = None
            written += 1
    del m2
    m1.save_packed_safetensors(OUT / "quillan_base_3merge.safetensors")
    print(
        f"  combined {written} matrices -> {OUT / 'quillan_base_3merge.safetensors'}",
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
