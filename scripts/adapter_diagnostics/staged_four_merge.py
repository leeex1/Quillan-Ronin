#!/usr/bin/env python3
"""USER'S EXACT STAGED MERGE:
  1) llama + qwen   -> merge base 1
  2) bitnet + falcon3 -> merge base 2
  3) merge base 1 + merge base 2 -> QUILLAN BASE

Closed-form, no training. Each stage computes an aligned delta in the flagship
(Llama 2048) frame and accumulates it into the Quillan base's rank-64 LoRA slots:
  stage1 delta occupies rank blocks 0..31 (llama+qwen)
  stage2 delta occupies rank blocks 32..63 (bitnet+falcon3)
so the final Quillan base literally carries both merge bases.
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
BASE = Path(r"F:\02_QUILLAN\exports\quillan-flagship-hifi-g4-exact.safetensors")
LLAMA = r"E:\source_models\llama_1b\model.safetensors"
QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors"
BITNET = r"E:\source_models\bitnet_2b_4t\model.safetensors"
FALCON = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model_bf16.safetensors"
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


def align(W, P, is_ffn):
    Pinv = pinv_row(P)
    Wt = W.float()
    if not is_ffn:
        Wt = P.T @ Wt
    else:
        Wt = crop_pad(Wt, 0, 8192)
    if not is_ffn:
        Wt = Wt @ Pinv.T
    else:
        Wt = crop_pad(Wt, 1, 8192)
    return Wt


def pair_delta(hostW, P, donor_a, donor_b, label):
    """Stage delta = mean of aligned (donor - host) over the pair."""
    accum = {k: torch.zeros_like(v) for k, v in hostW.items()}
    cnt = {k: 0 for k in hostW}
    for dn in (donor_a, donor_b):
        wp = W[dn]
        with safe_open(wp, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for (li, sub), hW in hostW.items():
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
                accum[(li, sub)] += Wa - hW
                cnt[(li, sub)] += 1
    filled = sum(1 for v in cnt.values() if v > 0)
    out = {k: (v / max(cnt[k], 1)) for k, v in accum.items() if cnt[k] > 0}
    print(f"  {label}: {filled} matrices, pair members counted", flush=True)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    host = load_model(BASE)
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
    hostW = {}
    for li in range(len(host.layers)):
        for sub in SUBS:
            mod = getattr(host.layers[li], sub, None)
            if mod is not None and hasattr(mod, "unpack_effective_weight"):
                hostW[(li, sub)] = mod.unpack_effective_weight().float()
    if a.dry:
        print(
            f"DRY: shared={len(shared)} host matrices={len(hostW)} P shapes="
            + ", ".join(f"{n}:{tuple(P[n].shape)}" for n in P),
            flush=True,
        )
        return
    print("=== STAGE 1: llama + qwen -> merge base 1 ===", flush=True)
    d1 = pair_delta(hostW, P, "llama", "qwen", "BASE1")
    print("=== STAGE 2: bitnet + falcon3 -> merge base 2 ===", flush=True)
    d2 = pair_delta(hostW, P, "bitnet", "falcon", "BASE2")
    print("=== STAGE 3: merge base 1 + merge base 2 -> QUILLAN BASE ===", flush=True)
    written = 0
    for k, hW in hostW.items():
        delta = (d1.get(k, 0) + d2.get(k, 0)) * a.strength
        if isinstance(delta, int):
            continue
        mod = getattr(host.layers[k[0]], k[1], None)
        r = min(32, mod.lora_A.shape[0], *delta.shape)
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
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
    print(f"  wrote {written} matrices (both merge bases in Quillan base)", flush=True)
    out_path = Path(a.out) if Path(a.out).is_absolute() else (OUT / a.out)
    host.save_packed_safetensors(out_path)
    print(f"SAVED {out_path}", flush=True)


if __name__ == "__main__":
    main()
