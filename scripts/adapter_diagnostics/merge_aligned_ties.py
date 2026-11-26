#!/usr/bin/env python3
"""ALIGN-THEN-MERGE: closed-form, no training, no backprop.

The 2026 merge literature (HR-Merge, AMARIS, TIES, MaLoRA) all require models
to share a backbone. Our donors don't -- but we can CREATE a shared frame:
  1. Fit P (2048x2048) mapping Falcon hidden states into Llama's frame via
     ridge regression on shared-token embeddings (closed-form, seconds).
  2. Transform every Falcon weight matrix into Llama's frame using P:
     - input aligned by P^-1, output aligned by P (depending on dims)
  3. TIES sign-aware merge the aligned Falcon weights with Llama weights.
  4. Repack into the faithful Llama base, save, evaluate.

This is the only remaining no-training path that respects the literature.
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
LLAMA_TOK = r"E:\source_models\llama_1b\tokenizer.json"
FALCON_TOK = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\tokenizer.json"
FALCON_W = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model_bf16.safetensors"
HOST = ROOT / "exports" / "quillan-flagship-hifi-g4-exact.safetensors"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")


def shared_token_embeddings():
    from tokenizers import Tokenizer

    tl = Tokenizer.from_file(LLAMA_TOK)
    tf = Tokenizer.from_file(FALCON_TOK)
    vl = tl.get_vocab()
    vf = tf.get_vocab()
    shared = sorted(set(vl) & set(vf))
    print(f"shared tokens={len(shared)}", flush=True)
    # load embeddings from safetensors (no full model)
    llE = None
    faE = None
    with safe_open(
        r"E:\source_models\llama_1b\model.safetensors", framework="pt", device="cpu"
    ) as s:
        llE = s.get_tensor("model.embed_tokens.weight").float()
    with safe_open(FALCON_W, framework="pt", device="cpu") as s:
        faE = s.get_tensor("model.embed_tokens.weight").float()
    li = torch.tensor([vl[t] for t in shared])
    fi = torch.tensor([vf[t] for t in shared])
    return llE[li], faE[fi]


def fit_P(Ed, Eh, ridge=1e-2):
    # x_host ~= x_donor @ P ; ridge solve
    G = Ed.T @ Ed + ridge * torch.eye(Ed.shape[1]) * Ed.abs().mean()
    P = torch.linalg.solve(G, Ed.T @ Eh)
    return P


def ties(a, b, trim=0.2):
    sa = torch.sign(a)
    sb = torch.sign(b)
    agree = sa == sb
    out = torch.where(agree, (a + b) / 2, torch.where(a.abs() >= b.abs(), a, b))
    k = int(trim * out.numel())
    if k > 0:
        th = torch.kthvalue(out.abs().flatten(), k + 1).values
        out = torch.where(out.abs() < th, torch.zeros_like(out), out)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge", type=float, default=1e-2)
    a = ap.parse_args()
    Ed, Eh = shared_token_embeddings()
    P = fit_P(Ed, Eh, a.ridge)
    Pinv = torch.linalg.pinv(P)
    cos = torch.nn.functional.cosine_similarity(Ed @ P, Eh, dim=-1).mean().item()
    print(
        f"P shape={tuple(P.shape)} aligned_cos={cos:.4f} cond={torch.linalg.cond(P).item():.1f}",
        flush=True,
    )
    host = load_model(HOST)
    falcon_layers = [round(i * 17 / 15) for i in range(16)]
    with safe_open(FALCON_W, framework="pt", device="cpu") as ff:
        n = 0
        for li, layer in enumerate(host.layers):
            fl = falcon_layers[li]
            for sub in ATTN + FFN:
                mm = getattr(layer, sub, None)
                if mm is None or not hasattr(mm, "unpack_effective_weight"):
                    continue
                Wh = mm.unpack_effective_weight().float()
                fk = (
                    f"model.layers.{fl}."
                    + ("self_attn." if sub in ATTN else "mlp.")
                    + sub
                    + ".weight"
                )
                if fk not in ff.keys():
                    continue
                Wf = ff.get_tensor(fk).float()
                if tuple(Wf.shape) != tuple(Wh.shape):
                    if (
                        sub in ("k_proj", "v_proj")
                        and Wf.shape[0] == 1024
                        and Wh.shape[0] == 512
                    ):
                        Wf = Wf.view(512, 2, Wf.shape[1]).mean(1)
                    else:
                        continue
                # align Falcon weight into Llama frame
                # input dim (in_features) aligned by Pinv; output dim aligned by P
                Wa = Wf
                if Wf.shape[1] == 2048:
                    Wa = Pinv @ Wa  # input hidden -> llama frame
                if Wf.shape[0] == 2048:
                    Wa = Wa @ P  # output hidden -> llama frame
                Wm = ties(Wh, Wa)
                mm.set_weights_from_float(Wm)
                n += 1
            if li % 4 == 0:
                print(f"  layer {li}", flush=True)
    out = ROOT / "exports" / a.out
    host.save_packed_safetensors(out)
    print(
        f"SAVED {out} merged={n} cos={cos:.4f} size={out.stat().st_size / 1e6:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
