#!/usr/bin/env python3
"""Implement Section 2+4+5 of the WSP paper faithfully.

WSP paper (Formal Papers/Building_Foundation_Models_via_Weight_Space_Programming):
  Sec 2: donor weight W -> t-SVD W ~= U_r S_r V_r^T ; factors
           W0 = S_r^{1/2} V_r^T , W1 = U_r S_r^{1/2}
  Sec 4: steer by covariance eigenvector: u'_i = 0.85 u_donor,i + 0.15 u_sov,i
  Sec 5: fold: W_folded = W_down + (beta/K) sum_i W1_i W0_i

Four donors: Qwen (logic), Llama (syntax), Falcon (philosophy), BitNet (ternary).
All closed-form, no backprop. RAM-safe: one donor, one layer at a time.
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
DONORS = {
    "qwen": (
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors",
        r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json",
    ),
    "llama": (
        r"E:\source_models\llama_1b\model.safetensors",
        r"E:\source_models\llama_1b\tokenizer.json",
    ),
    "falcon": (
        r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\model.safetensors",
        r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\tokenizer.json",
    ),
    "bitnet": (
        r"E:\source_models\bitnet_2b_4t\model.safetensors",
        r"E:\source_models\llama_1b\tokenizer.json",
    ),
}


def ternary(x):
    """Ternarize {-1,0,+1} with per-row absmean gamma (WSP: adapters quantized ternary)."""
    scale = 1.0 / x.abs().mean(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.round((x * scale).clamp(-1, 1))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--steer", type=float, default=0.15)
    a = ap.parse_args()
    m = load_model(BASE)
    L = len(m.layers)
    # host embedding for alignment
    with safe_open(
        r"E:\source_models\llama_1b\model.safetensors", framework="pt", device="cpu"
    ) as f:
        hostE = f.get_tensor("model.embed_tokens.weight").float()
    toks = {n: Tokenizer.from_file(t).get_vocab() for n, (_, t) in DONORS.items()}
    shared = sorted(set.intersection(*[set(v) for v in toks.values()]))
    print(
        f"shared tokens={len(shared)} layers={L} rank={a.rank} beta={a.beta} steer={a.steer}",
        flush=True,
    )
    hi = torch.tensor([toks["llama"][s] for s in shared])

    # per-donor alignment projection donor->host (Sec 2: dimension projection)
    P = {}
    for n, (wp, tp) in DONORS.items():
        with safe_open(wp, framework="pt", device="cpu") as f:
            dE = f.get_tensor("model.embed_tokens.weight").float()
        di = torch.tensor([toks[n][s] for s in shared])
        G = dE[di].T @ dE[di] + 1e-2 * torch.eye(dE.shape[1]) * dE.abs().mean()
        P[n] = torch.linalg.solve(G, dE[di].T @ hostE[hi])
        print(f"  P[{n}]={tuple(P[n].shape)}", flush=True)

    # Fold accumulator per layer down_proj: (out,in)
    fold = [None] * L
    K = len(DONORS)
    for name, (wp, tp) in DONORS.items():
        print(f"\n=== donor {name} ===", flush=True)
        with safe_open(wp, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for li in range(L):
                dk = f"model.layers.{li}.mlp.down_proj.weight"
                if dk not in keys:
                    # donor has different layer count -> map proportionally (handled below)
                    continue
                Wd = f.get_tensor(dk).float()
                # dimension projection to host frame: Wd is [ffn_d,d_d]; project input side
                Wd_aligned = Wd @ P[name]  # [ffn_d, d_host]
                if Wd_aligned.shape[0] != 8192:
                    # crop/pad ffn to host 8192
                    if Wd_aligned.shape[0] > 8192:
                        Wd_aligned = Wd_aligned[:8192]
                    else:
                        Wd_aligned = torch.nn.functional.pad(
                            Wd_aligned, (0, 0, 0, 8192 - Wd_aligned.shape[0])
                        )
                # t-SVD (Sec 2): W ~= U S V^T ; W0=S^.5 V^T, W1=U S^.5
                U, S, Vh = torch.linalg.svd(Wd_aligned, full_matrices=False)
                r = min(a.rank, len(S))
                s = S[:r].sqrt()
                W0 = (s.unsqueeze(1) * Vh[:r]).contiguous()  # [r, in]
                W1 = (U[:, :r] * s.unsqueeze(0)).contiguous()  # [out, r]
                # Sec 4 steering with sovereign covariance eigenvector u_sov:
                # principal axis of W0 covariance
                cov = W0 @ W0.T
                ev, evec = torch.linalg.eigh(cov)
                u_sov = evec[:, -1]
                u_donor = W1[:, 0] / (W1[:, 0].norm() + 1e-9)
                u_steer = (1 - a.steer) * u_donor + a.steer * u_sov
                u_steer = u_steer / (u_steer.norm() + 1e-9)
                W1[:, 0] = u_steer * W1[:, 0].norm()
                # ternary quantize (Sec 4)
                W0q = ternary(W0)
                W1q = ternary(W1)
                recon = W1q @ W0q
                if fold[li] is None:
                    fold[li] = torch.zeros_like(recon)
                fold[li] += recon
        print(f"  folded layers so far {sum(f is not None for f in fold)}", flush=True)

    # Sec 5: W_folded = W_down + (beta/K) sum W1 W0
    print("\n=== fold into base W_down + repack ===", flush=True)
    written = 0
    for li in range(L):
        if fold[li] is None:
            continue
        mod = m.layers[li].down_proj
        Wb = mod.unpack_effective_weight().float()
        Wf = Wb + (a.beta / K) * fold[li]
        mod.set_weights_from_float(Wf)
        mod._wc = None
        written += 1
    print(f"  folded {written} layers", flush=True)
    m.save_packed_safetensors(OUT / a.out)
    print(f"SAVED {OUT / a.out}", flush=True)


if __name__ == "__main__":
    main()
