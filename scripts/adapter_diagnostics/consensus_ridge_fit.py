#!/usr/bin/env python3
"""Final closed-form four-donor integration test. Simple ridge least-squares,
no online_wsp_update complexity. Writes rank-r correction to down_proj lora slots.

k = student last-layer down_proj INPUT (8192-dim) per calibration prompt
current = student down_proj OUTPUT (2048-dim)
target = aligned+scaled consensus of donor final hidden states (2048-dim)
Delta = (target - current) @ pinv(K)   [closed-form, no grad]
Store rank-r SVD of Delta in lora_B/lora_A.
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
from heldout_final40 import prompts as pf

torch.set_num_threads(2)
CACHE = ROOT / "exports" / "donor_hidden_cache"
STUDENT = ROOT / "exports" / "quillan-flagship-hifi-g4-exact.safetensors"
N = 10


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    # consensus target from cached donor hidden (already aligned to llama frame in log step? No - raw). Use llama as frame anchor.
    Hl = torch.load(CACHE / "llama_hidden.pt", weights_only=True).float()[
        :N
    ]  # (N,2048)
    Hq = torch.load(CACHE / "qwen_hidden.pt", weights_only=True).float()[:N]
    Hf = torch.load(CACHE / "falcon_hidden.pt", weights_only=True).float()[:N]
    # qwen/falcon need alignment; load P from cached fit via re-fit on shared embeddings
    from safetensors import safe_open
    from tokenizers import Tokenizer

    tp = {
        "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json",
        "llama": r"E:\source_models\llama_1b\tokenizer.json",
        "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\tokenizer.json",
    }
    wp = {
        "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors",
        "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\model.safetensors",
    }
    toks = {k: Tokenizer.from_file(v).get_vocab() for k, v in tp.items()}
    shared = sorted(set.intersection(*[set(v) for v in toks.values()]))
    with safe_open(
        r"E:\source_models\llama_1b\model.safetensors", framework="pt", device="cpu"
    ) as f:
        hostE = f.get_tensor("model.embed_tokens.weight").float()
    hi = torch.tensor([toks["llama"][t] for t in shared])

    def align(H, d):
        with safe_open(wp[d], framework="pt", device="cpu") as f:
            dE = f.get_tensor("model.embed_tokens.weight").float()
        di = torch.tensor([toks[d][t] for t in shared])
        G = dE[di].T @ dE[di] + 1e-2 * torch.eye(dE.shape[1]) * dE.abs().mean()
        P = torch.linalg.solve(G, dE[di].T @ hostE[hi])
        Ha = H @ P
        return Ha * (Hl.norm(dim=1).mean() / Ha.norm(dim=1).mean())

    HqA = align(Hq, "qwen")
    HfA = align(Hf, "falcon")

    # weight by cosine to llama (falcon near-orthogonal -> low weight)
    def w(H):
        return (
            torch.nn.functional.cosine_similarity(H, Hl, dim=-1)
            .clamp(min=0)
            .mean()
            .item()
        )

    wq, wf = w(HqA), w(HfA)
    print(f"align cos qwen={wq:.3f} falcon={wf:.3f}", flush=True)
    target = (Hl + wq * HqA + wf * HfA) / (1 + wq + wf)
    # student: log k (down_proj input) and current output
    m = load_model(STUDENT)
    dp = m.layers[-1].down_proj
    cap = {}
    hk = dp.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    hk2 = dp.register_forward_hook(
        lambda mod, inp, out: cap.__setitem__("out", out.detach().float())
    )
    tok = Tokenizer.from_file(tp["llama"])

    class Shim:
        def encode(self, t, add_special_tokens=True):
            return tok.encode(t, add_special_tokens=add_special_tokens).ids

    seqs = pf(Shim())[:N]
    Ks = []
    outs = []
    with torch.inference_mode():
        for s in seqs:
            m(torch.tensor([s]))
            Ks.append(cap["k"][0, -1, :].detach().clone())
            outs.append(cap["out"][0, -1, :].detach().clone())
    hk.remove()
    hk2.remove()
    K = torch.stack(Ks)
    cur = torch.stack(outs)
    print(
        f"K={tuple(K.shape)} cur={tuple(cur.shape)} target={tuple(target.shape)}",
        flush=True,
    )
    # closed-form ridge: Delta = E @ K.T (K K.T + ridge I)^-1
    E = (target - cur) * a.strength
    Gk = K @ K.T + 1e-2 * torch.eye(N) * K.norm().item()
    Delta = E.T @ torch.linalg.solve(Gk, K)  # (2048,10)@(10,8192)->(2048,8192)
    print(f"Delta norm={Delta.norm().item():.3f}", flush=True)
    # rank-r SVD into lora, padded to the model's fixed lora rank
    r = min(a.rank, dp.lora_A.shape[0], *Delta.shape)
    U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)
    A = Vh[:r].contiguous()
    B = (U[:, :r] * S[:r]).contiguous()
    pad = dp.lora_A.shape[0] - r
    if pad > 0:
        A = torch.nn.functional.pad(A, (0, 0, 0, pad))
        B = torch.nn.functional.pad(B, (0, pad))
    with torch.no_grad():
        dp.lora_A.copy_(A.to(dp.lora_A.dtype))
        dp.lora_B.copy_(B.to(dp.lora_B.dtype))
        dp._wc = None
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out} rank={r}", flush=True)


if __name__ == "__main__":
    main()
