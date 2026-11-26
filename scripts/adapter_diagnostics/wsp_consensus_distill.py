#!/usr/bin/env python3
"""CLOSED-FORM FOUR-DONOR (3 valid) CONSENSUS DISTILLATION into flagship student.
No backprop. Uses the repo's online_wsp_update (rank-r closed-form write).

Recipe:
  1. Load logged final hidden states from each valid donor (qwen, llama, falcon).
  2. Align each into student (2048) frame via ridge P fitted on shared-token embeddings.
  3. Consensus target per prompt = mean of aligned donor hidden states (scale-normalized).
  4. For each calibration prompt, use student's last-layer down_proj input as key k,
     consensus hidden as target v, and apply online_wsp_update (closed-form, no grad).
  5. Save student checkpoint; evaluate on a FRESH bank (heldout_domain_large).
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
VALID = ["qwen", "llama", "falcon"]


def fit_align(Ed, Eh, ridge=1e-2):
    G = Ed.T @ Ed + ridge * torch.eye(Ed.shape[1]) * Ed.abs().mean()
    return torch.linalg.solve(G, Ed.T @ Eh)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.5)
    a = ap.parse_args()
    # shared-token embeddings for alignment P (student=llama frame)
    from safetensors import safe_open
    from tokenizers import Tokenizer

    tokpaths = {
        "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json",
        "llama": r"E:\source_models\llama_1b\tokenizer.json",
        "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\tokenizer.json",
    }
    wpaths = {
        "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors",
        "llama": r"E:\source_models\llama_1b\model.safetensors",
        "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\model.safetensors",
    }
    toks = {k: Tokenizer.from_file(v).get_vocab() for k, v in tokpaths.items()}
    shared = sorted(set.intersection(*[set(v) for v in toks.values()]))
    print(f"shared tokens={len(shared)}", flush=True)
    hostE = None
    with safe_open(wpaths["llama"], framework="pt", device="cpu") as f:
        hostE = f.get_tensor("model.embed_tokens.weight").float()
    hi = torch.tensor([toks["llama"][t] for t in shared])
    Ps = {}
    for d in VALID:
        with safe_open(wpaths[d], framework="pt", device="cpu") as f:
            dE = f.get_tensor("model.embed_tokens.weight").float()
        di = torch.tensor([toks[d][t] for t in shared])
        Ps[d] = fit_align(dE[di], hostE[hi])
        print(f"  P[{d}] {tuple(Ps[d].shape)}", flush=True)
    # load logged hidden, align, scale-normalize to student scale, average
    Haligned = []
    for d in VALID:
        H = torch.load(CACHE / f"{d}_hidden.pt", weights_only=True).float()
        Ha = H @ Ps[d]  # (40,2048)
        s = 30.0 / Ha.norm(dim=1).mean()  # normalize to llama scale
        Haligned.append(Ha * s)
        cos = (
            torch.nn.functional.cosine_similarity(
                Ha,
                torch.load(CACHE / "llama_hidden.pt", weights_only=True).float()
                @ Ps["llama"],
                dim=-1,
            )
            .mean()
            .item()
        )
        print(
            f"  {d} aligned norm={Ha.norm(dim=1).mean().item():.1f} cos-to-llama={cos:.3f}",
            flush=True,
        )
    target = torch.stack(Haligned).mean(0)  # (40,2048) consensus
    # load student, WSP-write from last-layer down_proj activations toward consensus
    m = load_model(STUDENT)
    dp = m.layers[-1].down_proj
    cap = {}
    hk = dp.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    tok = Tokenizer.from_file(tokpaths["llama"])

    class Shim:
        def encode(self, t, add_special_tokens=True):
            return tok.encode(t, add_special_tokens=add_special_tokens).ids

    seqs = pf(Shim())
    keys = []
    with torch.inference_mode():
        for s in seqs:
            m(torch.tensor([s]))
            keys.append(cap["k"][0, -1, :].detach().clone())
    hk.remove()
    K = torch.stack(keys)
    print(f"keys {tuple(K.shape)} target {tuple(target.shape)}", flush=True)
    # closed-form WSP: one batch update mapping keys -> (target - current final hidden)
    with torch.no_grad():
        dp._wc = None
        # online_wsp_update expects k:(in_dim,n_cols), v:(out_dim,n_cols).
        dp.online_wsp_update(
            K.T.contiguous(),
            target.T.contiguous() * a.strength,
            a_anchor=None,
            ridge_lambda=1e-2,
            max_rel_frob=0.05,
        )
    out = ROOT / "exports" / a.out
    m.save_packed_safetensors(out)
    print(f"SAVED {out} rank={a.rank}", flush=True)


if __name__ == "__main__":
    main()
