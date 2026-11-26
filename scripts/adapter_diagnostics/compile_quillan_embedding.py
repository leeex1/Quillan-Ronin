#!/usr/bin/env python3
"""COMPILE STEP 2: reindex all four donor embeddings into Quillan token IDs.

Consumes quillan_token_maps_q2d.pt. For each donor:
  - load only embed_tokens.weight (no full model, no training)
  - fit a closed-form ridge map P_d: donor_hidden -> flagship_hidden(2048)
    on the tokens shared by all four donors, with HELD-OUT validation
  - place E_d[row] @ P_d into canonical row q

Output: one embedding tensor (235845, 2048) where all four donors contribute.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)

MODEL_FILES = {
    "qwen": r"E:\source_models\qwen_0.5b\model.safetensors",
    "llama": r"E:\source_models\llama_1b\model.safetensors",
    "bitnet": r"E:\source_models\bitnet_2b_4t\model.safetensors",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model.safetensors",
}
DIM = 2048  # flagship hidden (Llama/Falcon native)


def load_embedding(path: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in (
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.embedding.weight",
            "transformer.wte.weight",
        ):
            if k in keys:
                return f.get_tensor(k).float()
        # fallback: first 2-D tensor whose rows look like a vocab
        for k in keys:
            t = f.get_tensor(k)
            if t.ndim == 2 and t.shape[0] > 10000:
                print(f"    (fallback key {k}, shape {tuple(t.shape)})", flush=True)
                return t.float()
    raise RuntimeError(f"no embedding tensor in {path}")


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    G = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    return torch.linalg.solve(G, X.T @ Y)


def main():
    q2d = torch.load(OUT / "quillan_token_maps_q2d.pt", weights_only=True)
    Q = q2d["llama"].shape[0]
    # tokens present in ALL four donors
    common = torch.ones(Q, dtype=torch.bool)
    for name in MODEL_FILES:
        common &= q2d[name] >= 0
    common_q = torch.nonzero(common).flatten()
    print(f"canonical slots={Q}  shared-by-all-4={len(common_q)}", flush=True)
    perm = torch.randperm(len(common_q))
    fit_q = common_q[perm[: int(0.8 * len(perm))]]
    held_q = common_q[perm[int(0.8 * len(perm)) :]]
    print(f"fit={len(fit_q)} held_out={len(held_q)}", flush=True)

    print("\n=== load all embeddings ===", flush=True)
    E = {}
    for name, path in MODEL_FILES.items():
        print(f"  {name}: {path}", flush=True)
        if not Path(path).exists():
            print("    MISSING -> skipped", flush=True)
            continue
        E[name] = load_embedding(path)
        print(f"    shape={tuple(E[name].shape)}", flush=True)

    print("\n=== fit align maps (donor -> llama frame) ===", flush=True)
    P = {}
    for name in MODEL_FILES:
        if name not in E:
            continue
        if name == "llama":
            P[name] = torch.eye(E[name].shape[1], dtype=torch.float32)
            continue
        X = E[name][q2d[name][fit_q]]
        Y = E["llama"][q2d["llama"][fit_q]]
        P[name] = fit_ridge(X, Y)
        Xh = E[name][q2d[name][held_q]]
        Yh = E["llama"][q2d["llama"][held_q]]
        cos = (
            torch.nn.functional.cosine_similarity(Xh @ P[name], Yh, dim=-1)
            .mean()
            .item()
        )
        print(
            f"    {name} align {tuple(P[name].shape)} heldout_cos={cos:.3f}", flush=True
        )

    print("\n=== compile canonical embedding ===", flush=True)
    Ec = torch.zeros(Q, DIM, dtype=torch.float32)
    cnt = torch.zeros(Q, dtype=torch.int32)
    for name in MODEL_FILES:
        if name not in E or P.get(name) is None:
            continue
        rows = torch.nonzero(q2d[name] >= 0).flatten()
        vals = E[name][q2d[name][rows]]
        if P[name].shape[0] == vals.shape[1]:
            vals = vals @ P[name]
        else:
            continue
        Ec[rows] += vals
        cnt[rows] += 1
    nz = (cnt > 0).float().clamp_min(1).unsqueeze(1)
    Ec = Ec / nz
    print(f"  filled rows={int((cnt > 0).sum())} / {Q}", flush=True)
    print(f"  contributed-by-4 rows={int((cnt == 4).sum())}", flush=True)
    print(f"  norm mean={Ec.norm(dim=-1).mean():.4f}", flush=True)

    torch.save(
        {"embedding": Ec, "coverage": cnt, "q2d": q2d},
        OUT / "quillan_embedding_compiled.pt",
    )
    from safetensors.torch import save_file

    save_file(
        {"model.embed_tokens.weight": Ec.contiguous()},
        str(OUT / "quillan_embedding_compiled.safetensors"),
    )
    print("SAVED quillan_embedding_compiled.pt / .safetensors", flush=True)


if __name__ == "__main__":
    main()
