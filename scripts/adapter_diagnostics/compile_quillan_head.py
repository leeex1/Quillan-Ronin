#!/usr/bin/env python3
"""COMPILE STEP 3: reindex all four donor lm_head matrices into Quillan token IDs.

Same mechanism as compile_quillan_embedding.py, applied to the output head.
Consumes quillan_token_maps_q2d.pt. Closed-form, no training.
"""

from __future__ import annotations
import sys
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


def load_head(path: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in ("lm_head.weight", "output.weight", "model.output.weight"):
            if k in keys:
                return f.get_tensor(k).float()
        for k in keys:
            t = f.get_tensor(k)
            if t.ndim == 2 and t.shape[0] > 10000:
                print(f"    (fallback key {k}, {tuple(t.shape)})", flush=True)
                return t.float()
    raise RuntimeError(f"no head tensor in {path}")


def fit_ridge(X, Y, ridge=1e-2):
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    G = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    return torch.linalg.solve(G, X.T @ Y)


def main():
    q2d = torch.load(OUT / "quillan_token_maps_q2d.pt", weights_only=True)
    Q = q2d["llama"].shape[0]
    common = torch.ones(Q, dtype=torch.bool)
    for name in MODEL_FILES:
        common &= q2d[name] >= 0
    common_q = torch.nonzero(common).flatten()
    perm = torch.randperm(len(common_q))
    fit_q = common_q[perm[: int(0.8 * len(perm))]]
    held_q = common_q[perm[int(0.8 * len(perm)) :]]
    print(
        f"shared-by-all-4={len(common_q)} fit={len(fit_q)} held={len(held_q)}",
        flush=True,
    )

    print("\n=== load heads ===", flush=True)
    H = {}
    for name, path in MODEL_FILES.items():
        if not Path(path).exists():
            print(f"  {name}: MISSING", flush=True)
            continue
        H[name] = load_head(path)
        print(f"  {name}: {tuple(H[name].shape)}", flush=True)

    D = H["llama"].shape[1]
    print(f"\n=== fit head align maps -> {D}d ===", flush=True)
    P = {}
    for name in MODEL_FILES:
        if name not in H:
            continue
        if name == "llama":
            P[name] = torch.eye(D, dtype=torch.float32)
            continue
        X = H[name][q2d[name][fit_q]]
        Y = H["llama"][q2d["llama"][fit_q]]
        if X.shape[1] != H[name].shape[1]:
            continue
        P[name] = fit_ridge(X, Y)
        Xh = H[name][q2d[name][held_q]]
        Yh = H["llama"][q2d["llama"][held_q]]
        cos = (
            torch.nn.functional.cosine_similarity(Xh @ P[name], Yh, dim=-1)
            .mean()
            .item()
        )
        print(f"  {name} {tuple(P[name].shape)} heldout_cos={cos:.3f}", flush=True)

    print("\n=== compile canonical head ===", flush=True)
    Hc = torch.zeros(Q, D, dtype=torch.float32)
    cnt = torch.zeros(Q, dtype=torch.int32)
    for name in MODEL_FILES:
        if name not in H or name not in P:
            continue
        rows = torch.nonzero(q2d[name] >= 0).flatten()
        vals = H[name][q2d[name][rows]]
        if P[name].shape[0] != vals.shape[1]:
            continue
        vals = vals @ P[name]
        Hc[rows] += vals
        cnt[rows] += 1
    Hc = Hc / (cnt > 0).float().clamp_min(1).unsqueeze(1)
    print(
        f"  filled rows={int((cnt > 0).sum())} / {Q}  all4={int((cnt == 4).sum())}",
        flush=True,
    )
    print(f"  norm mean={Hc.norm(dim=-1).mean():.4f}", flush=True)

    torch.save(
        {"head": Hc, "coverage": cnt, "q2d": q2d}, OUT / "quillan_head_compiled.pt"
    )
    from safetensors.torch import save_file

    save_file(
        {"lm_head.weight": Hc.contiguous()},
        str(OUT / "quillan_head_compiled.safetensors"),
    )
    print("SAVED quillan_head_compiled.pt / .safetensors", flush=True)


if __name__ == "__main__":
    main()
