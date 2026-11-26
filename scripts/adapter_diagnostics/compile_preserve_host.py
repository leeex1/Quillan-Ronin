#!/usr/bin/env python3
"""COMPILE (preserve-host): keep host embedding EXACT for every host token;
fill only slots the host lacks from the other donors (ridge-projected).

Rationale: averaging broke the matched embedding<->body pair (PPL 9-11 vs 1.5).
This version leaves every host-token row bit-identical to the host, so the
body sees exactly what it was trained on; new vocabulary slots get donor
knowledge. No training.
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

FILES = {
    "qwen": r"E:\source_models\qwen_0.5b\model.safetensors",
    "llama": r"E:\source_models\llama_1b\model.safetensors",
    "bitnet": r"E:\source_models\bitnet_2b_4t\model.safetensors",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model.safetensors",
}
HOST = "llama"


def load_emb(path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        for k in ("model.embed_tokens.weight", "embed_tokens.weight"):
            if k in f.keys():
                return f.get_tensor(k).float()
        for k in f.keys():
            t = f.get_tensor(k)
            if t.ndim == 2 and t.shape[0] > 10000:
                return t.float()
    raise RuntimeError(path)


def fit_ridge(X, Y, ridge=1e-2):
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    return torch.linalg.solve(
        X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y
    )


def main():
    q2d = torch.load(OUT / "quillan_token_maps_q2d.pt", weights_only=True)
    Q = q2d[HOST].shape[0]
    E = {n: load_emb(p) for n, p in FILES.items() if Path(p).exists()}
    host = E[HOST]
    D = host.shape[1]
    print(f"Q={Q} host_dim={D}", flush=True)

    # shared tokens (all 4) for ridge fitting
    common = torch.ones(Q, dtype=torch.bool)
    for n in FILES:
        common &= q2d[n] >= 0
    cq = torch.nonzero(common).flatten()
    perm = torch.randperm(len(cq))
    fit_q = cq[perm[: int(0.8 * len(perm))]]

    # fit donor->host maps
    P = {}
    for n in FILES:
        if n == HOST or n not in E:
            continue
        X = E[n][q2d[n][fit_q]]
        Y = host[q2d[HOST][fit_q]]
        P[n] = fit_ridge(X, Y)
        print(f"  {n} map {tuple(P[n].shape)}", flush=True)

    Ec = host[
        q2d[HOST].clamp_min(0)
    ].clone()  # default: host row for host tokens (exact)
    # zero rows for slots host lacks
    missing_host = q2d[HOST] < 0
    Ec[missing_host] = 0
    filled_new = 0
    for n in FILES:
        if n == HOST or n not in E:
            continue
        # only fill slots the host lacks
        sel = missing_host & (q2d[n] >= 0)
        rows = torch.nonzero(sel).flatten()
        if len(rows) == 0:
            continue
        vals = E[n][q2d[n][rows]] @ P[n]
        Ec[rows] += vals
        filled_new += len(rows)
    # normalize the + contributions where >1 donor filled
    cnt = missing_host.int()
    for n in FILES:
        if n == HOST or n not in E:
            continue
        cnt += (missing_host & (q2d[n] >= 0)).int()
    Ec = Ec / cnt.clamp_min(1).unsqueeze(1)

    # VERIFY host tokens bit-identical
    host_rows = q2d[HOST] >= 0
    same = torch.equal(Ec[host_rows], host[q2d[HOST][host_rows]])
    print(f"host-token rows bit-identical: {same}", flush=True)
    print(f"new slots filled by other donors: {filled_new}", flush=True)
    print(f"empty rows: {int((Ec.norm(dim=-1) == 0).sum())}", flush=True)

    torch.save({"embedding": Ec, "q2d": q2d}, OUT / "quillan_embedding_preserve.pt")
    # assemble checkpoint
    from safetensors import safe_open
    from safetensors.torch import save_file
    import shutil, json as J

    DEST = OUT / "quillan_flagship_preserve"
    DEST.mkdir(exist_ok=True)
    tensors = {}
    with safe_open(FILES[HOST], framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in ("model.embed_tokens.weight", "lm_head.weight"):
                t = Ec
            tensors[k] = t.to(torch.bfloat16).contiguous()
    save_file(tensors, str(DEST / "model.safetensors"))
    cfg = J.loads((Path(r"E:\source_models\llama_1b") / "config.json").read_text())
    cfg["vocab_size"] = Q
    (DEST / "config.json").write_text(J.dumps(cfg, indent=2))
    for fn in ("generation_config.json",):
        s = Path(r"E:\source_models\llama_1b") / fn
        if s.exists():
            shutil.copy(s, DEST / fn)
    print(f"SAVED {DEST}", flush=True)


if __name__ == "__main__":
    main()
