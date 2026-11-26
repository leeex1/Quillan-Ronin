#!/usr/bin/env python3
"""Compile the MINI base: all four donors reindexed into Quillan IDs, 896-d frame."""

from __future__ import annotations
import sys, json, shutil
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
DEST = OUT / "quillan_mini_compiled"
QWEN = Path(r"E:\source_models\qwen_0.5b")
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
HOST = "qwen"


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
    common = torch.ones(Q, dtype=torch.bool)
    for n in MODEL_FILES:
        common &= q2d[n] >= 0
    cq = torch.nonzero(common).flatten()
    perm = torch.randperm(len(cq))
    fit_q, held_q = cq[perm[: int(0.8 * len(perm))]], cq[perm[int(0.8 * len(perm)) :]]
    print(f"shared={len(cq)} fit={len(fit_q)} held={len(held_q)}", flush=True)

    D = load_emb(MODEL_FILES[HOST]).shape[1]
    print(f"target dim={D}", flush=True)
    E = {}
    for n, p in MODEL_FILES.items():
        if Path(p).exists():
            E[n] = load_emb(p)
            print(f"  loaded {n} {tuple(E[n].shape)}", flush=True)

    P = {}
    for n in MODEL_FILES:
        if n not in E:
            continue
        if n == HOST:
            P[n] = torch.eye(D, dtype=torch.float32)
            continue
        X = E[n][q2d[n][fit_q]]
        Y = E[HOST][q2d[HOST][fit_q]]
        P[n] = fit_ridge(X, Y)
        cos = (
            torch.nn.functional.cosine_similarity(
                E[n][q2d[n][held_q]] @ P[n], E[HOST][q2d[HOST][held_q]], dim=-1
            )
            .mean()
            .item()
        )
        print(f"  {n} heldout_cos={cos:.3f}", flush=True)

    Ec = torch.zeros(Q, D, dtype=torch.float32)
    cnt = torch.zeros(Q, dtype=torch.int32)
    for n in MODEL_FILES:
        if n not in E or n not in P:
            continue
        rows = torch.nonzero(q2d[n] >= 0).flatten()
        vals = E[n][q2d[n][rows]]
        if P[n].shape[0] == vals.shape[1]:
            Ec[rows] += vals @ P[n]
            cnt[rows] += 1
    Ec = Ec / (cnt > 0).float().clamp_min(1).unsqueeze(1)
    print(
        f"compiled emb {tuple(Ec.shape)} all4={int((cnt == 4).sum())} norm={Ec.norm(dim=-1).mean():.4f}",
        flush=True,
    )

    # assemble Qwen body + compiled emb/head
    from safetensors import safe_open
    from safetensors.torch import save_file

    DEST.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with safe_open(MODEL_FILES[HOST], framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in ("model.embed_tokens.weight", "lm_head.weight"):
                t = Ec
            tensors[k] = t.to(torch.bfloat16).contiguous()
    save_file(tensors, str(DEST / "model.safetensors"))
    cfg = json.loads((QWEN / "config.json").read_text(encoding="utf-8"))
    cfg["vocab_size"] = Q
    (DEST / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for fn in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        if (QWEN / fn).exists():
            shutil.copy(QWEN / fn, DEST / fn)
    print(f"SAVED {DEST} vocab={Q}", flush=True)

    # smoke test with canonical encoder
    text2q = json.loads((OUT / "quillan_vocab.json").read_text(encoding="utf-8"))
    q2text = {v: k for k, v in text2q.items()}
    by_first = {}
    for s in text2q:
        if s:
            by_first.setdefault(s[0], []).append(s)
    for k in by_first:
        by_first[k].sort(key=len, reverse=True)

    def enc(text):
        ids = []
        i = 0
        while i < len(text):
            m = None
            for s in by_first.get(text[i], ()):
                if text.startswith(s, i):
                    m = s
                    break
            if m is None:
                i += 1
                continue
            ids.append(text2q[m])
            i += len(m)
        return ids

    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        str(DEST), local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    print("\n=== smoke ===", flush=True)
    with torch.inference_mode():
        for p in ["The capital of France is", "2 + 2 ="]:
            ids = enc(p)
            out = m.generate(
                torch.tensor([ids]), max_new_tokens=14, do_sample=False, pad_token_id=0
            )
            gen = out[0, len(ids) :].tolist()
            print(
                f"  {p!r} -> {''.join(q2text.get(t, f'<{t}>') for t in gen)!r}",
                flush=True,
            )


if __name__ == "__main__":
    main()
