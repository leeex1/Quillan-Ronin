#!/usr/bin/env python3
"""Surgical alpha sweep: blend donor embeddings into host rows at small alpha.
Load host model ONCE, swap embedding+head in memory, eval. No training, no disk reload.
Tests whether a small precise donor contribution improves CE vs full-average collapse.
"""

from __future__ import annotations
import sys, json, math
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)
LETTERS = ["A", "B", "C", "D"]
FILES = {
    "qwen": r"E:\source_models\qwen_0.5b\model.safetensors",
    "llama": r"E:\source_models\llama_1b\model.safetensors",
    "bitnet": r"E:\source_models\bitnet_2b_4t\model.safetensors",
    "falcon": r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf\model.safetensors",
}


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


def fit_ridge(X, Y, ridge=1e-2):
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    return torch.linalg.solve(
        X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y
    )


def letter_ids(tok, vocab):
    out = []
    for L in LETTERS:
        for c in (f" {L}", L):
            e = tok.encode(c, add_special_tokens=False)
            if len(e) == 1 and e[0] < vocab:
                out.append(e[0])
                break
        else:
            out.append(tok.encode(f" {L}", add_special_tokens=False)[0] % vocab)
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        r"E:\source_models\llama_1b", local_files_only=True
    )
    m = AutoModelForCausalLM.from_pretrained(
        r"E:\source_models\llama_1b",
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    d2q = json.loads((OUT / "quillan_token_maps.json").read_text())["donor_to_quillan"]
    q2d = {n: {int(v): int(k) for k, v in d.items()} for n, d in d2q.items()}
    # common native llama ids that all donors share (via quillan)
    q2dd = {
        n: torch.load(OUT / "quillan_token_maps_q2d.pt", weights_only=True)[n]
        for n in FILES
    }
    nv = m.get_input_embeddings().weight.shape[0]
    common = []
    for lid in range(nv):
        q = None
        # find quillan slot for this llama id
        pass
    # build native-id -> donor-embedding mapping using quillan as pivot
    E = {n: load_emb(p) for n, p in FILES.items()}
    host = E["llama"]
    # fit donor->llama maps on native llama ids present in donors
    # use a sample of llama ids that exist in all donors
    # llama native id L -> quillan q -> donor native id
    qllama = q2dd["llama"]
    cand = []
    for L in range(nv):
        q = int(qllama[L]) if L < len(qllama) and qllama[L] >= 0 else -1
        if q < 0:
            continue
        ok = True
        for n in ("qwen", "bitnet", "falcon"):
            if q2dd[n][q] < 0:
                ok = False
                break
        if ok:
            cand.append((L, q))
    print(f"shared native llama ids={len(cand)}", flush=True)
    sel = cand[:8000]
    P = {}
    for n in ("qwen", "bitnet", "falcon"):
        X = torch.stack([E[n][int(q2dd[n][q])] for _, q in sel])
        Y = torch.stack([host[L] for L, _ in sel])
        P[n] = fit_ridge(X, Y)
        print(f"  {n} map {tuple(P[n].shape)}", flush=True)
    # donor contribution per llama native id (projected to llama space)
    donor_sum = torch.zeros_like(host)
    donor_cnt = torch.zeros(nv, dtype=torch.int32)
    Ls = torch.tensor([L for L, _ in cand])
    for n in ("qwen", "bitnet", "falcon"):
        vals = torch.stack([E[n][int(q2dd[n][q])] for _, q in cand]) @ P[n]
        donor_sum[Ls] += vals
        donor_cnt[Ls] += 1
    donor_avg = donor_sum / donor_cnt.clamp_min(1).unsqueeze(1)
    lids = letter_ids(tok, nv)
    banks = {}
    for bn in ("fresh40", "final40"):
        mod = __import__(
            {"fresh40": "heldout_fresh40", "final40": "heldout_final40"}[bn]
        )
        banks[bn] = (mod.prompts(tok), mod.KEY)
    orig = host.clone()
    for alpha in (0.0, 0.01, 0.03, 0.05, 0.10, 0.20):
        blended = orig + alpha * (donor_avg - orig)
        blended[torch.nonzero(donor_cnt == 0).flatten()] = orig[
            torch.nonzero(donor_cnt == 0).flatten()
        ]
        with torch.no_grad():
            m.get_input_embeddings().weight.copy_(blended)
            if (
                m.get_output_embeddings() is not None
                and m.get_output_embeddings().weight.shape[0] == nv
            ):
                m.get_output_embeddings().weight.copy_(blended)
        line = [f"alpha={alpha:.2f}"]
        for bn, (seqs, key) in banks.items():
            acc = 0
            cl = []
            with torch.inference_mode():
                for s, g in zip(seqs, key):
                    lg = m(torch.tensor([s]))[0][0, -1, lids].float()
                    lp = F.log_softmax(lg, dim=-1)
                    acc += int(lp.argmax() == g)
                    cl.append(-lp[g].item())
            line.append(f"{bn}={acc}/{len(key)} CE={sum(cl) / len(cl):.4f}")
        print("  " + "  ".join(line), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
