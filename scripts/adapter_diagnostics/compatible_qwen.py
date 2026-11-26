#!/usr/bin/env python3
"""Best-aligned pair (Qwen->Llama) with FULL-RANK maps and enough samples.

Qwen hidden 896->Llama 2048, FFN 4864->8192. Use 6400 token positions so
rank(P_ffn) can reach 4864 (full). Full similarity transform of MLP:
  gate/up: W_gh = P_ffn.T @ W_gd @ Pinv_res.T
  down:    W_dh = P_res.T @ W_dd @ Pinv_ffn.T
One model at a time, activations cached. Alpha sweep on held-out banks.
"""

from __future__ import annotations
import sys, gc
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
CACHE = Path(r"F:\quillan_cache\compat_qwen")
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(2)
LETTERS = ["A", "B", "C", "D"]
HOST_PATH = r"E:\source_models\llama_1b"
DONOR_PATH = r"E:\source_models\qwen_0.5b"
SEQ_LEN, NSEQ = 32, 200  # 6400 positions


def ridge_full(X, Y, ridge=1e-3):
    """P: X@P ~= Y, plus low-rank pseudo-inverse via small Gram."""
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    G = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    P = torch.linalg.solve(G, X.T @ Y)  # (dx,dy)
    # pinv(P) for wide P (dx<dy): P has row rank dx
    PPt = P @ P.T + lam * torch.eye(P.shape[0], dtype=P.dtype)
    pinv = P.T @ torch.linalg.inv(PPt)  # (dy,dx)
    return P, pinv


def capture(model, seqs, li, which="h"):
    cap = {}
    mod = (
        model.model.layers[li] if which == "h" else model.model.layers[li].mlp.gate_proj
    )
    h = mod.register_forward_hook(
        lambda m, i, o: cap.__setitem__("v", o[0] if isinstance(o, tuple) else o)
    )
    out = []
    with torch.inference_mode():
        for s in seqs:
            model(torch.tensor([s]))
            out.append(cap["v"][0].detach())
    h.remove()
    return torch.cat(out, dim=0).float()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import Tokenizer
    from safetensors.torch import load_file

    CACHE.mkdir(exist_ok=True)
    ht = Tokenizer.from_file(r"E:\source_models\llama_1b\tokenizer.json")
    dt = Tokenizer.from_file(r"E:\source_models\qwen_0.5b\tokenizer.json")
    hv, dv = ht.get_vocab(), dt.get_vocab()
    shared = sorted(set(hv) & set(dv))
    seqs = [
        [shared[(i * SEQ_LEN + j) % len(shared)] for j in range(SEQ_LEN)]
        for i in range(NSEQ)
    ]
    hseq = [[hv[s] for s in q] for q in seqs]
    dseq = [[dv[s] for s in q] for q in seqs]
    print(
        f"calibration {NSEQ}x{SEQ_LEN}={NSEQ * SEQ_LEN}, shared={len(shared)}",
        flush=True,
    )

    cached = all(
        (CACHE / f"{w}_{i}.pt").exists()
        for w in ("host", "donor", "hostf", "donorf")
        for i in range(16)
    )
    HL, DL = 16, 24
    if not cached:
        print("[1] host activations...", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            HOST_PATH,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        HL = len(m.model.layers)
        for li in range(HL):
            torch.save(
                capture(m, hseq, li, "h").to(torch.bfloat16), CACHE / f"host_{li}.pt"
            )
            torch.save(
                capture(m, hseq, li, "g").to(torch.bfloat16), CACHE / f"hostf_{li}.pt"
            )
        del m
        gc.collect()
        print("[2] donor activations...", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            DONOR_PATH,
            local_files_only=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).eval()
        DL = len(m.model.layers)
        for li in range(HL):
            dl = round(li * (DL - 1) / (HL - 1))
            torch.save(
                capture(m, dseq, dl, "h").to(torch.bfloat16), CACHE / f"donor_{li}.pt"
            )
            torch.save(
                capture(m, dseq, dl, "g").to(torch.bfloat16), CACHE / f"donorf_{li}.pt"
            )
        del m
        gc.collect()
        print(f"  cached host={HL} donor={DL}", flush=True)
    else:
        print("using cached activations", flush=True)

    print("[3] fit full-rank maps + transform...", flush=True)
    dsd = load_file(r"E:\source_models\qwen_0.5b\model.safetensors")
    trans = {}
    for li in range(HL):
        dl = round(li * (DL - 1) / (HL - 1))
        Xd = torch.load(CACHE / f"donor_{li}.pt", weights_only=True).float()
        Xh = torch.load(CACHE / f"host_{li}.pt", weights_only=True).float()
        Df = torch.load(CACHE / f"donorf_{li}.pt", weights_only=True).float()
        Hf = torch.load(CACHE / f"hostf_{li}.pt", weights_only=True).float()
        Pr, Pir = ridge_full(Xd, Xh)
        Pf, Pif = ridge_full(Df, Hf)
        cos = F.cosine_similarity(Xd @ Pr, Xh, dim=-1).mean().item()
        for sub in ("gate_proj", "up_proj"):
            Wd = dsd[f"model.layers.{dl}.mlp.{sub}.weight"].float()
            trans[f"model.layers.{li}.mlp.{sub}.weight"] = Pf.T @ Wd @ Pir.T
        Wd = dsd[f"model.layers.{dl}.mlp.down_proj.weight"].float()
        trans[f"model.layers.{li}.mlp.down_proj.weight"] = Pr.T @ Wd @ Pif.T
        if li in (0, 4, 8, 12, 15):
            print(
                f"  L{li}<-{dl} cos={cos:.3f} rank_Pr={torch.linalg.matrix_rank(Pr).item()} rank_Pf={torch.linalg.matrix_rank(Pf).item()}",
                flush=True,
            )
    del dsd
    gc.collect()
    print(f"  {len(trans)} tensors", flush=True)

    print("[4] alpha sweep...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        HOST_PATH, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    tok = AutoTokenizer.from_pretrained(HOST_PATH, local_files_only=True)
    lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
    banks = {}
    for bn in ("fresh40", "final40"):
        mod = __import__(
            {"fresh40": "heldout_fresh40", "final40": "heldout_final40"}[bn]
        )
        banks[bn] = (mod.prompts(tok), mod.KEY)
    params = dict(m.named_parameters())
    base = {k: params[k].data.clone() for k in trans}
    with torch.inference_mode():
        for alpha in (0.0, 0.02, 0.05, 0.1, 0.25):
            for k, Wt in trans.items():
                params[k].data.copy_(base[k] + alpha * (Wt - base[k]))
            line = [f"alpha={alpha:.2f}"]
            for bn, (seqs_b, key) in banks.items():
                acc = 0
                cl = []
                for s, g in zip(seqs_b, key):
                    lp = F.log_softmax(
                        m(torch.tensor([s]))[0][0, -1, lids].float(), dim=-1
                    )
                    acc += int(lp.argmax() == g)
                    cl.append(-lp[g].item())
                line.append(f"{bn}={acc}/{len(key)} CE={sum(cl) / len(cl):.4f}")
            print("  " + "  ".join(line), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
