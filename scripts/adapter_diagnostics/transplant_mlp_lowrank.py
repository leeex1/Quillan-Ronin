#!/usr/bin/env python3
"""Body transplant with low-rank pseudo-inverse (no 8192x8192 SVD).

P = M @ Y with M=(XtX+λI)^-1 Xt (dx,n), Y=(n,dy), n=#calib samples.
pinv(P) = pinv(Y) @ pinv(M), computed via small n×n Gram inversions.
Then similarity-transform donor MLP into host basis and alpha-sweep CE.
"""

from __future__ import annotations
import sys
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
HOST_PATH = r"E:\source_models\llama_1b"
DONOR_PATH = r"E:\source_models\qwen_0.5b"


def fit_factor(X, Y, ridge=1e-2):
    """Return (M,Y) with P=M@Y, and low-rank pinv(P)."""
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    G = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    M = torch.linalg.solve(G, X.T)  # (dx,n)
    # pinv(M) = (MtM)^-1 Mt   (M tall, full col rank)
    MtM = M.T @ M + lam * torch.eye(M.shape[1], dtype=M.dtype)
    pinv_M = torch.linalg.solve(MtM, M.T)  # (n,dx)
    Yn = Y @ Y.T + lam * torch.eye(Y.shape[0], dtype=Y.dtype)  # (n,n) small
    pinv_Y = Y.T @ torch.linalg.inv(Yn)  # (dy,n)
    pinv_P = pinv_Y @ pinv_M  # (dy,dx)
    P = M @ Y
    return P, pinv_P


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tokenizers import Tokenizer

    ht = Tokenizer.from_file(r"E:\source_models\llama_1b\tokenizer.json")
    dt = Tokenizer.from_file(r"E:\source_models\qwen_0.5b\tokenizer.json")
    hv, dv = ht.get_vocab(), dt.get_vocab()
    shared = sorted(set(hv) & set(dv))[:512]
    hids = torch.tensor([hv[s] for s in shared]).unsqueeze(0)
    dids = torch.tensor([dv[s] for s in shared]).unsqueeze(0)
    print(f"shared tokens={len(shared)}", flush=True)
    host = AutoModelForCausalLM.from_pretrained(
        HOST_PATH, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    donor = AutoModelForCausalLM.from_pretrained(
        DONOR_PATH, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    HL = len(host.model.layers)
    DL = len(donor.model.layers)
    nfit = int(0.7 * len(shared))
    print(f"host layers={HL} donor layers={DL} nfit={nfit}", flush=True)

    print("\n=== fit low-rank maps ===", flush=True)
    maps = {}
    with torch.inference_mode():
        for li in range(HL):
            dl = round(li * (DL - 1) / max(HL - 1, 1))
            cH = {}
            cD = {}
            a = host.model.layers[li].register_forward_hook(
                lambda m, i, o: cH.__setitem__("h", o[0] if isinstance(o, tuple) else o)
            )
            b = host.model.layers[li].mlp.gate_proj.register_forward_hook(
                lambda m, i, o: cH.__setitem__("g", o)
            )
            c = donor.model.layers[dl].register_forward_hook(
                lambda m, i, o: cD.__setitem__("h", o[0] if isinstance(o, tuple) else o)
            )
            d = donor.model.layers[dl].mlp.gate_proj.register_forward_hook(
                lambda m, i, o: cD.__setitem__("g", o)
            )
            host(hids)
            donor(dids)
            for h in (a, b, c, d):
                h.remove()
            Xr = cD["h"][0, :nfit, :].float()
            Yr = cH["h"][0, :nfit, :].float()
            Xf = cD["g"][0, :nfit, :].float()
            Yf = cH["g"][0, :nfit, :].float()
            Pr, Pir = fit_factor(Xr, Yr)
            Pf, Pif = fit_factor(Xf, Yf)
            cos = (
                F.cosine_similarity(
                    cD["h"][0, nfit:, :] @ Pr, cH["h"][0, nfit:, :], dim=-1
                )
                .mean()
                .item()
            )
            maps[li] = (Pr, Pir, Pf, Pif)
            if li in (0, 4, 8, 12, 15):
                print(
                    f"  L{li}<-{dl} res_cos={cos:.3f} rank(Pf)={torch.linalg.matrix_rank(Pf).item()}",
                    flush=True,
                )

    print("\n=== build transplanted weights ===", flush=True)
    from safetensors import safe_open

    src = Path(HOST_PATH) / "model.safetensors"
    tensors = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).float()
    dstate = donor.state_dict()
    trans = {}
    for li in range(HL):
        dl = round(li * (DL - 1) / max(HL - 1, 1))
        Pr, Pir, Pf, Pif = maps[li]
        for sub in ("gate_proj", "up_proj"):
            Wd = dstate[f"model.layers.{dl}.mlp.{sub}.weight"].float().T  # (d_d,ffn_d)
            trans[f"model.layers.{li}.mlp.{sub}.weight"] = (Pir @ Wd @ Pf).T
        Wd = dstate[f"model.layers.{dl}.mlp.down_proj.weight"].float().T  # (ffn_d,d_d)
        trans[f"model.layers.{li}.mlp.down_proj.weight"] = (Pif @ Wd @ Pr).T
    print(f"  transplanted {len(trans)} tensors", flush=True)

    tok = AutoTokenizer.from_pretrained(HOST_PATH, local_files_only=True)
    lids = [tok.encode(f" {L}", add_special_tokens=False)[0] for L in LETTERS]
    banks = {}
    for bn in ("fresh40", "final40"):
        mod = __import__(
            {"fresh40": "heldout_fresh40", "final40": "heldout_final40"}[bn]
        )
        banks[bn] = (mod.prompts(tok), mod.KEY)
    print("\n=== alpha sweep ===", flush=True)
    with torch.inference_mode():
        for alpha in (0.0, 0.05, 0.1, 0.25):
            sd = {k: v.clone() for k, v in tensors.items()}
            for k, Wt in trans.items():
                W0 = sd[k]
                scale = (W0.norm(dim=1) / Wt.norm(dim=1).clamp_min(1e-9)).unsqueeze(1)
                sd[k] = W0 + alpha * (Wt * scale - W0)
            host.load_state_dict(sd)
            line = [f"alpha={alpha:.2f}"]
            for bn, (seqs, key) in banks.items():
                acc = 0
                cl = []
                for s, g in zip(seqs, key):
                    lp = F.log_softmax(
                        host(torch.tensor([s]))[0][0, -1, lids].float(), dim=-1
                    )
                    acc += int(lp.argmax() == g)
                    cl.append(-lp[g].item())
                line.append(f"{bn}={acc}/{len(key)} CE={sum(cl) / len(cl):.4f}")
            print("  " + "  ".join(line), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
