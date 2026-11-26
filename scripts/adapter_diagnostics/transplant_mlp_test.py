#!/usr/bin/env python3
"""CHANGE-OF-BASIS BODY TRANSPLANT (closed-form, no training).

Fit per-layer maps:
  P_res[l]: donor residual -> host residual
  P_ffn[l]: donor FFN intermediate -> host FFN intermediate
Then map donor MLP weights into host basis via similarity transform:
  gate/up:  W_host = (pinv(P_res) @ W_donor.T @ P_ffn).T
  down:     W_host = (pinv(P_ffn) @ W_donor.T @ P_res).T
Blend into host MLP at alpha; measure CE. No gradient, no training.
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
HOST_PATH = r"E:\source_models\llama_1b"
DONOR_PATH = r"E:\source_models\qwen_0.5b"


def fit_ridge(X, Y, ridge=1e-2):
    lam = ridge * X.abs().mean().clamp_min(1e-9)
    return torch.linalg.solve(
        X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y
    )


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # shared raw token strings -> single-token prompts in each vocab
    from tokenizers import Tokenizer

    ht = Tokenizer.from_file(r"E:\source_models\llama_1b\tokenizer.json")
    dt = Tokenizer.from_file(r"E:\source_models\qwen_0.5b\tokenizer.json")
    hv, dv = ht.get_vocab(), dt.get_vocab()
    shared = sorted(set(hv) & set(dv))[:256]
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
    print(f"host layers={HL} donor layers={DL}", flush=True)

    nfit = int(0.7 * len(shared))
    P_res = {}
    P_ffn = {}
    print("\n=== capture + fit maps ===", flush=True)
    with torch.inference_mode():
        for li in range(HL):
            dl = round(li * (DL - 1) / max(HL - 1, 1))
            capH = {}
            capD = {}
            hh = host.model.layers[li].register_forward_hook(
                lambda m, i, o, k="h": capH.__setitem__(
                    k, o[0] if isinstance(o, tuple) else o
                )
            )
            hg = host.model.layers[li].mlp.gate_proj.register_forward_hook(
                lambda m, i, o, k="g": capH.__setitem__(k, o)
            )
            dh = donor.model.layers[dl].register_forward_hook(
                lambda m, i, o, k="h": capD.__setitem__(
                    k, o[0] if isinstance(o, tuple) else o
                )
            )
            dg = donor.model.layers[dl].mlp.gate_proj.register_forward_hook(
                lambda m, i, o, k="g": capD.__setitem__(k, o)
            )
            host(hids)
            donor(dids)
            for h in (hh, hg, dh, dg):
                h.remove()
            Hr = capH["h"][0, :nfit, :].float()
            Dr = capD["h"][0, :nfit, :].float()
            Hf = capH["g"][0, :nfit, :].float()
            Df = capD["g"][0, :nfit, :].float()
            P_res[li] = fit_ridge(Dr, Hr)
            P_ffn[li] = fit_ridge(Df, Hf)
            # held-out cos
            Hrh = capH["h"][0, nfit:, :].float()
            Drh = capD["h"][0, nfit:, :].float()
            cos = F.cosine_similarity(Drh @ P_res[li], Hrh, dim=-1).mean().item()
            if li in (0, 4, 8, 12, 15) or li == HL - 1:
                print(
                    f"  layer {li:2d}<-{dl:2d} res_cos={cos:.3f} P_res={tuple(P_res[li].shape)} P_ffn={tuple(P_ffn[li].shape)}",
                    flush=True,
                )

    print("\n=== build transplanted host weights ===", flush=True)
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(HOST_PATH) / "model.safetensors"
    tensors = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).float()
    dstate = donor.state_dict()
    transplanted = {}
    for li in range(HL):
        dl = round(li * (DL - 1) / max(HL - 1, 1))
        pre = f"model.layers.{li}.mlp."
        dpre = f"model.layers.{dl}.mlp."
        Pin = P_res[li]
        Pout = P_ffn[li]
        Pinv = torch.linalg.pinv(Pin)
        Pfinv = torch.linalg.pinv(Pout)
        for sub in ("gate_proj", "up_proj"):
            Wd = dstate[f"{dpre}{sub}.weight"].float()  # (ffn_d, d_d)
            Wh_host = (Pinv @ Wd.T @ Pout).T  # (ffn_h, d_h)
            transplanted[f"{pre}{sub}.weight"] = Wh_host
        Wd = dstate[f"{dpre}down_proj.weight"].float()  # (d_d, ffn_d)
        Wh_host = (Pfinv @ Wd.T @ Pin).T  # (d_h, ffn_h)
        transplanted[f"{pre}down_proj.weight"] = Wh_host
    print(f"  transplanted {len(transplanted)} MLP tensors", flush=True)

    # alpha sweep in memory
    tok = AutoTokenizer.from_pretrained(HOST_PATH, local_files_only=True)
    nv = tensors["lm_head.weight"].shape[0]
    lids = []
    for L in LETTERS:
        e = tok.encode(f" {L}", add_special_tokens=False)
        lids.append(e[0] if len(e) == 1 else tok.encode(L, add_special_tokens=False)[0])
    banks = {}
    for bn in ("fresh40", "final40"):
        mod = __import__(
            {"fresh40": "heldout_fresh40", "final40": "heldout_final40"}[bn]
        )
        banks[bn] = (mod.prompts(tok), mod.KEY)

    def load_to_model(alpha):
        sd = {k: v.clone() for k, v in tensors.items()}
        for k, Wt in transplanted.items():
            W0 = sd[k]
            # row-norm match to avoid scale blowup
            scale = (W0.norm(dim=1) / Wt.norm(dim=1).clamp_min(1e-9)).unsqueeze(1)
            sd[k] = W0 + alpha * (Wt * scale - W0)
        host.load_state_dict(sd)

    print("\n=== alpha sweep (Qwen MLP -> Llama body) ===", flush=True)
    with torch.inference_mode():
        for alpha in (0.0, 0.1, 0.25, 0.5, 1.0):
            load_to_model(alpha)
            line = [f"alpha={alpha:.2f}"]
            for bn, (seqs, key) in banks.items():
                acc = 0
                cl = []
                for s, g in zip(seqs, key):
                    lg = host(torch.tensor([s]))[0][0, -1, lids].float()
                    lp = F.log_softmax(lg, dim=-1)
                    acc += int(lp.argmax() == g)
                    cl.append(-lp[g].item())
                line.append(f"{bn}={acc}/{len(key)} CE={sum(cl) / len(cl):.4f}")
            print("  " + "  ".join(line), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
