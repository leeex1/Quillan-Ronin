#!/usr/bin/env python3
"""DECISIVE closed-form 4-donor consensus graft into Quillan flagship LoRA.

All four donors healthy (BitNet unpack fixed). Align each donor's final hidden
into the student (Llama 2048) frame via ridge on shared-token embeddings,
scale-normalize, weight by cosine agreement, form consensus target, and write
the (consensus - student) delta into the student's last down_proj LoRA via
closed-form ridge. Sham control: random target of same norm. No training.
"""

from __future__ import annotations
import sys, json
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
OUT = ROOT / "exports"
CACHE = OUT / "donor_hidden_cache"
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model
from tokenizers import Tokenizer

torch.set_num_threads(2)
STUDENT = OUT / "quillan-flagship-hifi-g4-exact.safetensors"
TP = {
    "qwen": r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\Lib\site-packages\x",  # overwritten below
}
# tokenizer paths
TOK = {
    "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json",
    "llama": r"E:\source_models\llama_1b\tokenizer.json",
    "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\tokenizer.json",
    "bitnet": r"E:\source_models\llama_1b\tokenizer.json",  # bitnet uses llama tok loader
}
EMB = {
    "qwen": r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\model.safetensors",
    "falcon": r"C:\Users\Admin\AppData\Local\Temp\opencode\falcon32\model.safetensors",
    "bitnet": r"E:\source_models\bitnet_2b_4t\model.safetensors",
}


def main():
    from safetensors import safe_open
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--sham", action="store_true")
    a = ap.parse_args()

    H = {
        n: torch.load(CACHE / f"{n}_hidden.pt", weights_only=True).float()
        for n in ("qwen", "llama", "falcon", "bitnet")
    }
    N = H["llama"].shape[0]
    Hl = H["llama"][:N]
    print(
        f"hidden shapes: " + ", ".join(f"{n}={tuple(H[n].shape)}" for n in H),
        flush=True,
    )
    toks = {k: Tokenizer.from_file(v).get_vocab() for k, v in TOK.items()}
    shared = sorted(set(toks["qwen"]) & set(toks["llama"]) & set(toks["falcon"]))
    print(f"shared tokens (qwen/llama/falcon)={len(shared)}", flush=True)
    with safe_open(
        r"E:\source_models\llama_1b\model.safetensors", framework="pt", device="cpu"
    ) as f:
        hostE = f.get_tensor("model.embed_tokens.weight").float()
    hi = torch.tensor([toks["llama"][s] for s in shared])

    def align(name):
        with safe_open(EMB[name], framework="pt", device="cpu") as f:
            key = "model.embed_tokens.weight"
            dE = f.get_tensor(key).float()
        di = torch.tensor([toks[name][s] for s in shared])
        G = dE[di].T @ dE[di] + 1e-2 * torch.eye(dE.shape[1]) * dE.abs().mean()
        P = torch.linalg.solve(G, dE[di].T @ hostE[hi])  # (d_donor,2048)
        Ha = H[name][:N] @ P
        Ha = Ha * (Hl.norm(dim=1).mean() / Ha.norm(dim=1).mean())
        return Ha, P

    aligned = {"llama": Hl}
    for n in ("qwen", "falcon", "bitnet"):
        Ha, _ = align(n)
        aligned[n] = Ha
        cos = torch.nn.functional.cosine_similarity(Ha, Hl, dim=-1).mean().item()
        print(f"  {n} aligned cos to llama={cos:.3f}", flush=True)

    def w(n):
        return max(
            0.0,
            torch.nn.functional.cosine_similarity(aligned[n], Hl, dim=-1).mean().item(),
        )

    ws = {n: w(n) for n in aligned if n != "llama"}
    weight_sum = sum(ws.values())
    target = Hl.clone() * 1.0
    for n, ww in ws.items():
        target += aligned[n] * ww
    target = target / (1.0 + weight_sum)
    print("weights:", {n: round(v, 3) for n, v in ws.items()}, flush=True)

    # student forward: capture last down_proj input (key) and current output
    m = load_model(STUDENT)
    dp = m.layers[-1].down_proj
    cap = {}
    hk = dp.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    hk2 = dp.register_forward_hook(
        lambda mod, inp, o: cap.__setitem__("out", o.detach().float())
    )
    from heldout_final40 import prompts as pf

    class Shim:
        def __init__(self, t):
            self.t = t

        def encode(self, s, add_special_tokens=True):
            return self.t.encode(s, add_special_tokens=add_special_tokens).ids

    tok = Tokenizer.from_file(TOK["llama"])
    seqs = pf(Shim(tok))[:N]
    Ks = []
    cur = []
    with torch.inference_mode():
        for s in seqs:
            m(torch.tensor([s]))
            Ks.append(cap["k"][0, -1, :].clone())
            cur.append(cap["out"][0, -1, :].clone())
    hk.remove()
    hk2.remove()
    K = torch.stack(Ks)
    C = torch.stack(cur)
    print(
        f"K={tuple(K.shape)} cur={tuple(C.shape)} target={tuple(target.shape)}",
        flush=True,
    )

    if a.sham:
        g = torch.Generator().manual_seed(20261125)
        target = (
            torch.randn_like(target)
            * target.norm(dim=1, keepdim=True)
            / target.shape[1] ** 0.5
        )
        print("SHAM target (random matched norm)", flush=True)
    E = (target - C) * a.strength
    Gk = K @ K.T + 1e-2 * torch.eye(N) * K.norm().item()
    Delta = E.T @ torch.linalg.solve(Gk, K)
    print(f"Delta norm={Delta.norm().item():.3f}", flush=True)
    r = min(a.rank, dp.lora_A.shape[0], *Delta.shape)
    U, S, Vh = torch.linalg.svd(Delta, full_matrices=False)
    A = Vh[:r].contiguous()
    B = (U[:, :r] * S[:r]).contiguous()
    with torch.no_grad():
        # zero existing then write; repack cache
        dp.lora_A.zero_()
        dp.lora_B.zero_()
        dp.lora_A[:r].copy_(A.to(dp.lora_A.dtype))
        dp.lora_B[:, :r].copy_(B.to(dp.lora_B.dtype))
        if hasattr(dp, "_wc"):
            dp._wc = None
    m.save_packed_safetensors(OUT / a.out)
    print(f"SAVED {OUT / a.out} rank={r}", flush=True)


if __name__ == "__main__":
    main()
