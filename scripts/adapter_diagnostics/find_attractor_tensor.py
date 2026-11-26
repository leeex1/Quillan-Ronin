#!/usr/bin/env python3
"""Test deflation of the repetition attractor in EACH candidate tensor; measure rep + GK."""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache

CK = ROOT / "exports" / "quillan-mini-math40-rank64.safetensors"
tok = tok_for(151936)
lids = letter_ids(tok, 151936)


def build():
    m = load_model(CK)
    enable_cache(m)
    return m


def attractor(m):
    probes = [
        "The meaning of life is",
        "In conclusion, the main point is",
        "Explain the water cycle:",
        "Question: What is 2 + 2?\nA) 3\nB) 4\nC) 5\nD) 6\nAnswer:",
        "The capital of France is",
        "Water freezes at",
        "Hello, my name is",
    ]
    cap = {}
    hs = []
    for p in probes:
        ids = tok.encode(p, add_special_tokens=False)
        hk = m.norm.register_forward_pre_hook(
            lambda mod, inp: cap.__setitem__("h", inp[0].detach().float())
        )
        with torch.inference_mode():
            m(torch.tensor([ids]))
        hk.remove()
        hs.append(cap["h"][0, -1])
    H = torch.stack(hs)
    Hc = H - H.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(Hc, full_matrices=False)
    print(
        f"  SVD top: {[round(float(S[i]), 1) for i in range(min(5, len(S)))]}",
        flush=True,
    )
    return V[0]  # [hidden]


def rep(m):
    ids = tok.encode("The meaning of life is", add_special_tokens=False)
    with torch.inference_mode():
        out = m.generate(torch.tensor([ids]), max_new_tokens=60, temperature=0.0)
    g = out[0, len(ids) :].tolist()
    if len(g) < 5:
        return 0.0
    gr = [tuple(g[i : i + 5]) for i in range(len(g) - 4)]
    return 1 - len(set(gr)) / len(gr)


def gk(m):
    ok = 0
    for i in range(len(GQ)):
        ids = tok.encode(
            f"Question {i + 1}: {GQ[i][0]}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, GQ[i][1]))
            + "Answer:",
            add_special_tokens=False,
        )
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids]
        ok += int(int(lg.argmax()) == GK[i])
    return ok


def deflate_B(B, v, alpha):
    """B is [out, rank], v is [out]. Remove output component along v."""
    proj = (v @ B) / (v @ v + 1e-8)  # [rank]
    return B - alpha * v.unsqueeze(1) @ proj.unsqueeze(0)


def main():
    m = build()
    v = attractor(m)
    print(f"[BASE] rep={rep(m):.3f} gk={gk(m)}/{len(GQ)}", flush=True)
    del m

    for label, layers, alpha in [
        ("last down_proj.lora_B", [23], 0.5),
        ("last o_proj.lora_B", [23], 0.5),
        ("upper 4 down_proj.lora_B", [20, 21, 22, 23], 0.3),
        ("all down_proj.lora_B", list(range(24)), 0.3),
    ]:
        print(f"[candidate: {label}]", flush=True)
        m = build()
        for li in layers:
            d = m.layers[li].down_proj if "down" in label else m.layers[li].o_proj
            B = d.lora_B.detach().clone().float()
            with torch.no_grad():
                d.lora_B.copy_(deflate_B(B, v, alpha))
        print(f"  rep={rep(m):.3f} gk={gk(m)}/{len(GQ)}", flush=True)
        del m

    print("[candidate: final norm weight strong]", flush=True)
    m = build()
    w = m.norm.weight.detach().clone().float()
    with torch.no_grad():
        m.norm.weight.copy_((w - 0.8 * (w * v).sum() * v).to(m.norm.weight.dtype))
    print(f"  rep={rep(m):.3f} gk={gk(m)}/{len(GQ)}", flush=True)


if __name__ == "__main__":
    main()
