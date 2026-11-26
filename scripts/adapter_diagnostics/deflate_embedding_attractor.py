#!/usr/bin/env python3
"""Deflate the repetition attractor in the TIED EMBEDDING (the actual logit carrier).

logits = E . RMSNorm(h). Repetition loop = a dominant direction in E that keeps
winning argmax. Deflate it in the embedding output rows (closed-form), measure
rep5gram + general knowledge. This is the 'fix the broken pattern in the tensor
that carries it' approach the user described.
"""

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


def attractor(m):
    probes = [
        "The meaning of life is",
        "In conclusion, the main point is",
        "Explain the water cycle:",
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
    return V[0], float(S[0]), float(S[1])


def rep(m):
    ids = tok.encode("The meaning of life is", add_special_tokens=False)
    with torch.inference_mode():
        out = m.generate(torch.tensor([ids]), max_new_tokens=80, temperature=0.0)
    g = out[0, len(ids) :].tolist()
    if len(g) < 5:
        return 0.0, g
    gr = [tuple(g[i : i + 5]) for i in range(len(g) - 4)]
    return 1 - len(set(gr)) / len(gr), g


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


def main():
    m = load_model(CK)
    enable_cache(m)
    v, s0, s1 = attractor(m)
    br, g0 = rep(m)
    bg = gk(m)
    print(
        f"[BASE] SVD {s0:.0f} vs {s1:.0f} | rep={br:.3f} gk={bg}/{len(GQ)}", flush=True
    )
    print(
        f"[BASE gen] {tok.decode(g0[:40], skip_special_tokens=True)[:120]!r}",
        flush=True,
    )

    # Deflate the attractor direction from the embedding output (tied head)
    E = m.embed_tokens.unpack_float().float()  # [vocab, hidden]
    # E_def = E - alpha * (E @ v) outer v  -> removes v component from all rows
    for alpha in (0.1, 0.3, 0.5):
        m = load_model(CK)
        enable_cache(m)
        E = m.embed_tokens.unpack_float().float()
        proj = (E @ v) / (v @ v + 1e-8)  # [vocab]
        Edef = E - alpha * proj.unsqueeze(1) @ v.unsqueeze(0)
        m.embed_tokens.set_weights_from_float(Edef)
        r, g = rep(m)
        gkv = gk(m)
        print(
            f"[alpha={alpha}] rep={r:.3f} gk={gkv}/{len(GQ)} | {tok.decode(g[:30], skip_special_tokens=True)[:80]!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()
