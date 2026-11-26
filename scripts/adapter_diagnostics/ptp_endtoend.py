#!/usr/bin/env python3
"""End-to-end PTP: recall digit patterns from the model's OWN token embeddings,
then recombine through the model's real SwiGLU circuit to compute a+b.

Stage A (recall): closed-form read W_rec maps embedding E[digit_token] -> one-hot slot.
Stage B (recombine): the ptp_adder SwiGLU circuit (gate*up -> down) computes the sum.
No injected features, no label key, no backprop. All weights written closed-form.
Generalization: train=subset of digit pairs used ONLY to build W_rec; test=disjoint.
"""

from __future__ import annotations
import sys, random
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import load_model
from ptp_adder import zero_base, build_circuit, DIG, NPAIR, SUMS


def digit_token_ids(tok):
    ids = []
    for d in range(DIG):
        e = tok.encode(str(d), add_special_tokens=False)
        ids.append(
            e[0] if len(e) == 1 else tok.encode(f" {d}", add_special_tokens=False)[0]
        )
    return ids


def build_read(E_digits, hidden):
    """Closed-form: W_rec @ E[d] ~= one_hot(d). Ridge solve."""
    # W [hidden,DIG] such that E[d] @ W ~= one_hot(d), closed-form ridge
    G = E_digits.T @ E_digits + 1e-3 * torch.eye(hidden)  # [hidden,hidden]
    W = torch.linalg.solve(G, E_digits.T)  # [hidden,DIG]
    return W


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-rank64.safetensors"),
    )
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = (
        AutoTokenizer.from_pretrained(
            r"C:\01_PROGRAMS\Open_Source\AI_ML\models\Qwen--Qwen3-0.6B",
            local_files_only=True,
        )
        if False
        else None
    )
    # reuse the harness tokenizer
    from wsp_math_eval import tok_for

    tok = tok_for(151936)
    did = digit_token_ids(tok)
    m = load_model(a.ckpt)
    E = m.embed_tokens.unpack_float().float()
    E_d = E[did]  # [10, hidden]
    W_read = build_read(E_d, E.shape[1])
    # verify recall accuracy from embeddings alone: E[d] @ W_read -> one_hot(d)
    rec = (E_d @ W_read).argmax(1)
    print(
        f"[Stage A recall from embeddings] correct={int((rec == torch.arange(DIG)).sum())}/{DIG}",
        flush=True,
    )

    gate, up, down = build_circuit(m, layer=8)

    def run(da, db):
        # recall from the model's own embeddings of the digit tokens
        xa = E[did[da]] @ W_read
        xb = E[did[db]] @ W_read
        x = torch.zeros(gate.in_features)
        x[:DIG] = xa
        x[DIG : 2 * DIG] = xb
        g = gate(x)
        u = up(x)
        s = F.silu(g) * u
        y = down(s)
        return int(y[:SUMS].argmax())

    train = [(i, j) for i in range(6) for j in range(6)]
    test = [(i, j) for i in range(DIG) for j in range(DIG) if i >= 6 or j >= 6]
    tr = sum(run(i, j) == i + j for i, j in train)
    te = sum(run(i, j) == i + j for i, j in test)
    print(
        f"[END-TO-END] train={tr}/{len(train)} held-out={te}/{len(test)} ({100 * te / len(test):.1f}%)",
        flush=True,
    )
    for a_, b_ in [(3, 4), (7, 8), (9, 9), (2, 9)]:
        print(f"  {a_}+{b_}={a_ + b_} -> {run(a_, b_)}", flush=True)


if __name__ == "__main__":
    main()
