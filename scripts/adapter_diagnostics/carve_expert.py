#!/usr/bin/env python3
"""Carve a clean MoE expert in a LIVE checkpoint, install the closed-form PTP
circuit there, force routing to it, open the gate, and measure:
  - GK preservation (in-memory and fresh reload)
  - fixed-set retention
  - circuit correctness on the carved expert
  - repetition

If GK survives, the circuit coexists with knowledge. If not, report honestly.
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import quantize_group_ternary_ls, pack_ternary_2bit
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from hard_math20 import PROBLEMS as HP, KEY as HK, LETTERS as HL, format_problem as hfmt
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from ptp_adder import DIG, SUMS


def write_packed(buf_q, buf_g, expert, W, group_size):
    t, g = quantize_group_ternary_ls(W, group_size=group_size)
    packed, _ = pack_ternary_2bit(t)
    buf_q[expert].copy_(packed)
    buf_g[expert].copy_(g)


def gk(m, tok, lids):
    ok = 0
    for i, (q, opts, _) in enumerate(GQ):
        p = (
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:"
        )
        ids = tok.encode(p, add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids]
        ok += int(int(lg.argmax()) == GK[i])
    return ok


def fixed(m, tok, lids):
    ok = 0
    for i in range(len(HP)):
        ids = tok.encode(hfmt(i), add_special_tokens=False)
        with torch.inference_mode():
            lg = m(torch.tensor([ids]))[0][0, -1, lids]
        ok += int(int(lg.argmax()) == HK[i])
    return ok


def carve(m, layer=8, expert=0):
    moe = m.layers[layer].council_moe
    H = moe.hidden_size
    F = moe.moe_ffn_dim
    gs = moe.group_size
    # circuit weights (SwiGLU): gate wg[a-pair]=gain iff digit a==i; w1[b-pair] iff b==j; w2[sum,pair]
    wg = torch.zeros(F, H)
    w1 = torch.zeros(F, H)
    w2 = torch.zeros(H, F)
    gain = 2.0
    for i in range(DIG):
        for j in range(DIG):
            p = i * DIG + j
            wg[p, i] = gain
            w1[p, DIG + j] = 1.0
            w2[i + j, p] = 1.0
    # zero expert's existing base then write circuit via ternary packing
    write_packed(moe.wgate_q, moe.wgate_gamma, expert, wg, gs)
    write_packed(moe.w1_q, moe.w1_gamma, expert, w1, gs)
    write_packed(moe.w2_q, moe.w2_gamma, expert, w2, gs)
    # zero the swarm fp16 residual on this expert
    with torch.no_grad():
        moe.swarm_a[expert].zero_()
        moe.swarm_b[expert].zero_()
    # verify unpacked weights match
    wg_u = moe._unpack_expert_mat(moe.wgate_q, moe.wgate_gamma, expert, F, H)
    w1_u = moe._unpack_expert_mat(moe.w1_q, moe.w1_gamma, expert, F, H)
    w2_u = moe._unpack_expert_mat(moe.w2_q, moe.w2_gamma, expert, H, F)
    err = max(
        float((wg_u - wg).abs().max()),
        float((w1_u - w1).abs().max()),
        float((w2_u - w2).abs().max()),
    )
    print(f"[carve L{layer} E{expert}] pack_err={err:.4f} gs={gs}", flush=True)
    # force routing to this expert and open gate
    with torch.no_grad():
        moe.top_k = 1
        moe.router.weight.zero_()
        moe.router.weight[expert, :] = (
            1.0  # logit for expert e = sum(x) -> highest if x positive
        )
        moe.gate_alpha.fill_(2.0)  # tanh(2)=0.964 -> gate open
    return moe


def circuit_check(moe, layer, expert):
    H = moe.hidden_size

    def out(a, b):
        x = torch.zeros(1, 1, H)
        x[0, 0, a] = 1.0
        x[0, 0, DIG + b] = 1.0
        with torch.inference_mode():
            y = moe(x)
        return int(y[0, 0, :SUMS].argmax())

    tr = sum(out(i, j) == i + j for i in range(6) for j in range(6))
    te = sum(
        out(i, j) == i + j for i in range(DIG) for j in range(DIG) if i >= 6 or j >= 6
    )
    print(f"[circuit in MoE] train={tr}/36 held-out={te}/64", flush=True)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-rank64.safetensors"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-carved-circuit.safetensors"),
    )
    a = ap.parse_args()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(a.ckpt)
    enable_cache(m)
    print(
        f"[BASE] GK={gk(m, tok, lids)}/25 fixed={fixed(m, tok, lids)}/{len(HP)}",
        flush=True,
    )
    moe = carve(m, layer=8, expert=0)
    circuit_check(moe, 8, 0)
    print(
        f"[AFTER carve, in-memory] GK={gk(m, tok, lids)}/25 fixed={fixed(m, tok, lids)}/{len(HP)}",
        flush=True,
    )
    m.save_packed_safetensors(a.out)
    print(f"[saved] {a.out}", flush=True)
    # fresh reload
    m2 = load_model(a.out)
    enable_cache(m2)
    print(
        f"[FRESH RELOAD] GK={gk(m2, tok, lids)}/25 fixed={fixed(m2, tok, lids)}/{len(HP)}",
        flush=True,
    )
    circuit_check(m2.layers[8].council_moe, 8, 0)


if __name__ == "__main__":
    main()
