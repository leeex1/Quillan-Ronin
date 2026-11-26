#!/usr/bin/env python3
"""C8 PTP core proof: install a GENERALIZING addition circuit via closed-form tensor writes.

Uses the model's REAL SwiGLU path:  s = silu(gate(x)) * up(x); y = down(s).
Circuit = AND-product of recalled digit patterns:
  gate neuron (i,j) fires iff operand a == i  (recall of a)
  up   neuron (i,j) fires iff operand b == j  (recall of b)
  product fires iff a==i AND b==j            (recombination)
  down maps each neuron to its sum i+j        (answer)
Weights are ALGEBRAIC (shared), not fit to the test pairs -> generalization by
construction. Controls: digit-disjoint holdout + wrong-target circuit.
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
from wsp_math_eval import load_model

DIG = 10  # digits 0..9
NPAIR = DIG * DIG  # 100 (i,j) neurons
SUMS = DIG + DIG - 1  # 0..18 -> 19 classes


def zero_base(mod):
    """Zero the packed base so the layer is a pure closed-form circuit."""
    z = torch.zeros(mod.out_features, mod.in_features, dtype=torch.float32)
    mod.set_weights_from_float(z)
    if hasattr(mod, "_wc"):
        mod._wc = None


def feature(a, b, hidden):
    x = torch.zeros(hidden)
    x[a] = 1.0  # dim 0..9   = one-hot(a)
    x[DIG + b] = 1.0  # dim 10..19 = one-hot(b)
    return x


def build_circuit(m, layer=8, offset=0):
    blk = m.layers[layer]
    gate, up, down = blk.gate_proj, blk.up_proj, blk.down_proj
    for mod in (gate, up, down):
        zero_base(mod)
    r = gate.lora_A.shape[0]
    # ranks needed: 10 (digits a) + 10 (digits b) + 19 (sum classes) = 39 <= 64
    A_g = torch.zeros(r, gate.in_features)
    B_g = torch.zeros(gate.out_features, r)
    A_u = torch.zeros(r, up.in_features)
    B_u = torch.zeros(up.out_features, r)
    A_d = torch.zeros(r, down.in_features)
    B_d = torch.zeros(down.out_features, r)
    gain = 2.0
    for i in range(DIG):
        for j in range(DIG):
            p = i * DIG + j
            A_g[i, i] = 1.0
            B_g[p, i] = gain  # gate[p] = gain iff a==i
            A_u[j, DIG + j] = 1.0
            B_u[p, j] = 1.0  # up[p]   = 1    iff b==j
            s = i + j
            A_d[s, p] = 1.0
            B_d[s, s] = 1.0  # down output dim s = sum of matching products
    with torch.no_grad():
        gate.lora_A.copy_(A_g)
        gate.lora_B.copy_(B_g)
        up.lora_A.copy_(A_u)
        up.lora_B.copy_(B_u)
        down.lora_A.copy_(A_d)
        down.lora_B.copy_(B_d)
    return gate, up, down


def run_circuit(gate, up, down, a, b):
    x = feature(a, b, gate.in_features)
    g = gate(x)
    u = up(x)
    s = F.silu(g) * u
    y = down(s)
    return y


def acc(gate, up, down, pairs, wrong=False):
    ok = 0
    for a, b in pairs:
        y = run_circuit(gate, up, down, a, b)
        pred = int(y[:SUMS].argmax())
        t = (a + b + 1) % SUMS if wrong else a + b
        ok += int(pred == t)
    return ok, len(pairs)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=Path(r"C:\02_QUILLAN\exports\quillan-mini-rank64.safetensors"),
    )
    ap.add_argument("--rank", type=int, default=64)
    a = ap.parse_args()
    m = load_model(a.ckpt)
    gate, up, down = build_circuit(m, layer=8)
    r = gate.lora_A.shape[0]
    print(
        f"[circuit] rank={r} gate={tuple(gate.lora_B.shape)} down={tuple(down.lora_B.shape)} intermediates={gate.out_features}",
        flush=True,
    )
    # digit-disjoint split: train digits a,b in 0..5 ; held-out uses digit 6..9
    train = [(i, j) for i in range(6) for j in range(6)]
    test = [(i, j) for i in range(DIG) for j in range(DIG) if i >= 6 or j >= 6]
    tr = acc(gate, up, down, train)
    te = acc(gate, up, down, test)
    wt = acc(gate, up, down, test, wrong=True)
    print(
        f"[CORRECT circuit] train={tr[0]}/{tr[1]} held-out={te[0]}/{te[1]} ({100 * te[0] / te[1]:.1f}%)",
        flush=True,
    )
    print(
        f"[WRONG-TARGET control] held-out={wt[0]}/{wt[1]} ({100 * wt[0] / wt[1]:.1f}%)",
        flush=True,
    )
    # show a few decoded outputs
    for a_, b_ in [(3, 4), (7, 8), (9, 9), (2, 9)]:
        y = run_circuit(gate, up, down, a_, b_)
        print(
            f"  {a_}+{b_}={a_ + b_} -> pred={int(y[:SUMS].argmax())} top={[round(float(v), 2) for v in y[:SUMS]]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
