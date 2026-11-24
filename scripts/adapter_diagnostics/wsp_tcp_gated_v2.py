#!/usr/bin/env python3
"""WSP + Truth Calibration Protocol gate (v2).

Same promotion rule as wsp_tcp_gated.py: a closed-form write is PROMOTED only if
  (1) train does not drop, (2) falsification does not drop,
  (3) protected baseline does not drop, and (4) total strictly rises.
Otherwise REVERTED. Zero backprop.

v2 fixes two operational defects found in v1:
  - torch.set_num_threads(1): v1 thrashed 8 OpenMP threads on the old CPU
    (17 min wall / 6.9 s CPU for 6 steps).
  - checkpoint saved on EVERY promotion, so a promoted gain cannot be lost
    if the process is killed before the final loop step.
  - snapshots cached across reject steps (weights unchanged on reject).
"""

from __future__ import annotations
import sys, time, os
from pathlib import Path
import torch

# --- SM61 GPU AUTO-ROUTE (2026-11-23, reversible) -------------------------------
# System torch (cu130) has no sm_61 kernels; the isolated cu118 venv does. Re-exec
# through the GPU wrapper so the DEFAULT invocation uses the GTX 1050. Model logic
# below is untouched. Disable: set env QUILLAN_SM61_DIRECT=1, or restore .bak file.
if os.environ.get("QUILLAN_SM61_DIRECT") != "1" and "+cu130" in torch.__version__:
    _vpy = Path(r"C:\02_QUILLAN\venv_sm61_test\Scripts\python.exe")
    _wrap = Path(r"C:\02_QUILLAN\scripts\adapter_diagnostics\wsp_tcp_gpu_sm61.py")
    if _vpy.exists() and _wrap.exists():
        os.environ["QUILLAN_SM61_DIRECT"] = "1"  # guard: wrapper imports this module
        os.execv(str(_vpy), [str(_vpy), "-u", str(_wrap)] + sys.argv[1:])
# -------------------------------------------------------------------------------

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from general_knowledge import QUESTIONS as GQ, KEY as GK, LETTERS
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache

torch.set_num_threads(1)

CK = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
OUT = ROOT / "exports" / "quillan-mini-wsp-tcp2.safetensors"


def prompts(tok):
    return [
        tok.encode(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:",
            add_special_tokens=False,
        )
        for i, (q, opts, _) in enumerate(GQ)
    ]


def snap(m, all_ids, lids):
    cap = {}
    hk = m.layers[-1].down_proj.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("k", inp[0].detach().float())
    )
    L, Ks = [], []
    with torch.inference_mode():
        for ids in all_ids:
            out = m(torch.tensor([ids]))[0]
            L.append(out[0, -1, lids].float())
            Ks.append(cap["k"][0, -1, :].detach().clone())
    hk.remove()
    return torch.stack(L), torch.stack(Ks)


def acc(X, idx, K):
    return int((X[idx].argmax(-1) == K[idx]).sum())


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=8)
    ap.add_argument("--falsify-n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--frob", type=float, default=0.010)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--max-wall", type=float, default=480.0)
    a = ap.parse_args()

    t0 = time.time()
    tok = tok_for(151936)
    lids = letter_ids(tok, 151936)
    m = load_model(CK)
    enable_cache(m)
    down = m.layers[-1].down_proj
    ids = prompts(tok)
    K = torch.tensor(GK)
    E = m.embed_tokens.unpack_float().float()[lids]

    tr = list(range(a.train_n))
    fal = list(range(a.train_n, a.train_n + a.falsify_n))
    base = list(range(a.train_n + a.falsify_n, len(GQ)))

    L, Ks = snap(m, ids, lids)
    b_tr, b_fal, b_base = acc(L, tr, K), acc(L, fal, K), acc(L, base, K)
    print(
        f"[init] train={b_tr}/{len(tr)} falsify={b_fal}/{len(fal)} "
        f"baseline={b_base}/{len(base)}  ({time.time() - t0:.1f}s)",
        flush=True,
    )
    promoted = 0
    for step in range(a.steps):
        if time.time() - t0 > a.max_wall:
            print(f"[stop] wall budget reached at step {step}", flush=True)
            break
        pred = L.argmax(-1)
        wrong = [i for i in tr if int(pred[i]) != int(K[i])]
        if not wrong:
            print(f"[step {step}] all train correct", flush=True)
            break
        i = wrong[step % len(wrong)]
        k, v = Ks[i], E[int(K[i])]
        A0 = down.lora_A.detach().clone()
        B0 = down.lora_B.detach().clone()
        down._wc = None
        down.online_wsp_update(
            k.unsqueeze(1).contiguous(),
            v.unsqueeze(1),
            a_anchor=None,
            ridge_lambda=a.lam,
            max_rel_frob=a.frob,
        )
        L2, _ = snap(m, ids, lids)
        n_tr, n_fal, n_base = acc(L2, tr, K), acc(L2, fal, K), acc(L2, base, K)
        ok = (
            (n_tr >= b_tr)
            and (n_fal >= b_fal)
            and (n_base >= b_base)
            and (n_tr + n_fal + n_base) > (b_tr + b_fal + b_base)
        )
        if ok:
            b_tr, b_fal, b_base = n_tr, n_fal, n_base
            L = L2
            # weights changed -> activations changed; refresh Ks once
            _, Ks = snap(m, ids, lids)
            promoted += 1
            m.save_packed_safetensors(OUT)  # incremental: never lose a promotion
            print(
                f"[step {step}] PROMOTE q{i}: train={n_tr} falsify={n_fal} "
                f"baseline={n_base}  SAVED ({time.time() - t0:.1f}s)",
                flush=True,
            )
        else:
            with torch.no_grad():
                down.lora_A.copy_(A0)
                down.lora_B.copy_(B0)
            print(
                f"[step {step}] reject q{i}: {n_tr}/{n_fal}/{n_base} "
                f"(best {b_tr}/{b_fal}/{b_base})  ({time.time() - t0:.1f}s)",
                flush=True,
            )

    m.save_packed_safetensors(OUT)
    print(
        f"[SAVED] promoted={promoted} best={b_tr}/{b_fal}/{b_base}",
        flush=True,
    )
    m2 = load_model(OUT)
    enable_cache(m2)
    L3, _ = snap(m2, ids, lids)
    print(
        f"[FRESH RELOAD] train={acc(L3, tr, K)}/{len(tr)} "
        f"falsify={acc(L3, fal, K)}/{len(fal)} baseline={acc(L3, base, K)}/{len(base)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
