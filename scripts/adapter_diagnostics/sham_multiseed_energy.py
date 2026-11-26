#!/usr/bin/env python3
"""Energy-matched multi-seed sham (Nullion's requirement).

Builds a random residual with the SAME Frobenius norm as the true SVD residual
on each attention module (measured: true/sham energy ratio was 0.516 before).
Runs N seeds, each scored on the 80q large bank with pinned 1 thread, reusing
one model instance. Reports the null distribution (mean/sd/95th percentile)
so the real +9 must exceed the sham 95th percentile.
"""

from __future__ import annotations
import sys, math, json
from pathlib import Path
from dataclasses import asdict
import torch

ROOT = Path(r"C:\02_QUILLAN")
for p in (
    str(ROOT / "scripts"),
    str(ROOT / "03 - Training & Model"),
    str(ROOT / "scripts" / "adapter_diagnostics"),
):
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_sovereign_bitnet import QuillanSovereignConfig, QuillanSovereignBitNetLM
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache
from heldout_domain_large import BANKS, prompts

torch.set_num_threads(1)

QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
BASE = ROOT / "exports" / "quillan-mini-glassbox-base.safetensors"
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def build_targets():
    from transformers import AutoModelForCausalLM

    q = AutoModelForCausalLM.from_pretrained(
        QWEN, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    m = load_model(BASE)
    targets = []
    for li, layer in enumerate(m.layers):
        ql = q.model.layers[li]
        for sub in ATTN:
            mm = getattr(layer, sub, None)
            qq = getattr(ql.self_attn, sub, None)
            if mm is None or qq is None:
                continue
            We = qq.weight.detach().float()
            Wq = mm.unpack_effective_weight().float()
            if tuple(We.shape) != tuple(Wq.shape):
                continue
            R = We - Wq
            r = min(64, mm.lora_A.shape[0], *R.shape)
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            Atrue = Vh[:r].contiguous()
            Btrue = (U[:, :r] * S[:r]).contiguous()
            targets.append((mm, r, (Btrue @ Atrue).norm().item()))
    del q, m
    return targets


def score(m, tok, lids):
    tot = 0
    for dom in BANKS:
        Q, K = BANKS[dom]
        with torch.inference_mode():
            for seq, gold in zip(prompts(tok, dom), K):
                lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
                tot += int(int(lg.argmax()) == gold)
    return tot


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    targets = build_targets()
    cfgQ = 64
    base = load_model(BASE)
    cfg = QuillanSovereignConfig(**{**asdict(base.cfg), "eggroll_rank": cfgQ})
    m = QuillanSovereignBitNetLM(cfg).eval()
    bsd, msd = base.state_dict(), m.state_dict()
    m.load_state_dict(
        {
            k: v
            for k, v in bsd.items()
            if k in msd and ".lora_" not in k and msd[k].shape == v.shape
        },
        strict=False,
    )
    enable_cache(m)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    # find the corresponding modules in m by index order
    mods = [
        getattr(m.layers[li], sub)
        for li in range(len(m.layers))
        for sub in ATTN
        if getattr(m.layers[li], sub, None) is not None
    ]
    assert len(mods) == len(targets), (len(mods), len(targets))
    base_score = score(m, tok, lids)
    print(f"base={base_score}/80 modules={len(targets)}", flush=True)
    scores = []
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(7000 + seed)
        for mod, (_, r, target_norm) in zip(mods, targets):
            out, inp = mod.out_features, mod.in_features
            A = torch.randn(r, inp, generator=g)
            B = torch.randn(out, r, generator=g)
            cur = (B @ A).norm().item()
            scale = target_norm / max(cur, 1e-9)
            A = A * math.sqrt(scale)
            B = B * math.sqrt(scale)
            with torch.no_grad():
                mod.lora_A.copy_(A.to(mod.lora_A.dtype))
                mod.lora_B.copy_(B.to(mod.lora_B.dtype))
                mod._wc = None
        s = score(m, tok, lids)
        scores.append(s - base_score)
        print(f"  seed {seed}: {s}/80 gain={s - base_score:+d}", flush=True)
    scores_sorted = sorted(scores)
    p95 = scores_sorted[int(0.95 * len(scores))]
    mean = sum(scores) / len(scores)
    sd = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
    print(
        f"SHAM null: mean={mean:+.2f} sd={sd:.2f} p95={p95:+d} max={max(scores):+d}",
        flush=True,
    )
    print(f"REAL gain=+9 ; exceeds sham p95? {'YES' if 9 > p95 else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
