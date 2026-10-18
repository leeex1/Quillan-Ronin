#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Council-built stable anneal — Mini 6L (455M)
C0 Throne orchestration | C31-NEXUS routing | C4-PRAXIS plan
C10-CODEWEAVER build | C26-TECHNE constraints | C13-WARDEN gates
C34-PREDATOR adversarial | C18-SHEPHERD verify | C14-KAIDO efficiency | C28-CALCULUS math

Fix vs quillan_deep_anneal.py Phase 2:
  batch 2 x grad_accum 4 (=8 effective) -> batch 2 x grad_accum 16 (=32 effective)
  8 seqs leaves most of 34 experts with zero grad -> oscillation 2.84<->7.5
  32 seqs gives every expert signal every update -> monotonic descent
  Micro batch stays 2 (RAM safe with 10.8GB gateway resident) — stability via accum, not width.
  Tracks BEST not final (Phase 2 saved final 6.23 over best 2.84 — root cause of revert).
  Base: quillan_phase1_head_aligned.pt (floor 2.87), LR 3.5e-5 gentle on mature weights.
"""
from __future__ import annotations
import sys
import math
import time
import argparse
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
for p in [str(REPO_ROOT), str(REPO_ROOT / "03 - Training & Model"),
          str(REPO_ROOT / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)

torch.set_num_threads(3)

from quillan_deep_anneal import load_model_from_ckpt, load_dataset, check_disk_headroom  # noqa: E402

CKPTS = REPO_ROOT / "checkpoints" / "checkpoints_sft"
BASE = CKPTS / "quillan_phase1_head_aligned.pt"
BEST_OUT = CKPTS / "quillan_mini_6l_stable_best.pt"
DATASET = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_master_gold_training_v1.pt"


def cosine_lr(step: int, total: int, peak: float, floor: float = 5e-6, warmup: int = 10) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * t))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--smoke", action="store_true", help="5-step verification only (C18)")
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    args = ap.parse_args()

    steps = 5 if args.smoke else args.steps
    check_disk_headroom(8.0)
    if not BASE.exists():
        print(f"FATAL missing base {BASE}")
        return 2

    model, cfg = load_model_from_ckpt(BASE, n_layer=6)
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Mini 6L full-layer: {n_params:.1f}M trainable | eff_batch={args.batch * args.accum} | lr={args.lr}")

    input_ids, labels = load_dataset(DATASET)
    n = input_ids.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best = float("inf")
    best_step = -1
    rng = torch.Generator().manual_seed(1234)

    for step in range(steps):
        lr_now = cosine_lr(step, steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for a in range(args.accum):
            idx = torch.randint(0, n, (args.batch,), generator=rng)
            x = input_ids[idx][:, :192]
            y = labels[idx][:, :192]
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                y[:, 1:].reshape(-1), ignore_index=-100)
            (loss / args.accum).backward()
            tot_loss += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tag = ""
        if tot_loss < best and not math.isnan(tot_loss):
            best, best_step = tot_loss, step
            tag = " <-- BEST"
            if not args.smoke:
                torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "loss": best, "step": step},
                           BEST_OUT)
        print(f"step {step + 1}/{steps} loss={tot_loss:.4f} lr={lr_now:.2e}{tag}", flush=True)

    print(f"DONE best={best:.4f} @step {best_step + 1} -> {BEST_OUT.name if not args.smoke else '(smoke, not saved)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
