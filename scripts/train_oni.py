#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4-ONI trainer.

Wired to the unified build (quillan_v5_4_oni.py):
  - dense_pull council (all 34 deliberate) or gumbel_topk legacy
  - aux losses: load-balance KL, z-loss, CCRL entropy bonus, E_ICE ethics, QHIS, QICS
  - Gumbel tau annealing (gumbel_topk mode)
  - Lee-Mach-6 governor consumed: sigma scales swarm, ema_decay drives weight EMA
  - Cosine LR + warmup, grad clip, AdamW(0.9, 0.95)
  - Periodic val loss + decoded sample generation (unified tokenizer)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\02_QUILLAN\_dev")
from quillan_tokenizer_unified import UnifiedQuillanTokenizer  # noqa: E402
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni  # noqa: E402

DATA = Path(r"C:\02_QUILLAN\training_data\v9")
CKPT_DIR = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni")
LOG_DIR = Path(r"C:\02_QUILLAN\training_logs")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--router-mode", type=str, default="dense_pull",
                    choices=["dense_pull", "gumbel_topk"])
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=int, default=1)
    return ap.parse_args()


class Corpus:
    def __init__(self, split: str, seq_len: int):
        ids = np.memmap(DATA / f"{split}_ids.bin", dtype=np.uint16, mode="r")
        labels = np.memmap(DATA / f"{split}_labels.bin", dtype=np.int32, mode="r")
        n = min(len(ids), len(labels)) // seq_len * seq_len
        self.ids = ids[:n].reshape(-1, seq_len)
        self.labels = labels[:n].reshape(-1, seq_len)

    def __len__(self):
        return len(self.ids)

    def batch(self, bs, rng):
        idx = rng.integers(0, len(self), size=bs)
        return (torch.from_numpy(self.ids[idx].astype(np.int64)),
                torch.from_numpy(self.labels[idx].astype(np.int64)))


def cosine_lr(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))


def evaluate(model, val, rng, bs):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(16):
            x, y = val.batch(bs, rng)
            _, ce, _ = model(x, labels=y)
            losses.append(ce.item())
    model.train()
    return sum(losses) / len(losses)


def sample(model, tok, prompt="User: Hello, who are you?\n\nAssistant:"):
    model.eval()
    ids = tok.encode(prompt, domain="dialogue")
    out = model.generate(ids, max_tokens=60, temp=0.8)
    model.train()
    return tok.decode(out)


def main():
    args = parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    rng = np.random.default_rng(42)
    tok = UnifiedQuillanTokenizer()
    train = Corpus("train", args.seq_len)
    val = Corpus("val", args.seq_len)
    print(f"[DATA] train={len(train)} seqs  val={len(val)} seqs")

    cfg = QuillanOniConfig(n_layer=args.n_layer, max_seq_len=args.seq_len,
                           router_mode=args.router_mode, grad_checkpoint=False)
    model = QuillanRoninOni(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Quillan-Ronin v5.4.0-oni  n_layer={args.n_layer}  "
          f"router={args.router_mode}  params={n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    step, best_val = 0, float("inf")
    ema_sd = None
    latest = CKPT_DIR / "quillan_oni_latest.pt"

    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best_val = ck.get("best_val", best_val)
        ema_sd = ck.get("ema_sd")
        print(f"[RESUME] step {step}, best_val {best_val:.4f}")

    log_f = open(LOG_DIR / "oni_train_log.jsonl", "a", encoding="utf-8")
    model.train()
    t_start = time.time()
    running = []

    while step < args.steps:
        lr = cosine_lr(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        model.set_router_tau(model.tau_for_step(step, args.steps))

        opt.zero_grad()
        step_loss = 0.0
        t0 = time.time()
        for _ in range(args.grad_accum):
            x, y = train.batch(args.batch_size, rng)
            _, ce, aux = model(x, labels=y)
            loss = (ce + model.total_aux_loss(aux)) / args.grad_accum
            loss.backward()
            step_loss += ce.item() / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        step += 1
        running.append(step_loss)

        latency_ms = (time.time() - t0) * 1000.0
        _, ema_decay, _ = model.governor_adjust(latency_ms)
        if args.ema:
            with torch.no_grad():
                if ema_sd is None:
                    ema_sd = {k: v.detach().clone() for k, v in model.state_dict().items()
                              if v.dtype.is_floating_point}
                else:
                    d = ema_decay
                    for k, v in model.state_dict().items():
                        if k in ema_sd and v.dtype.is_floating_point:
                            ema_sd[k].mul_(d).add_(v.detach(), alpha=1 - d)

        if step % 10 == 0:
            avg = sum(running[-10:]) / len(running[-10:])
            sps = (time.time() - t_start) / step
            eta_h = (args.steps - step) * sps / 3600
            print(f"step {step}/{args.steps} loss={avg:.4f} lr={lr:.2e} gn={gn:.2f} "
                  f"{sps:.2f}s/st ETA={eta_h:.1f}h", flush=True)
            log_f.write(json.dumps({"step": step, "loss": avg, "lr": lr,
                                    "grad_norm": float(gn), "latency_ms": latency_ms}) + "\n")
            log_f.flush()

        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate(model, val, rng, args.batch_size)
            best_val = min(best_val, vl)
            print(f"[EVAL] step {step} val_loss={vl:.4f} (best {best_val:.4f})", flush=True)
            log_f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")

        if step % args.sample_every == 0 or step == args.steps:
            print("[SAMPLE]", repr(sample(model, tok)), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "best_val": best_val, "ema_sd": ema_sd, "cfg": vars(cfg),
                        "version": "5.4.0-oni"}, latest)
            print(f"[SAVE] {latest} @ step {step}", flush=True)

    print(f"[COMPLETE] {args.steps} steps. best_val={best_val:.4f}")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
