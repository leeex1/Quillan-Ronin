#!/usr/bin/env python3
"""Quillan-Ronin v5.4-ONI : Colab GPU continued pre-training (all HF datasets).

Resumable, validation-selected checkpoints, AMP (fp16+GradScaler on T4, bf16 on A100).
Run from a Colab kernel:  !python colab_train.py --minutes 60
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HF_REPO = "CrashOverrideX/Quillan_Samurai_sets"
HF_FILES = [
    "quillan_corpus_CLEAN_V7.pt", "full_train.pt", "instruct_train.pt",
    "GPT_5.5_Distilled.pt", "quillan_tokenized.pt", "code_train.pt", "train.pt",
    "quillan_12mb_training_dataset.pt", "quillan_science_absolute.pt",
    "quillan_science_additional.pt", "full_dataset.pt",
]
VOCAB = 50257
EOS = 50256


def log(**kw):
    print(json.dumps(kw, default=str), flush=True)


def to_stream(obj) -> torch.Tensor | None:
    """Normalise any saved token container into one flat 1-D int tensor."""
    if isinstance(obj, dict):
        for k in ("input_ids", "ids", "tokens", "train"):
            if k in obj:
                return to_stream(obj[k])
        vals = [v for v in obj.values() if torch.is_tensor(v)]
        return to_stream(vals[0]) if vals else None
    if torch.is_tensor(obj):
        return obj.reshape(-1).long()
    if isinstance(obj, (list, tuple)) and obj and torch.is_tensor(obj[0]):
        return torch.cat([t.reshape(-1).long() for t in obj])
    return None


def load_sources(data_dir: Path, extra: list[Path]):
    from huggingface_hub import hf_hub_download

    srcs = []
    paths = [(p.name, p) for p in extra if p.exists()]
    for name in HF_FILES:
        try:
            paths.append((name, Path(hf_hub_download(HF_REPO, name, repo_type="dataset", local_dir=str(data_dir)))))
        except Exception as e:  # noqa: BLE001 - report and continue; never silently skip
            log(event="hf_download_failed", file=name, err=str(e)[:200])
    for name, p in paths:
        t = to_stream(torch.load(p, map_location="cpu", weights_only=False))
        if t is None or t.numel() < 4096:
            log(event="source_skipped", file=name, reason="unreadable_or_tiny")
            continue
        mx, mn = int(t.max()), int(t.min())
        if mx >= VOCAB or mn < 0:
            log(event="source_skipped", file=name, reason="token_ids_out_of_range", min=mn, max=mx)
            continue
        t = t.to(torch.int32)  # int32: safe and 2x smaller than int64
        n_val = max(2048, t.numel() // 200)
        srcs.append({"name": name, "train": t[:-n_val], "val": t[-n_val:]})
        log(event="source_loaded", file=name, tokens=int(t.numel()))
    if not srcs:
        raise RuntimeError("no usable data sources")
    total = sum(s["train"].numel() for s in srcs)
    w = np.array([s["train"].numel() ** 0.5 for s in srcs], dtype=np.float64)
    gold = [i for i, s in enumerate(srcs) if "gold" in s["name"].lower()]
    w = w / w.sum()
    log(event="data_ready", sources=len(srcs), total_train_tokens=total)
    return srcs, w


def batch(srcs, w, bs, seq, rng, key="train", device="cuda"):
    xs, ys = [], []
    for _ in range(bs):
        s = srcs[rng.choice(len(srcs), p=w)][key]
        i = int(rng.integers(0, s.numel() - seq - 1))
        c = s[i:i + seq + 1].long()
        xs.append(c[:-1])
        ys.append(c[1:])
    return torch.stack(xs).to(device, non_blocking=True), torch.stack(ys).to(device, non_blocking=True)


def lr_at(step, warm, total, peak, floor):
    if step < warm:
        return peak * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/content/colab_bundle")
    ap.add_argument("--data-dir", default="/content/hf_data")
    ap.add_argument("--out-dir", default="/content/ckpt")
    ap.add_argument("--init", default="quillan_6l_baseline_fp16.pt")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=0, help="wall-clock budget; 0 = unlimited")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--unfreeze", choices=["wide", "full"], default="wide")
    ap.add_argument("--smoke", action="store_true", help="3-step functional test then exit")
    a = ap.parse_args()

    bundle, out = Path(a.bundle), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(bundle / "modules"))
    sys.path.insert(0, str(bundle))
    import quillan_v5_4_oni as _m
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

    # The model imports every formal-paper pack inside try/except ImportError and silently
    # disables it on failure. Make that failure loud instead.
    flags = {k: v for k, v in vars(_m).items() if k.endswith("WIRED")}
    off = sorted(k for k, v in flags.items() if v is not True)
    log(event="paper_packs", wired=len(flags) - len(off), total=len(flags), not_wired=off,
        loaded_from=str(Path(_m.paper_01_profiler.__file__).parent) if hasattr(_m, "paper_01_profiler") else "n/a")
    if off:
        raise RuntimeError(f"formal-paper packs NOT wired on this machine: {off}")

    assert torch.cuda.is_available(), "no GPU: Runtime > Change runtime type > GPU"
    dev = torch.device("cuda")
    cc = torch.cuda.get_device_capability()
    use_bf16 = cc[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    log(event="gpu", name=torch.cuda.get_device_name(), cc=cc, vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), amp=str(amp_dtype))

    ck = torch.load(bundle / a.init, map_location="cpu", weights_only=False)
    valid = {f.name for f in dataclasses.fields(QuillanOniConfig)}
    cfg_d = {k: v for k, v in dict(ck["config"]).items() if k in valid}
    cfg_d["device"] = "cuda"
    cfg = QuillanOniConfig(**cfg_d)
    model = QuillanRoninOni(cfg)
    sd = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v) for k, v in ck["model_state_dict"].items()}
    miss, unexp = model.load_state_dict(sd, strict=True), None
    del sd, ck
    log(event="checkpoint_loaded", strict=True)

    keys = ("expert", "router", "ln", "norm", "gate", "w_gate", "attn", "c_attn", "c_proj", "wte", "prism", "bridge", "finalizer", "lora")
    for n, p in model.named_parameters():
        p.requires_grad = True if a.unfreeze == "full" else any(k in n for k in keys)
    for n, p in model.named_parameters():
        if not p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.half()  # frozen weights fp16 to save VRAM; autocast handles it
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    log(event="params", trainable_m=round(sum(p.numel() for p in params) / 1e6, 1),
        total_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        vram_gb=round(torch.cuda.memory_allocated() / 1e9, 2))

    srcs, w = load_sources(Path(a.data_dir), [bundle / "quillan_master_gold_training_v1.pt"])
    rng = np.random.default_rng(1234)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    @torch.no_grad()
    def val_loss(n=24):
        model.eval()
        tot = 0.0
        vr = np.random.default_rng(7)
        for _ in range(n):
            x, y = batch(srcs, w, a.bs, a.seq, vr, "val")
            with torch.autocast("cuda", dtype=amp_dtype):
                o = model(x, use_cache=False, deliberation=False)
            lg = (o[0] if isinstance(o, tuple) else o).float()
            tot += F.cross_entropy(lg.view(-1, lg.size(-1)), y.view(-1)).item()
        model.train()
        return tot / n

    def save(tag, step, vl):
        sdh = {k: (v.half() if torch.is_tensor(v) and v.dtype == torch.float32 else v) for k, v in model.state_dict().items()}
        tmp = out / f"{tag}.tmp"
        torch.save({"model_state_dict": sdh, "config": dataclasses.asdict(cfg), "step": step, "loss": vl}, tmp)
        os.replace(tmp, out / f"{tag}.pt")
        log(event="saved", file=f"{tag}.pt", step=step, val=vl)

    steps = 3 if a.smoke else a.steps
    t0, best, tok, run_loss = time.time(), float("inf"), 0, []
    for step in range(1, steps + 1):
        lr = lr_at(step, a.warmup, a.steps, a.lr, a.lr * 0.05)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        sl = 0.0
        for _ in range(a.accum):
            x, y = batch(srcs, w, a.bs, a.seq, rng)
            with torch.autocast("cuda", dtype=amp_dtype):
                o = model(x, use_cache=False, deliberation=False)
            lg = (o[0] if isinstance(o, tuple) else o).float()
            loss = F.cross_entropy(lg.view(-1, lg.size(-1)), y.view(-1)) / a.accum
            scaler.scale(loss).backward()
            sl += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        tok += a.bs * a.accum * a.seq
        run_loss.append(sl)
        if step % 10 == 0 or step == 1 or a.smoke:
            el = time.time() - t0
            log(event="step", step=step, loss=round(sum(run_loss[-10:]) / len(run_loss[-10:]), 4), lr=lr,
                gn=round(float(gn), 3), tok_s=round(tok / el), tokens_seen=tok, vram_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        if step % a.val_every == 0 or step == steps:
            vl = val_loss()
            log(event="val", step=step, val_loss=round(vl, 4), val_ppl=round(math.exp(min(20, vl)), 2))
            if vl < best:
                best = vl
                save("quillan_6l_colab_best", step, vl)
        if step % a.save_every == 0:
            save("quillan_6l_colab_latest", step, sl)
        if a.minutes and (time.time() - t0) / 60 > a.minutes:
            log(event="time_budget_reached", step=step)
            break
    save("quillan_6l_colab_latest", step, sl)
    log(event="done", steps=step, best_val=best, tokens=tok)


if __name__ == "__main__":
    main()
