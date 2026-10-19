#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN DEEP ANNEALING ENGINE
=================================
Two-phase curriculum convergence to breach Loss < 1.5:
  Phase 1 (Head-Only, 150 steps): Aligns lm_head, LayerNorms, and 34 MoE routers
            at LR 1.5e-4 for rapid distribution collapse from ~7.7 → ~2.5
  Phase 2 (Full-Layer, 300 steps): Full 455M-param annealing at LR 6.5e-5 with
            cosine decay to floor 5e-6, targeting final Loss < 1.5

Security: weights_only=True throughout (CWE-502).
Resource:  PyTorch CPU ceiling = 3 threads; grad clip max_norm=1.0.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\02_QUILLAN")
for p in [
    str(REPO_ROOT),
    str(REPO_ROOT / "03 - Training & Model"),
    str(REPO_ROOT / "09 - Projects" / "projects" / "oni"),
    str(REPO_ROOT / "scripts"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("quillan_deep_anneal")

# ── CPU thread ceiling ─────────────────────────────────────────────────────────
torch.set_num_threads(3)
LOGGER.info("PyTorch CPU ceiling: 3 threads")


def load_model_from_ckpt(ckpt_path: Path, n_layer: int = 6):
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
    cfg = QuillanOniConfig(
        vocab_size=50257, hidden_dim=1024, ffn_dim=2048,
        n_layer=n_layer, num_experts=34, top_k=4, max_seq_len=512,
    )
    model = QuillanRoninOni(cfg)
    LOGGER.info("Loading checkpoint: %s", ckpt_path.name)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = data.get("model_state_dict", data.get("model", data))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    LOGGER.info("Checkpoint bound (missing=%d, unexpected=%d)", len(missing), len(unexpected))
    return model, cfg


def load_dataset(path: Path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    input_ids = data["input_ids"]
    labels = data["labels"]
    LOGGER.info(
        "Dataset loaded: %d samples, seq_len=%d, supervised_ratio=%.2f%%",
        input_ids.shape[0], input_ids.shape[1],
        100.0 * (labels != -100).sum().item() / labels.numel(),
    )
    return input_ids, labels


def check_disk_headroom(floor_gb: float = 8.0):
    import shutil
    free_gb = shutil.disk_usage("C:\\").free / (1024**3)
    if free_gb < floor_gb:
        raise RuntimeError(f"Disk critically low: {free_gb:.2f} GB free (floor: {floor_gb} GB)")
    LOGGER.info("Disk headroom: %.2f GB free", free_gb)


def run_phase(
    model: "QuillanRoninOni",
    cfg,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    lr: float,
    batch_size: int = 2,
    grad_accum: int = 2,
    seq_len: int = 192,
    head_only: bool = False,
    phase_name: str = "Phase",
    save_path: Optional[Path] = None,
    warmup_frac: float = 0.05,
) -> float:
    """Runs one training phase. Returns final step loss."""
    n_samples = input_ids.shape[0]

    if head_only:
        trainable = []
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ["lm_head", "ln", "norm", "gate", "router"]):
                param.requires_grad = True
                trainable.append(param)
            else:
                param.requires_grad = False
        trainable_m = sum(p.numel() for p in trainable) / 1e6
        LOGGER.info("%s — Head-Only mode: %.2fM trainable params", phase_name, trainable_m)
    else:
        for p in model.parameters():
            p.requires_grad = True
        trainable = list(model.parameters())
        LOGGER.info("%s — Full-parameter mode: %.2fM params", phase_name, sum(p.numel() for p in trainable) / 1e6)

    warmup_steps = max(1, int(steps * warmup_frac))
    optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    LOGGER.info(
        "%s START — steps=%d, lr=%.2e, batch=%d, grad_accum=%d, seq_len=%d, warmup=%d",
        phase_name, steps, lr, batch_size, grad_accum, seq_len, warmup_steps,
    )

    model.train()
    optimizer.zero_grad()
    t0 = time.perf_counter()
    last_loss = float("nan")

    for step in range(1, steps + 1):
        # LR schedule: warmup then cosine
        if step <= warmup_steps:
            cur_lr = lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = lr * 0.05 + 0.5 * (lr * 0.95) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Random batch
        idx = torch.randint(0, n_samples, (batch_size,))
        x = input_ids[idx, :seq_len]
        y = labels[idx, :seq_len]

        # Forward
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

        # Causal shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, cfg.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        ) / grad_accum
        loss.backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        last_loss = loss.item() * grad_accum

        if step % 25 == 0 or step == steps:
            elapsed = time.perf_counter() - t0
            tps = (step * batch_size * seq_len) / max(1e-3, elapsed)
            LOGGER.info(
                "%s Step %4d/%d | Loss: %.4f | LR: %.2e | %.0f tok/s | %.1fs elapsed",
                phase_name, step, steps, last_loss, cur_lr, tps, elapsed,
            )
            check_disk_headroom()

    # Save checkpoint
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, save_path)
        LOGGER.info("%s checkpoint saved → %s (loss=%.4f)", phase_name, save_path.name, last_loss)

    return last_loss


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("  👑 QUILLAN DEEP ANNEALING ENGINE — Two-Phase Convergence")
    LOGGER.info("=" * 70)
    check_disk_headroom()

    base_ckpt = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_frontier_v2_best.pt"
    dataset_path = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_master_gold_training_v1.pt"
    phase1_out = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_phase1_head_aligned.pt"
    phase2_out = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_frontier_v2_best.pt"  # overwrite production

    LOGGER.info("Base checkpoint : %s", base_ckpt.name)
    LOGGER.info("Dataset         : %s", dataset_path.name)

    input_ids, labels = load_dataset(dataset_path)
    model, cfg = load_model_from_ckpt(base_ckpt, n_layer=6)

    # ── Phase 1: Head & Router Alignment ──────────────────────────────────────
    p1_loss = run_phase(
        model, cfg, input_ids, labels,
        steps=150,
        lr=1.5e-4,
        batch_size=2,
        grad_accum=2,
        seq_len=192,
        head_only=True,
        phase_name="Phase-1[Head+Router]",
        save_path=phase1_out,
        warmup_frac=0.08,
    )
    LOGGER.info("Phase 1 complete → Final Loss: %.4f", p1_loss)
    gc.collect()

    # ── Phase 2: Full-Layer Deep Anneal ───────────────────────────────────────
    p2_loss = run_phase(
        model, cfg, input_ids, labels,
        steps=300,
        lr=6.5e-5,
        batch_size=2,
        grad_accum=4,
        seq_len=192,
        head_only=False,
        phase_name="Phase-2[FullLayer]",
        save_path=phase2_out,
        warmup_frac=0.05,
    )
    LOGGER.info("Phase 2 complete → Final Loss: %.4f", p2_loss)
    LOGGER.info("=" * 70)
    LOGGER.info("  🎯 DEEP ANNEAL COMPLETE | Production Loss: %.4f", p2_loss)
    LOGGER.info("  📁 Saved → %s", phase2_out.name)
    LOGGER.info("=" * 70)

    # Notify gateway to hot-reload
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/reload",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        LOGGER.info("Gateway hot-reload triggered successfully.")
    except Exception as e:
        LOGGER.warning("Gateway reload: %s (manual restart may be needed)", e)


if __name__ == "__main__":
    main()
