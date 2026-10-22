#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_sovereign_sft.py — Hardened Master SFT Launcher for Quillan-Ronin v5.4.0-ONI (Rev 2)
===========================================================================================
Includes all 6 Phase-1 hardening upgrades PLUS the Rev-2 Dry-Run Telemetry Fixes:
1. Fixes `Aux: 16.3803` domination via `compute_safe_aux_loss()` (scales by `aux_alpha` &
   excludes the singular `x - x.detach()` zero-norm derivative in `qhis_fidelity`).
2. Fixes `GradNorm: 91,424,712.0` LoRA starvation via `clip_gradients_two_stage()`:
   per-tensor pre-clipping (`max_tensor_norm=5.0`) + NaN/Inf scrubbing before global L2 clip (`1.0`).
3. Logs `max_grad_tensor` and `grad_norm` in `sft_telemetry.jsonl` for full observability.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_live_training_upgrades import (
    canonical_state_dict_for_save,
    sample_early_ternary_logits,
    vectorize_model_in_place,
)


# ============================================================================
# 1. HARDWARE & OS TUNING (C14-KAIDO i5-7500 Optimization)
# ============================================================================
def configure_host_environment(threads: int = 4, low_priority: bool = True) -> None:
    """Optimizes CPU thread affinity, flushes subnormal floats, and sets OS priority."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads // 2))
    except RuntimeError:
        pass

    if hasattr(torch, "set_flush_denormal"):
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass

    if low_priority and sys.platform == "win32":
        try:
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass


# ============================================================================
# 2. ZERO-COPY BINARY MEMMAP CORPUS (Train + Validation)
# ============================================================================
class MemmapCorpus:
    """Zero-allocation memory-mapped token dataset with NaN target guard."""

    def __init__(self, data_dir: Path, split: str = "train", seq_len: int = 512, seed: int = 42):
        self.split = split
        self.seq_len = seq_len
        ids_path = data_dir / f"{split}_ids.bin"
        labels_path = data_dir / f"{split}_labels.bin"
        if not ids_path.exists() or not labels_path.exists():
            raise FileNotFoundError(f"Missing binary memmap files for split '{split}' in {data_dir}")

        self.ids_mmap = np.memmap(ids_path, dtype=np.uint16, mode="r")
        self.labels_mmap = np.memmap(labels_path, dtype=np.int32, mode="r")
        self.n_seqs = min(len(self.ids_mmap), len(self.labels_mmap)) // seq_len
        self.total_tokens = self.n_seqs * seq_len
        self.ids = self.ids_mmap[:self.total_tokens].reshape(-1, seq_len)
        self.labels = self.labels_mmap[:self.total_tokens].reshape(-1, seq_len)
        self.rng = np.random.default_rng(seed)

    def sample_batch(
        self,
        batch_size: int = 1,
        min_targets: int = 16,
        deterministic_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices: List[int] = []
        for b in range(batch_size):
            if deterministic_idx is not None:
                idx = (deterministic_idx + b) % self.n_seqs
                indices.append(idx)
                continue

            for _attempt in range(64):
                idx = int(self.rng.integers(0, self.n_seqs))
                valid_count = int((self.labels[idx, 1:] != -100).sum())
                if valid_count >= min_targets:
                    indices.append(idx)
                    break
            else:
                indices.append(idx)

        b_x = torch.from_numpy(self.ids[indices].astype(np.int64))
        b_y = torch.from_numpy(self.labels[indices].astype(np.int64))
        return b_x, b_y


# ============================================================================
# 3. MANDATORY RULE #12: ES-AT-SCALE FORGETTING MITIGATION ANCHOR
# ============================================================================
class ESAtScaleForgettingAnchor:
    """
    Implements Mandatory Rule #12 (ESFM — Formula 23):
    theta_{t+1} = theta_opt + mu_mem * (EMA[theta]_t - theta_opt) * 1(t mod 5 == 0)
    """

    def __init__(
        self,
        trainable_params: List[nn.Parameter],
        ema_decay: float = 0.995,
        mu_mem: float = 0.001,
        interval: int = 5,
    ):
        self.params = [p for p in trainable_params if p.is_floating_point()]
        self.ema_decay = ema_decay
        self.mu_mem = mu_mem
        self.interval = interval
        self.ema_snapshot: List[torch.Tensor] = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def step(self, global_step: int) -> None:
        pull_now = (global_step % self.interval == 0)
        for p, ema_p in zip(self.params, self.ema_snapshot):
            ema_p.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
        if pull_now:
            for p, ema_p in zip(self.params, self.ema_snapshot):
                p.data.add_(ema_p - p.data, alpha=self.mu_mem)


# ============================================================================
# 4. SCHEDULERS, SAFE AUX LOSS & TWO-STAGE GRADIENT CLIPPING (REV-2 FIXES)
# ============================================================================
def get_lr(step: int, total_steps: int, peak_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def get_staged_aux_alpha(step: int, total_steps: int) -> Tuple[float, str]:
    """
    Deliverable C 3-Stage Curriculum (handles short canary runs cleanly):
    - Stage 1 (Warm-In, first 25%): aux_alpha = 0.001
    - Stage 2 (Pressure, 25% - 75%): aux_alpha = 0.003
    - Stage 3 (Anneal, final 25%): aux_alpha = 0.0015
    """
    if total_steps < 10:
        return 0.001, "Stage1:WarmIn"

    frac = (step - 1) / max(1, total_steps)
    if frac < 0.25:
        return 0.001, "Stage1:WarmIn"
    elif frac < 0.75:
        return 0.003, "Stage2:Pressure"
    else:
        return 0.0015, "Stage3:Anneal"


def compute_safe_aux_loss(
    aux_dict: Dict[str, Any],
    aux_alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes a bounded, differentiable auxiliary loss from `aux_dict` while
    excluding the singular `qhis_fidelity(x.detach(), x)` zero-distance norm trap
    that caused `Aux: 16.3803` and `GradNorm: 91,424,712.0` on Step 1.
    """
    loss = torch.zeros((), device=device, dtype=torch.float32)

    # 1. Load balance (active if gumbel_topk, 0 in dense_pull)
    lb = aux_dict.get("load_balance")
    if torch.is_tensor(lb) and torch.isfinite(lb):
        loss = loss + aux_alpha * torch.clamp(lb.float(), min=0.0, max=10.0)

    # 2. Router Z-loss stabilization (clamped to prevent 16,384x spikes)
    zl = aux_dict.get("z_loss")
    if torch.is_tensor(zl) and torch.isfinite(zl):
        loss = loss + (aux_alpha * 0.5) * torch.clamp(zl.float(), min=0.0, max=20.0)

    # 3. Council Shannon Entropy bonus (encourages all 34 experts to stay active)
    # Offset by ln(34) = 3.52636 so term is a non-negative KL-to-uniform penalty >= 0.0
    ent = aux_dict.get("entropy")
    if torch.is_tensor(ent) and torch.isfinite(ent):
        target_ent = math.log(34.0)
        ent_gap = torch.clamp(target_ent - ent.float(), min=0.0, max=target_ent)
        loss = loss + (aux_alpha * 2.0) * ent_gap

    # 4. E_ICE Ethics constraint (if differentiable and finite)
    eth = aux_dict.get("ethics")
    if torch.is_tensor(eth) and eth.requires_grad and torch.isfinite(eth):
        loss = loss + (aux_alpha * 1.0) * torch.clamp(eth.float(), min=0.0, max=1.0)

    return loss


def clip_gradients_two_stage(
    named_trainable_params: List[Tuple[str, nn.Parameter]],
    per_tensor_max_norm: float = 5.0,
    global_max_norm: float = 1.0,
) -> Tuple[float, str, float]:
    """
    Two-Stage Gradient Clipper (C28-CALCULUS + C34-PREDATOR):
    Stage A: Scrubs NaN/Inf in-place and clips any single outlier tensor at `per_tensor_max_norm`
             so one noisy norm/gate tensor can NEVER divide all 696 other LoRA tensors by 91.4M.
    Stage B: Applies standard global L2 `clip_grad_norm_` at `global_max_norm`.

    Returns: (global_norm_before_global_clip, max_tensor_name, max_tensor_raw_norm)
    """
    max_tensor_norm = 0.0
    max_tensor_name = "none"
    params_with_grad: List[nn.Parameter] = []

    for name, p in named_trainable_params:
        if p.grad is None:
            continue

        # Scrub any element-wise NaN/Inf before norm computation
        if not torch.isfinite(p.grad).all():
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        t_norm = float(p.grad.detach().data.norm(2).item())
        if t_norm > max_tensor_norm:
            max_tensor_norm = t_norm
            max_tensor_name = name

        # Stage A: Per-tensor outlier clamp
        if t_norm > per_tensor_max_norm:
            scale = per_tensor_max_norm / (t_norm + 1e-6)
            p.grad.detach().mul_(scale)

        params_with_grad.append(p)

    if not params_with_grad:
        return 0.0, max_tensor_name, 0.0

    # Stage B: Global L2 norm clip across the sanitized & bounded parameter set
    global_norm = float(torch.nn.utils.clip_grad_norm_(params_with_grad, global_max_norm).item())
    return global_norm, max_tensor_name, max_tensor_norm


# ============================================================================
# 5. VALIDATION & ATOMIC CHECKPOINT MANAGEMENT
# ============================================================================
@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_corpus: Optional[MemmapCorpus],
    eval_batches: int = 8,
    device: str = "cpu",
) -> float:
    if val_corpus is None:
        return float("nan")
    model.eval()
    total_ce = 0.0
    valid_batches = 0
    for i in range(eval_batches):
        bx, by = val_corpus.sample_batch(batch_size=1, deterministic_idx=i * 19)
        bx, by = bx.to(device), by.to(device)
        _, ce, _ = model(bx, labels=by)
        if torch.isfinite(ce):
            total_ce += float(ce.item())
            valid_batches += 1
    model.train()
    return total_ce / max(1, valid_batches)


def save_checkpoint_bundle(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: QuillanOniConfig,
    step: int,
    train_ce: float,
    val_ce: float,
    best_val_ce: float,
    n_train: int,
    save_path: Path,
    include_optimizer: bool = True,
) -> None:
    sd_export = canonical_state_dict_for_save(model)
    payload: Dict[str, Any] = {
        "step": step,
        "model_state_dict": sd_export,
        "config": cfg.__dict__,
        "trainable_params": n_train,
        "train_ce": train_ce,
        "val_ce": val_ce,
        "best_val_ce": best_val_ce,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    tmp_path = save_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    try:
        os.replace(tmp_path, save_path)
    except (PermissionError, OSError):
        if save_path.exists():
            try:
                save_path.unlink()
            except OSError:
                pass
        try:
            tmp_path.rename(save_path)
        except OSError:
            pass


def rotate_step_checkpoints(out_dir: Path, keep_last: int = 3) -> None:
    step_files = sorted(
        out_dir.glob("quillan_sft_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
    )
    if len(step_files) > keep_last:
        for old_ckpt in step_files[:-keep_last]:
            try:
                old_ckpt.unlink()
            except OSError:
                pass


# ============================================================================
# 6. MAIN TRAINING ORCHESTRATOR
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quillan-Ronin v5.4.0-ONI Hardened SFT Runner (Rev 2)")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimizer steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-6, help="Minimum cosine learning rate")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Linear warmup steps")
    parser.add_argument("--eval-every", type=int, default=50, help="Validation interval")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--log-every", type=int, default=5, help="Console & JSONL telemetry interval")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for i5-7500")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest/interrupt checkpoint")
    parser.add_argument("--resume-from", type=str, default="", help="Explicit checkpoint path to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_host_environment(threads=args.threads, low_priority=True)
    device_obj = torch.device(args.device)

    base_ckpt_path = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_coherent_base_ternary.pt")
    data_dir = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\data_bin")
    out_dir = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "sft_telemetry.jsonl"

    print("=" * 78)
    print("QUILLAN-RONIN v5.4.0-ONI — HARDENED SOVEREIGN SFT LAUNCHER (REV 2)")
    print(f"  Device:   {args.device.upper()} (Threads: {args.threads}, Denormal Flush: ON, Priority: BELOW_NORMAL)")
    print(f"  Steps:    {args.steps} | Micro-Batch: {args.batch_size} | Grad Accum: {args.grad_accum} ({args.batch_size * args.grad_accum * args.seq_len:,} tok/step)")
    print(f"  Shields:  Bounded Aux-Loss + Two-Stage Grad Clipping (5.0 Tensor / 1.0 Global) + ES-EMA")
    print("=" * 78, flush=True)

    # 1. Mount Train + Val Memmap Datasets
    print("\n[1/5] Mounting Zero-Copy Binary Memmap Datasets...", flush=True)
    train_corpus = MemmapCorpus(data_dir, split="train", seq_len=args.seq_len, seed=42)
    val_corpus: Optional[MemmapCorpus] = None
    if (data_dir / "val_ids.bin").exists() and (data_dir / "val_labels.bin").exists():
        val_corpus = MemmapCorpus(data_dir, split="val", seq_len=args.seq_len, seed=1337)
        print(f"  Train: {train_corpus.n_seqs:,} seqs ({train_corpus.total_tokens / 1e6:.2f}M tokens)")
        print(f"  Val:   {val_corpus.n_seqs:,} seqs ({val_corpus.total_tokens / 1e6:.2f}M tokens)", flush=True)
    else:
        print(f"  Train: {train_corpus.n_seqs:,} seqs ({train_corpus.total_tokens / 1e6:.2f}M tokens)", flush=True)

    # 2. Determine Checkpoint Source
    resume_ckpt_path: Optional[Path] = None
    if args.resume_from:
        resume_ckpt_path = Path(args.resume_from)
    elif args.resume:
        for cand in [out_dir / "quillan_sft_interrupt.pt", out_dir / "quillan_sft_latest.pt"]:
            if cand.exists():
                resume_ckpt_path = cand
                break

    load_target = resume_ckpt_path if (resume_ckpt_path and resume_ckpt_path.exists()) else base_ckpt_path
    print(f"\n[2/5] Loading 1,441-Tensor Sovereign Checkpoint: {load_target.name}...", flush=True)
    ck = torch.load(load_target, map_location="cpu", weights_only=False)
    raw_sd = ck.get("model_state_dict") or ck.get("model") or ck
    cfg_dict = ck.get("config", {})
    cfg = QuillanOniConfig(**cfg_dict) if isinstance(cfg_dict, dict) else QuillanOniConfig()
    cfg.grad_checkpoint = False
    cfg.max_seq_len = args.seq_len
    cfg.router_mode = "dense_pull"

    model = QuillanRoninOni(cfg)
    missing, unexpected = model.load_state_dict(raw_sd, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, f"Key mismatch! Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    vectorize_model_in_place(model)
    model.to(device_obj)
    print("  1,441 / 1,441 tensors verified (0 missing, 0 unexpected). Vectorized einsum active.", flush=True)

    # 3. Isolate Frozen Base (492.12M) vs Trainable Adapters (70.70M)
    print("\n[3/5] Partitioning Frozen Base & Trainable Council Adapters...", flush=True)
    for p in model.parameters():
        p.requires_grad = False

    trainable_patterns = (
        "pull_gate", "moe_gate", "router", "vec_experts", "experts", "lora_", "swarm_",
        "clone_coupling", "ln_f", "mem_norm", "moe_stabilizer",
    )
    named_trainable: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if any(pat in name for pat in trainable_patterns):
            p.requires_grad = True
            named_trainable.append((name, p))

    trainable_params = [p for _, p in named_trainable]
    n_train = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Frozen Base Ternary Backbone : {n_frozen / 1e6:.2f}M parameters")
    print(f"  Active Council/Swarm Adapters: {n_train / 1e6:.2f}M parameters ({len(named_trainable)} tensors)", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    es_anchor = ESAtScaleForgettingAnchor(trainable_params, ema_decay=0.995, mu_mem=0.001, interval=5)

    start_step = 1
    best_val_ce = float("inf")
    if resume_ckpt_path and "optimizer_state_dict" in ck:
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_step = int(ck.get("step", 0)) + 1
        best_val_ce = float(ck.get("best_val_ce", float("inf")))
        print(f"  >>> Resumed optimizer & state from Step {start_step - 1} (Best Val CE: {best_val_ce:.4f})", flush=True)

    # 4. Graceful Ctrl+C Signal Trap
    stop_requested = {"flag": False}

    def _signal_handler(sig: int, _frame: Any) -> None:
        print(f"\n[!] Signal {sig} received — finishing current step and saving emergency checkpoint...", flush=True)
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 5. Initial Baseline Validation
    print("\n[4/5] Running Initial Baseline Validation...", flush=True)
    init_val_ce = evaluate_validation(model, val_corpus, eval_batches=4, device=args.device)
    if math.isfinite(init_val_ce) and init_val_ce < best_val_ce:
        best_val_ce = init_val_ce
    print(f"  Baseline Validation CE: {init_val_ce:.4f}", flush=True)

    # 6. Main Phased Training Loop
    print("\n[5/5] Engaging Sovereign SFT Training Loop...", flush=True)
    model.train()
    t_run_start = time.time()
    rolling_ce = 0.0
    rolling_aux = 0.0
    rolling_ent = 0.0
    rolling_count = 0
    nan_skips = 0

    for step in range(start_step, args.steps + 1):
        t_step = time.time()
        cur_lr = get_lr(step - 1, args.steps, args.lr, args.min_lr, args.warmup_steps)
        aux_alpha, stage_name = get_staged_aux_alpha(step, args.steps)
        model.cfg.aux_load_weight = aux_alpha
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_ce = 0.0
        step_aux = 0.0
        step_ent = 0.0
        valid_micros = 0

        for _micro in range(args.grad_accum):
            bx, by = train_corpus.sample_batch(args.batch_size, min_targets=16)
            bx, by = bx.to(device_obj), by.to(device_obj)
            _, ce, aux_dict = model(bx, labels=by)

            # REV-2 FIX: Use bounded, differentiable compute_safe_aux_loss instead of raw total_aux_loss
            aux_loss = compute_safe_aux_loss(aux_dict, aux_alpha=aux_alpha, device=device_obj)
            total_loss = (ce + aux_loss) / args.grad_accum

            if not torch.isfinite(total_loss):
                nan_skips += 1
                continue

            total_loss.backward()
            step_ce += float(ce.item())
            step_aux += float(aux_loss.item())
            if "entropy" in aux_dict and torch.is_tensor(aux_dict["entropy"]):
                step_ent += float(aux_dict["entropy"].item())
            valid_micros += 1

        if valid_micros > 0:
            # REV-2 FIX: Two-Stage Per-Tensor + Global Gradient Clipping
            gn, max_t_name, max_t_norm = clip_gradients_two_stage(
                named_trainable, per_tensor_max_norm=5.0, global_max_norm=1.0
            )
            if math.isfinite(gn):
                optimizer.step()
                es_anchor.step(step)
            else:
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)

            step_ce /= valid_micros
            step_aux /= valid_micros
            step_ent /= valid_micros
        else:
            gn, max_t_name, max_t_norm = 0.0, "none", 0.0

        step_dt = time.time() - t_step
        rolling_ce += step_ce
        rolling_aux += step_aux
        rolling_ent += step_ent
        rolling_count += 1

        # Console & JSONL Logging
        if step % args.log_every == 0 or step == 1:
            avg_ce = rolling_ce / max(1, rolling_count)
            avg_aux = rolling_aux / max(1, rolling_count)
            avg_ent = rolling_ent / max(1, rolling_count)
            rolling_ce = rolling_aux = rolling_ent = 0.0
            rolling_count = 0
            elapsed_min = (time.time() - t_run_start) / 60.0
            eta_min = ((args.steps - step) * step_dt) / 60.0
            print(
                f"  [{stage_name}] Step {step:4d}/{args.steps} | "
                f"CE: {avg_ce:.4f} | Aux: {avg_aux:.5f} | Ent: {avg_ent:.3f} | "
                f"GradNorm: {gn:.3f} (MaxT: {max_t_norm:.2f}) | LR: {cur_lr:.2e} | "
                f"{step_dt:.1f}s/st | ETA: {eta_min:.1f}m",
                flush=True,
            )
            log_record = {
                "step": step,
                "stage": stage_name,
                "train_ce": round(avg_ce, 5),
                "aux_loss": round(avg_aux, 6),
                "council_entropy": round(avg_ent, 4),
                "grad_norm": round(gn, 4),
                "max_tensor_grad_norm": round(max_t_norm, 4),
                "max_grad_tensor": max_t_name,
                "lr": cur_lr,
                "aux_alpha": aux_alpha,
                "step_sec": round(step_dt, 2),
                "nan_skips": nan_skips,
            }
            with open(telemetry_path, "a", encoding="utf-8") as tf:
                tf.write(json.dumps(log_record) + "\n")

        # Validation & Best Model Tracking
        val_ce = float("nan")
        if step % args.eval_every == 0 or step == args.steps:
            val_ce = evaluate_validation(model, val_corpus, eval_batches=6, device=args.device)
            is_best = math.isfinite(val_ce) and (val_ce < best_val_ce)
            if is_best:
                best_val_ce = val_ce
                best_path = out_dir / "quillan_sft_best.pt"
                save_checkpoint_bundle(
                    model,
                    optimizer,
                    cfg,
                    step,
                    step_ce,
                    val_ce,
                    best_val_ce,
                    n_train,
                    best_path,
                    include_optimizer=False,
                )
            print(
                f"  >>> [VAL AUDIT @ Step {step}] Val CE: {val_ce:.4f} "
                f"(Best: {best_val_ce:.4f}) {'[★ NEW BEST SAVED]' if is_best else ''}",
                flush=True,
            )

        # Periodic Milestone & Latest Checkpoint Saving
        if step % args.save_every == 0 or step == args.steps:
            step_path = out_dir / f"quillan_sft_step_{step}.pt"
            latest_path = out_dir / "quillan_sft_latest.pt"
            save_checkpoint_bundle(
                model,
                optimizer,
                cfg,
                step,
                step_ce,
                val_ce,
                best_val_ce,
                n_train,
                step_path,
                include_optimizer=False,
            )
            save_checkpoint_bundle(
                model,
                optimizer,
                cfg,
                step,
                step_ce,
                val_ce,
                best_val_ce,
                n_train,
                latest_path,
                include_optimizer=True,
            )
            rotate_step_checkpoints(out_dir, keep_last=3)
            print(f"  >>> [CHECKPOINT] Saved {step_path.name} + updated quillan_sft_latest.pt", flush=True)

        # Graceful Interrupt Exit
        if stop_requested["flag"]:
            int_path = out_dir / "quillan_sft_interrupt.pt"
            save_checkpoint_bundle(
                model,
                optimizer,
                cfg,
                step,
                step_ce,
                val_ce,
                best_val_ce,
                n_train,
                int_path,
                include_optimizer=True,
            )
            print(f"\n[SAFE EXIT] Saved interrupt state to {int_path}. Resume anytime with `--resume`.", flush=True)
            return

    total_min = (time.time() - t_run_start) / 60.0
    print("\n" + "=" * 78)
    print(f"TRAINING COMPLETE ({args.steps} Steps in {total_min:.2f} minutes).")
    print(f"  Best Validation CE : {best_val_ce:.4f} -> {out_dir / 'quillan_sft_best.pt'}")
    print(f"  Latest Checkpoint  : {out_dir / 'quillan_sft_latest.pt'}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
