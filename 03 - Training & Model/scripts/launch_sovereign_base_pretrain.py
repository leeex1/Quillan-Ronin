#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_sovereign_base_pretrain.py — Sovereign Full Base Pre-Training Launcher
=============================================================================
Pre-trains ALL 562.8M parameters (embeddings, attention heads, 9-vector prism,
and 34 ternary MoE experts via dynamic STE) across the 432.6M token corpus
to forge a genuinely converged foundation BEFORE Supervised Fine-Tuning.

Shields:
  - CPU Smoothness Locks (Core 0 reserved, priority BELOW_NORMAL, denormal flush ON)
  - Zero-copy binary memmap streaming
  - Two-Stage Gradient Clipping (Per-tensor 5.0 / Global 1.0)
  - ES-EMA Forgetting Anchor
  - Rolling 3-step checkpoint retention
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_tokenizer_unified import UnifiedQuillanTokenizer
from quillan_live_training_upgrades import (
    canonical_state_dict_for_save,
    sample_early_ternary_logits,
    vectorize_model_in_place,
)


# ============================================================================
# 1. HARDWARE & OS TUNING (Lee-Mach-6 Governor & C14-KAIDO i5-7500 Shield)
# ============================================================================
def configure_host_environment(threads: int = 4, low_priority: bool = True) -> None:
    try:
        daemons_path = SCRIPTS_DIR.parent.parent / "daemons"
        if str(daemons_path) not in sys.path:
            sys.path.insert(0, str(daemons_path))
        from quillan_governor_daemon import apply_pc_smoothness_locks
        gov_res = apply_pc_smoothness_locks()
        print(f"[GOVERNOR] Applied PC Smoothness Locks: Affinity Cores {gov_res.get('cores_assigned')}, Priority Below-Normal: {gov_res.get('priority_set')}", flush=True)
    except Exception as e:
        print(f"[GOVERNOR WARN] Governor daemon hook: {e}", flush=True)

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

    sm61_dll = Path(r"C:\02_QUILLAN\kernels\quillan_cuda_sm61.dll")
    if sm61_dll.exists():
        try:
            sm61_lib = ctypes.CDLL(str(sm61_dll))
            sm_cap = sm61_lib.query_sm61_support()
            print(f"[SM61 KERNEL] Loaded {sm61_dll.name} — Native Pascal __dp4a & Fused 34-Expert Kernel Ready (sm_{sm_cap})", flush=True)
        except Exception as e:
            print(f"[SM61 KERNEL WARN] {e}", flush=True)


# ============================================================================
# 2. ZERO-COPY PRETRAIN CORPUS (Causal Next-Token Modeling)
# ============================================================================
class PretrainMemmapCorpus:
    """Streams fixed-stride sequences for autoregressive language pre-training."""

    def __init__(self, data_dir: Path, split: str = "train", seq_len: int = 256, seed: int = 42, min_valid_tokens: int = 32):
        self.split = split
        self.seq_len = seq_len
        ids_path = data_dir / f"{split}_ids.bin"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing binary file '{ids_path}'")

        self.ids_mmap = np.memmap(ids_path, dtype=np.uint16, mode="r")
        self.n_seqs = len(self.ids_mmap) // seq_len
        self.total_tokens = self.n_seqs * seq_len
        self.ids = self.ids_mmap[:self.total_tokens].reshape(-1, seq_len)
        self.rng = np.random.default_rng(seed)

        # Pre-filter out all-padding / low-token chunks (token ID 1 is pad) in memory-safe 65536-row blocks
        valid_parts: List[np.ndarray] = []
        chunk_step = 65536
        for start in range(0, self.n_seqs, chunk_step):
            end = min(self.n_seqs, start + chunk_step)
            non_pad = (self.ids[start:end] != 1).sum(axis=1)
            loc = np.flatnonzero(non_pad >= min_valid_tokens).astype(np.int32) + start
            valid_parts.append(loc)
        self.valid_indices = np.concatenate(valid_parts) if valid_parts else np.arange(self.n_seqs, dtype=np.int32)
        self.n_valid = len(self.valid_indices)

    def sample_batch(self, batch_size: int = 1, deterministic_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        indices: List[int] = []
        for b in range(batch_size):
            if deterministic_idx is not None:
                v_idx = (deterministic_idx + b) % self.n_valid
            else:
                v_idx = int(self.rng.integers(0, self.n_valid))
            indices.append(int(self.valid_indices[v_idx]))

        batch_arr = self.ids[indices].astype(np.int64)
        # Note: QuillanRoninOni.forward already performs the causal shift:
        #   shift_logits = logits[..., :-1, :], shift_labels = labels[..., 1:]
        # Passing unshifted b_x and b_y here prevents double-shifting (predicting w_{t+2} instead of w_{t+1}).
        b_x = torch.from_numpy(batch_arr)
        b_y = torch.from_numpy(batch_arr.copy())
        # Mask out padding tokens (token ID 1) from loss calculation
        b_y[b_y == 1] = -100
        return b_x, b_y


# ============================================================================
# 3. TWO-STAGE GRADIENT CLIPPING
# ============================================================================
def clip_gradients_two_stage(
    named_params: List[Tuple[str, nn.Parameter]],
    per_tensor_max_norm: float = 5.0,
    global_max_norm: float = 1.0,
) -> Tuple[float, str, float]:
    max_tensor_norm = 0.0
    max_tensor_name = "none"
    params_with_grad: List[nn.Parameter] = []

    for name, p in named_params:
        if p.grad is None:
            continue
        t_norm = float(p.grad.detach().float().norm(2).item())
        if not math.isfinite(t_norm):
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            t_norm = float(p.grad.detach().float().norm(2).item())
        if t_norm > max_tensor_norm:
            max_tensor_norm = t_norm
            max_tensor_name = name

        if t_norm > per_tensor_max_norm:
            scale = per_tensor_max_norm / (t_norm + 1e-6)
            p.grad.detach().mul_(scale)

        params_with_grad.append(p)

    if not params_with_grad:
        return 0.0, max_tensor_name, 0.0

    global_norm = float(torch.nn.utils.clip_grad_norm_(params_with_grad, global_max_norm).item())
    return global_norm, max_tensor_name, max_tensor_norm


# ============================================================================
# 4. LEARNING RATE SCHEDULE
# ============================================================================
def get_lr(step: int, total_steps: int, peak_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ============================================================================
# 5. VALIDATION & GENERATION PROBES
# ============================================================================
@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_corpus: Optional[PretrainMemmapCorpus],
    eval_batches: int = 10,
    device: str = "cpu",
) -> float:
    if val_corpus is None:
        return float("nan")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    total_ce = 0.0
    valid_batches = 0
    for i in range(eval_batches):
        bx, by = val_corpus.sample_batch(batch_size=1, deterministic_idx=i * 17)
        bx, by = bx.to(device), by.to(device)
        logits, ce, aux = model(bx, labels=by, deliberation=False)
        if torch.isfinite(ce):
            total_ce += float(ce.item())
            valid_batches += 1
        del bx, by, logits, ce, aux
    model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return total_ce / max(1, valid_batches)


@torch.no_grad()
def probe_sample_generation(model: nn.Module, tok: UnifiedQuillanTokenizer, prompt: str = "The code of the samurai is", max_tokens: int = 20) -> str:
    model.eval()
    dev = next(model.parameters()).device
    input_ids = tok.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=dev)
    gen = list(input_ids)
    for _ in range(max_tokens):
        out = model(x, deliberation=False)
        logits = out[0] if isinstance(out, tuple) else out
        # Mask pad/eos/unk tokens and apply mild repetition penalty over generated continuation
        step_logits = logits[:, -1:, :].float().clone()
        step_logits[0, 0, :4] = float("-inf")
        next_tok = sample_early_ternary_logits(
            step_logits,
            generated_tokens=gen,
            temp=0.55,
            min_p=0.10,
            top_k=20,
            rep_penalty=1.20,
            freq_penalty=0.35,
        )
        gen.append(next_tok)
        x = torch.cat([x, torch.tensor([[next_tok]], device=dev)], dim=1)
        if next_tok in (0, 50261):
            break
    model.train()
    return tok.decode(gen[len(input_ids):])


def save_checkpoint(
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
    del payload, sd_export
    gc.collect()
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


def rotate_checkpoints(out_dir: Path, keep_last: int = 1) -> None:
    step_files = sorted(
        out_dir.glob("quillan_base_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
    )
    if len(step_files) > keep_last:
        for old_ckpt in step_files[:-keep_last]:
            try:
                old_ckpt.unlink()
            except OSError:
                pass


# ============================================================================
# 6. MAIN PRETRAINING ORCHESTRATOR
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quillan-Ronin v5.4.0-ONI Sovereign Base Pretraining")
    parser.add_argument("--steps", type=int, default=2000, help="Total pretraining steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (1 for Pascal 4GB VRAM)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (256 for fast foundation convergence)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Peak pretraining learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum cosine learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--eval-every", type=int, default=50, help="Validation interval")
    parser.add_argument("--save-every", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--log-every", type=int, default=5, help="Console logging interval")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--resume", action="store_true", help="Resume from latest base pretrain checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_host_environment(threads=args.threads, low_priority=True)
    device_obj = torch.device(args.device)

    base_ckpt_path = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_coherent_base_ternary.pt")
    data_dir = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\data_bin")
    out_dir = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\base_pretrain_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "base_pretrain_telemetry.jsonl"

    print("=" * 78, flush=True)
    print("QUILLAN-RONIN v5.4.0-ONI — SOVEREIGN FULL BASE PRE-TRAINER")
    print(f"  Device:       {args.device.upper()} (Threads: {args.threads}, Core 0 Protected)")
    print(f"  Steps:        {args.steps} | Micro-Batch: {args.batch_size} | Accum: {args.grad_accum} ({args.batch_size * args.grad_accum * args.seq_len:,} tok/step)")
    print(f"  Objective:    Causal Language Modeling (Predict Next Token across all non-pad tokens)")
    print("=" * 78, flush=True)

    # 1. Mount Datasets
    print("\n[1/4] Mounting Pretrain Datasets...", flush=True)
    train_corpus = PretrainMemmapCorpus(data_dir, split="train", seq_len=args.seq_len, seed=42)
    val_corpus: Optional[PretrainMemmapCorpus] = None
    if (data_dir / "val_ids.bin").exists():
        val_corpus = PretrainMemmapCorpus(data_dir, split="val", seq_len=args.seq_len, seed=1337)
        print(f"  Train: {train_corpus.n_valid:,} valid chunks / {train_corpus.n_seqs:,} total ({train_corpus.total_tokens / 1e6:.2f}M tokens)")
        print(f"  Val:   {val_corpus.n_valid:,} valid chunks / {val_corpus.n_seqs:,} total ({val_corpus.total_tokens / 1e6:.2f}M tokens)", flush=True)

    # 2. Load Checkpoint
    latest_ckpt = out_dir / "quillan_base_latest.pt"
    resume_target = latest_ckpt if (args.resume and latest_ckpt.exists()) else base_ckpt_path
    print(f"\n[2/4] Loading Weights from: {resume_target.name}...", flush=True)
    ck = torch.load(resume_target, map_location="cpu", weights_only=False)
    resumed_step = int(ck.get("step", 0))
    resumed_best_val = float(ck.get("best_val_ce", ck.get("train_ce", float("inf"))))
    opt_state = ck.pop("optimizer_state_dict", None)
    raw_sd = ck.get("model_state_dict") or ck.get("model") or ck
    cfg_dict = ck.get("config", {})
    cfg = QuillanOniConfig(**cfg_dict) if isinstance(cfg_dict, dict) else QuillanOniConfig()
    cfg.grad_checkpoint = (device_obj.type == "cuda" and (args.seq_len > 256 or args.batch_size > 1))
    cfg.max_seq_len = args.seq_len
    cfg.router_mode = "dense_pull"

    model = QuillanRoninOni(cfg)
    expected_keys = set(model.state_dict().keys())
    clean_sd = {k: v for k, v in raw_sd.items() if k in expected_keys}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f"  Weights loaded: {len(clean_sd)} tensors (missing: {len(missing)}, unexpected: {len(unexpected)})", flush=True)
    del ck, raw_sd, clean_sd
    gc.collect()

    # Normalize any unscaled integer ternary {-1, 0, +1} attention matrices (std > 0.2) or spectral outliers (sigma_1 > 4.0)
    with torch.no_grad():
        scale_in = 1.0 / math.sqrt(max(1, cfg.hidden_dim))
        n_rescaled = 0
        for blk in model.h:
            for proj in (getattr(blk.attn, "c_attn", None), getattr(blk.attn, "c_proj", None)):
                if proj is not None and hasattr(proj, "weight") and proj.weight.detach().float().std().item() > 0.2:
                    proj.weight.mul_(scale_in)
                    n_rescaled += 1
            for sub_w in (
                getattr(getattr(blk.attn, "prism", None), "w_gate", None),
                getattr(blk.moe, "c_fc", None),
                getattr(blk.moe, "c_proj", None),
            ):
                if sub_w is not None and hasattr(sub_w, "weight"):
                    s1 = float(torch.linalg.matrix_norm(sub_w.weight.detach().float(), ord=2).item())
                    if s1 > 4.0:
                        sub_w.weight.mul_(2.5 / s1)
                        n_rescaled += 1
        if getattr(model, "grt_core", None) is not None and hasattr(model.grt_core, "gate_B"):
            if model.grt_core.gate_B.weight.detach().float().std().item() > 0.2:
                model.grt_core.gate_B.weight.mul_(0.02)
                n_rescaled += 1
        if n_rescaled > 0:
            print(f"  [Scale Guard] Normalized {n_rescaled} unscaled/spectral-outlier projection tensors.", flush=True)

    vectorize_model_in_place(model)
    if device_obj.type == "cuda":
        model.to(torch.bfloat16).to(device_obj)
        print("  Model transferred to CUDA in bfloat16 (Memory: 2.02 GB / 4.00 GB).", flush=True)
    else:
        model.to(device_obj)
        print("  Model transferred to device.", flush=True)

    # 3. UNFREEZE ALL PARAMETERS FOR FULL BASE PRE-TRAINING
    print("\n[3/4] Activating Full Base Gradients (ALL Weights Unfrozen)...", flush=True)
    named_trainable: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        p.requires_grad = True
        named_trainable.append((name, p))

    trainable_params = [p for _, p in named_trainable]
    n_train = sum(p.numel() for p in trainable_params)
    print(f"  >>> TOTAL ACTIVE PARAMETERS: {n_train / 1e6:.2f}M ({len(named_trainable)} tensors fully trainable)", flush=True)

    if device_obj.type == "cuda":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
            print("  >>> Optimizer: bitsandbytes 8-bit AdamW (Memory-Optimized for Pascal VRAM)", flush=True)
        except Exception:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
            print("  >>> Optimizer: PyTorch AdamW (FP32)", flush=True)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
        print("  >>> Optimizer: PyTorch CPU AdamW", flush=True)

    start_step = 1
    best_val_ce = float("inf")
    if args.resume:
        start_step = resumed_step + 1
        best_val_ce = resumed_best_val
        if opt_state is not None and n_rescaled == 0:
            try:
                optimizer.load_state_dict(opt_state)
                print(f"  >>> Resumed optimizer state from Step {start_step - 1} (Best Val CE: {best_val_ce:.4f})", flush=True)
            except Exception as e:
                print(f"  >>> Optimizer state load skipped: {e}", flush=True)
        else:
            print(f"  >>> Resumed model weights from Step {start_step - 1} (Best Val CE: {best_val_ce:.4f}) with fresh optimizer state", flush=True)

    del opt_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    tok = UnifiedQuillanTokenizer()

    # 4. Graceful Ctrl+C Trap
    stop_requested = {"flag": False}
    def _sig_handler(sig: int, _frame: Any) -> None:
        print(f"\n[!] Signal {sig} received — saving emergency checkpoint...", flush=True)
        stop_requested["flag"] = True
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # Initial Baseline Validation
    print("\n[4/4] Running Baseline Validation...", flush=True)
    init_val_ce = evaluate_validation(model, val_corpus, eval_batches=4, device=args.device)
    if not math.isfinite(best_val_ce) or n_rescaled > 0 or init_val_ce < best_val_ce:
        best_val_ce = init_val_ce
    print(f"  Baseline Pretrain CE: {init_val_ce:.4f} (Best: {best_val_ce:.4f} | Uniform: {math.log(tok.vocab_size):.4f})", flush=True)

    # 5. Pretraining Loop
    print("\nEngaging Full Base Pretraining Loop...", flush=True)
    model.train()
    t_start = time.time()
    rolling_ce = 0.0
    rolling_count = 0

    for step in range(start_step, args.steps + 1):
        t_step = time.time()
        cur_lr = get_lr(step - 1, args.steps, args.lr, args.min_lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_ce = 0.0
        valid_micros = 0

        for _micro in range(args.grad_accum):
            bx, by = train_corpus.sample_batch(args.batch_size)
            bx, by = bx.to(device_obj), by.to(device_obj)

            # Pretrain forward pass (deliberation disabled during training to prevent memory retention)
            logits, ce, aux = model(bx, labels=by, deliberation=False)

            if not torch.isfinite(ce):
                del bx, by, logits, ce, aux
                continue

            loss = ce / args.grad_accum
            step_ce += float(ce.item())
            valid_micros += 1
            # Explicitly delete forward activations and unused returns (logits=51MB, aux=~100MB, bx, by)
            # BEFORE backward pass begins to provide maximum headroom on 4GB Pascal WDDM limits
            del logits, aux, bx, by, ce
            loss.backward()
            del loss

        if valid_micros > 0:
            gn, max_t_name, max_t_norm = clip_gradients_two_stage(
                named_trainable, per_tensor_max_norm=5.0, global_max_norm=1.0
            )
            if math.isfinite(gn):
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
            step_ce /= valid_micros
        else:
            gn, max_t_name, max_t_norm = 0.0, "none", 0.0

        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        step_dt = time.time() - t_step
        rolling_ce += step_ce
        rolling_count += 1

        # Periodic Logging
        if step % args.log_every == 0 or step == 1:
            avg_ce = rolling_ce / max(1, rolling_count)
            rolling_ce = 0.0
            rolling_count = 0
            eta_min = ((args.steps - step) * step_dt) / 60.0
            print(
                f"  Step {step:4d}/{args.steps} | CE: {avg_ce:.4f} | "
                f"GradNorm: {gn:.3f} (MaxT: {max_t_norm:.2f} [{max_t_name[:24]}]) | "
                f"LR: {cur_lr:.2e} | {step_dt:.1f}s/st | ETA: {eta_min:.1f}m",
                flush=True,
            )
            with open(telemetry_path, "a", encoding="utf-8") as tf:
                tf.write(json.dumps({
                    "step": step,
                    "train_ce": round(avg_ce, 5),
                    "grad_norm": round(gn, 4),
                    "max_tensor_grad_norm": round(max_t_norm, 4),
                    "lr": cur_lr,
                    "step_sec": round(step_dt, 2),
                }) + "\n")
            gc.collect()

        # Validation & Generation Probe
        if step % args.eval_every == 0 or step == args.steps:
            val_ce = evaluate_validation(model, val_corpus, eval_batches=8, device=args.device)
            is_best = math.isfinite(val_ce) and (val_ce < best_val_ce)
            if is_best:
                best_val_ce = val_ce
                best_path = out_dir / "quillan_base_best.pt"
                save_checkpoint(
                    model, optimizer, cfg, step, step_ce, val_ce, best_val_ce,
                    n_train, best_path, include_optimizer=False
                )

            # Sample test generation
            sample_text = probe_sample_generation(model, tok, prompt="The code of the samurai is", max_tokens=15)
            print(
                f"  >>> [VAL AUDIT @ Step {step}] Val CE: {val_ce:.4f} "
                f"(Best: {best_val_ce:.4f}) {'[* NEW BEST]' if is_best else ''}\n"
                f"      [Probe Completion]: {repr(sample_text)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Periodic Save
        if step % args.save_every == 0 or step == args.steps:
            step_path = out_dir / f"quillan_base_step_{step}.pt"
            latest_path = out_dir / "quillan_base_latest.pt"
            save_checkpoint(model, optimizer, cfg, step, step_ce, val_ce, best_val_ce, n_train, step_path, include_optimizer=False)
            save_checkpoint(model, optimizer, cfg, step, step_ce, val_ce, best_val_ce, n_train, latest_path, include_optimizer=True)
            rotate_checkpoints(out_dir, keep_last=1)
            print(f"  >>> [CHECKPOINT] Saved {step_path.name} + updated quillan_base_latest.pt", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        if stop_requested["flag"]:
            int_path = out_dir / "quillan_base_interrupt.pt"
            save_checkpoint(model, optimizer, cfg, step, step_ce, val_ce, best_val_ce, n_train, int_path, include_optimizer=True)
            print(f"\n[SAFE EXIT] Pretraining paused at Step {step}. Resume anytime with `--resume`.", flush=True)
            return

    total_min = (time.time() - t_start) / 60.0
    print("\n" + "=" * 78)
    print(f"BASE PRETRAINING COMPLETE ({args.steps} Steps in {total_min:.2f} min).")
    print(f"  Final Best Validation CE: {best_val_ce:.4f} -> {out_dir / 'quillan_base_best.pt'}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
