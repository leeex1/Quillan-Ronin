#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI — PRODUCTION CONTINUED PRE-TRAINING (CPT) ENGINE
==========================================================================
Resumes training directly from existing checkpoints (quillan_6l_ma_best.pt or
quillan_12l_ma_best.pt) without resetting weights.

Features:
  1. Checkpoint Resumption: Strict 100% parameter match for BitNet 1.58b STE.
  2. Anti-Forgetting Replay Buffer: Interleaves 10% gold SFT samples (15,631 samples)
     to preserve the rōnin persona, formatting, and matched-pair reasoning.
  3. Gradient Stabilization: Linear 500-step warm-up + cosine decay with grad clipping.
  4. Atomic Checkpointing: Safe saving (.tmp -> .pt) preventing state corruption.
  5. Multi-Device Compatibility: CPU (AVX2 optimized) or Pascal/CUDA.

Usage:
  python scripts/run_quillan_cpt_training.py --model mini --steps 5000 --batch-size 4 --lr 8e-5
"""

import os
import sys
import gc
import time
import math
import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Paths & Setup ─────────────────────────────────────────────────────────────
# NOTE (2026-10-02): the vault was reorganised so that everything moved under
# "03 - Training & Model". The old REPO_ROOT/scripts, /checkpoints and
# /training_data roots no longer exist. Resolve post-reorg paths, keeping the
# legacy roots as a fallback so this runs on either layout.
REPO_ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = REPO_ROOT / "03 - Training & Model"


def _resolve(*candidates: Path) -> Path:
    """Return the first existing candidate, else the first candidate."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


SCRIPTS_DIR = _resolve(MODEL_DIR / "scripts", REPO_ROOT / "scripts")
TRAIN_DATA_DIR = _resolve(MODEL_DIR / "training_data", REPO_ROOT / "training_data")
CKPT_DIR = _resolve(
    MODEL_DIR / "checkpoints" / "checkpoints_oni",
    REPO_ROOT / "checkpoints" / "checkpoints_oni",
)

for _p in (SCRIPTS_DIR, MODEL_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("QuillanCPT")


# ── Dataset & Replay Interleaver ──────────────────────────────────────────────

class CPTInterleavedDataset(Dataset):
    """Interleaves continuous domain knowledge tokens with gold persona replay samples."""

    def __init__(
        self,
        gold_archive_path: Path,
        raw_pt_path: Optional[Path] = None,
        seq_len: int = 256,
        replay_ratio: float = 0.15,
    ):
        self.seq_len = seq_len
        self.replay_ratio = replay_ratio

        # 1. Load Gold Replay Archive
        LOGGER.info("Loading Gold Replay Archive from: %s", gold_archive_path)
        gold_data = torch.load(gold_archive_path, map_location="cpu", weights_only=False)
        self.gold_tokens = gold_data["input_ids"].to(torch.int64)
        self.num_gold = self.gold_tokens.shape[0]
        LOGGER.info("Gold Replay Pool: %d sequences loaded (Persona Protection Active).", self.num_gold)

        # 2. Load Continuous Knowledge Corpus (if available)
        self.raw_tokens = None
        if raw_pt_path and raw_pt_path.exists():
            LOGGER.info("Loading Knowledge Corpus from: %s", raw_pt_path)
            raw_data = torch.load(raw_pt_path, map_location="cpu", weights_only=False)
            if isinstance(raw_data, dict):
                self.raw_tokens = raw_data.get("input_ids", next(iter(raw_data.values()))).to(torch.int64)
            else:
                self.raw_tokens = raw_data.to(torch.int64)
            LOGGER.info("Continuous Knowledge Pool: %d sequences loaded.", len(self.raw_tokens))
        else:
            LOGGER.warning("No secondary raw corpus specified; operating in expanded Gold Pre-Training mode.")

    def __len__(self) -> int:
        if self.raw_tokens is not None:
            return max(len(self.raw_tokens), self.num_gold * 2)
        return self.num_gold

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Interleave replay buffer probabilistically or systematically
        use_gold = (self.raw_tokens is None) or (torch.rand(1).item() < self.replay_ratio)

        if use_gold:
            g_idx = idx % self.num_gold
            seq = self.gold_tokens[g_idx]
        else:
            r_idx = idx % len(self.raw_tokens)
            seq = self.raw_tokens[r_idx]

        # Standard causal shift: input = seq[:-1], target = seq[1:]
        # Given a sequence of length L, input is first L-1 tokens, target is next L-1 tokens
        effective_len = min(len(seq) - 1, self.seq_len - 1)
        inp = seq[:effective_len]
        target = seq[1:effective_len + 1]

        target_len = self.seq_len - 1
        if len(inp) < target_len:
            pad_len = target_len - len(inp)
            inp = F.pad(inp, (0, pad_len), value=50256)
            target = F.pad(target, (0, pad_len), value=-100)

        return inp, target


# ── Learning Rate Scheduler ───────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 1e-6) -> float:
    """Linear warm-up followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Atomic Checkpointing ──────────────────────────────────────────────────────

def save_atomic_checkpoint(
    path: Path,
    model: nn.Module,
    config: QuillanOniConfig,
    step: int,
    loss: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Writes state to a temporary file before atomically renaming to prevent corruption."""
    tmp_path = path.with_suffix(".pt.tmp")
    ckpt_dict = {
        "step": step,
        "loss": loss,
        "config": config.__dict__,
        "model_state_dict": model.state_dict(),
        "timestamp": time.time(),
        "engine": "Quillan-Ronin v5.4.0-ONI CPT",
    }
    if optimizer is not None:
        ckpt_dict["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(ckpt_dict, tmp_path)
    if tmp_path.exists():
        tmp_path.replace(path)
    LOGGER.info("[CKPT] Saved checkpoint atomically: %s (Step %d, Loss %.4f)", path.name, step, loss)


# ── Main Training Loop ────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    LOGGER.info("Executing Continued Pre-Training on device: %s", device)

    # Resolve Checkpoint & Config
    # The original baselines (quillan_{6,12}l_ma_best.pt) are no longer on disk.
    # Walk a candidate list so CPT can resume from whatever baseline survives.
    ckpt_dir = CKPT_DIR
    if args.model.lower() == "main":
        candidates = [
            "quillan_12l_cpt_best.pt", "quillan_12l_clean_sft.pt",
            "quillan_12l_ma_best.pt", "step_0060_loss1.531.pt",
        ]
        out_ckpt = ckpt_dir / "quillan_12l_cpt_best.pt"
    else:
        candidates = [
            "quillan_6l_cpt_best.pt", "quillan_6l_clean_sft.pt",
            "quillan_6l_sft_v2.pt", "quillan_6l_ma_best.pt",
        ]
        out_ckpt = ckpt_dir / "quillan_6l_cpt_best.pt"

    if args.init_ckpt:
        in_ckpt = Path(args.init_ckpt)
    else:
        in_ckpt = next((ckpt_dir / c for c in candidates if (ckpt_dir / c).exists()), None)

    if in_ckpt is None or not in_ckpt.exists():
        raise FileNotFoundError(
            f"No source checkpoint found in {ckpt_dir}. Tried: {candidates}. "
            f"Use --init-ckpt <path> to point at a specific baseline."
        )
    LOGGER.info("Baseline resolved: %s", in_ckpt)

    LOGGER.info("Loading baseline checkpoint: %s", in_ckpt)
    ckpt_data = torch.load(in_ckpt, map_location="cpu", weights_only=False)
    state = ckpt_data["model_state_dict"]
    cfg_dict = dict(ckpt_data["config"])
    cfg_dict["device"] = args.device
    # Config-revision drift guard: this vault has SIX divergent copies of
    # quillan_v5_4_oni.py. If the active copy is not the one that saved the
    # checkpoint, unknown config keys raise TypeError and abort the run.
    # Drop them loudly so the drift is visible instead of fatal.
    valid_keys = {f.name for f in dataclasses.fields(QuillanOniConfig)}
    dropped = sorted(k for k in cfg_dict if k not in valid_keys)
    if dropped:
        LOGGER.warning(
            "Checkpoint config has %d key(s) this QuillanOniConfig does not define: %s",
            len(dropped), dropped,
        )
        LOGGER.warning("-> active quillan_v5_4_oni.py may differ from the revision that "
                       "saved this checkpoint. Continuing with the remaining keys.")
    cfg = QuillanOniConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})

    # Gradient checkpointing is already implemented in the model (cfg.grad_checkpoint,
    # consumed in QuillanRoninOni.forward) but was never switched on for CPT.
    # It trades ~30% step time for a large reduction in activation memory, which is
    # what makes the wider unfreeze below fit on a 4 GB card.
    if args.grad_checkpoint:
        cfg.grad_checkpoint = True
        LOGGER.info("Gradient checkpointing ENABLED (activation memory reduced).")

    LOGGER.info("Instantiating QuillanRoninOni architecture (Layers: %d, Experts: %d)...", cfg.n_layer, cfg.num_experts)
    model = QuillanRoninOni(cfg).to(device)
    incompatible = model.load_state_dict(state, strict=True)
    LOGGER.info("Checkpoint state_dict loaded with 100%% strict tensor alignment.")

    # ── Parameter Trainability Strategy ──────────────────────────────────────
    # WHY THIS CHANGED (2026-10-02):
    # The previous GPU path auto-froze everything except
    #   ("expert", "router", "ln", "norm", "gate", "w_gate")
    # which meant self-attention (c_attn/c_proj) and the embedding table (wte)
    # NEVER trained. Attention is where syntax lives, so the model learned topic
    # content while drifting and repeating. The widened set below unfreezes the
    # components that govern grammar, at the cost of ~2x the optimizer footprint.
    #
    # To pay for that on a 4 GB card, frozen weights are stored in fp16
    # (the same trick already used in train_expert_cycle.py). 812M frozen fp32
    # weights cost ~3.25 GB; in fp16 they cost ~1.62 GB, freeing ~1.6 GB.
    if args.unfreeze_set == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif args.unfreeze_set == "legacy":
        for n, p in model.named_parameters():
            p.requires_grad = any(
                k in n for k in ("expert", "router", "ln", "norm", "gate", "w_gate")
            )
    else:  # "wide" (default): legacy set + attention + embeddings + prism/bridge
        UNFREEZE_KEYS = (
            "expert", "router", "ln", "norm", "gate", "w_gate",
            "attn", "c_attn", "c_proj", "wte", "prism", "bridge",
            "finalizer", "lora",
        )
        for n, p in model.named_parameters():
            p.requires_grad = any(k in n for k in UNFREEZE_KEYS)

    # ── Frozen weights -> fp16 (frees VRAM; requires autocast in the step) ───
    half_frozen = False
    if args.half_frozen and device.type == "cuda":
        freed_bytes = 0
        for n, p in model.named_parameters():
            if not p.requires_grad and p.dtype == torch.float32:
                freed_bytes += p.numel() * 2
                p.data = p.data.half()
        half_frozen = True
        LOGGER.info("Frozen weights cast to fp16 — reclaimed ~%.2f GB of VRAM.", freed_bytes / 1e9)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # fp32 AdamW costs 16 bytes/param: 4 weight + 4 grad + 8 (m, v)
    est_opt_gb = trainable_params * 16 / 1e9
    est_frozen_gb = frozen_params * (2 if half_frozen else 4) / 1e9
    LOGGER.info(
        "Trainable: %s (%.1fM) | Frozen: %s (%.1fM) | unfreeze_set=%s | half_frozen=%s",
        f"{trainable_params:,}", trainable_params / 1e6,
        f"{frozen_params:,}", frozen_params / 1e6,
        args.unfreeze_set, half_frozen,
    )
    LOGGER.info(
        "Estimated VRAM: optimizer+grads ~%.2f GB + frozen weights ~%.2f GB = ~%.2f GB (card has 4.0 GB)",
        est_opt_gb, est_frozen_gb, est_opt_gb + est_frozen_gb,
    )
    if est_opt_gb + est_frozen_gb > 3.7:
        LOGGER.warning(
            "Estimated footprint is close to or over the 4 GB card limit. "
            "Reduce --unfreeze-set, or keep --half-frozen on, or lower --batch-size/--seq-len."
        )

    # Build Replay Dataset & DataLoader (post-reorg data root)
    gold_pt = TRAIN_DATA_DIR / "canonical_standardized" / "quillan_master_gold_training_v1.pt"
    default_raw = TRAIN_DATA_DIR / "quillan_pretrain_corpus_343mb.pt"
    raw_pt = Path(args.raw_corpus) if args.raw_corpus else (default_raw if default_raw.exists() else None)
    if not gold_pt.exists():
        raise FileNotFoundError(f"Gold replay archive not found: {gold_pt}")
    LOGGER.info("Gold replay: %s | raw corpus: %s", gold_pt.name, raw_pt.name if raw_pt else "none")

    dataset = CPTInterleavedDataset(
        gold_archive_path=gold_pt,
        raw_pt_path=raw_pt,
        seq_len=args.seq_len,
        replay_ratio=args.replay_ratio,
    )

    # ── Held-out validation split ────────────────────────────────────────────
    # Previously "best" was selected on the running TRAINING loss, which means
    # "best" meant "most memorised". Select on held-out loss instead.
    val_loader = None
    if args.val_fraction > 0 and len(dataset) > 20:
        n_val = max(1, int(len(dataset) * args.val_fraction))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(1337)
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        LOGGER.info("Dataset split: %d train / %d held-out val (%.1f%%)",
                    n_train, n_val, 100.0 * args.val_fraction)
    else:
        train_ds = dataset
        LOGGER.warning("No validation split (--val-fraction 0). Checkpoints will be "
                       "selected on training loss, which cannot distinguish learning "
                       "from memorisation.")

    dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(dataloader)

    # Optimizer (separate 2D weight decay from 1D biases/norms)
    decay_params = [p for p in model.parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # fp16 autocast + loss scaling is REQUIRED when frozen weights are stored as fp16,
    # otherwise the fp16 weight matmuls raise a dtype mismatch against fp32 activations.
    amp_enabled = bool(half_frozen)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if amp_enabled:
        LOGGER.info("Mixed precision ENABLED (frozen weights are fp16) with GradScaler.")

    val_iter = iter(val_loader) if val_loader is not None else None

    def evaluate_val() -> float:
        """Mean held-out cross-entropy. No grad, model restored to train() after."""
        if val_iter is None:
            return float("nan")
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for _ in range(args.val_batches):
                try:
                    v_inp, v_tgt = next(val_iter)
                except StopIteration:
                    break
                v_inp, v_tgt = v_inp.to(device), v_tgt.to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    v_out = model(v_inp, use_cache=False, deliberation=False)
                v_logits = v_out[0] if isinstance(v_out, tuple) else v_out
                v_loss = F.cross_entropy(
                    v_logits.contiguous().view(-1, cfg.vocab_size),
                    v_tgt.contiguous().view(-1),
                    ignore_index=-100,
                )
                total += float(v_loss.item())
                n += 1
        model.train()
        return total / max(1, n)

    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    LOGGER.info("Starting CPT: Steps=%d, Warmup=%d, MaxLR=%.2e, BatchSize=%d", args.steps, args.warmup_steps, args.lr, args.batch_size)

    running_loss = 0.0
    start_time = time.time()
    best_loss = float("inf")   # running training loss (telemetry only)
    best_val = float("inf")    # <-- checkpoint selection metric when val is available

    for step in range(1, args.steps + 1):
        try:
            batch_inp, batch_tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            batch_inp, batch_tgt = next(loader_iter)

        batch_inp = batch_inp.to(device)
        batch_tgt = batch_tgt.to(device)

        # Update dynamic learning rate
        curr_lr = get_lr(step, args.warmup_steps, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        optimizer.zero_grad(set_to_none=True)

        # Forward pass through BitNet 1.58b STE MoE
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = model(batch_inp, use_cache=False, deliberation=False)
        logits = (out[0] if isinstance(out, tuple) else out).float()

        # Causal Cross-Entropy Loss: logits at position t predicts batch_tgt at position t
        shift_logits = logits.contiguous().view(-1, cfg.vocab_size)
        shift_targets = batch_tgt.contiguous().view(-1)
        ce_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=-100)

        # Dynamic Entropy Regularization: penalizes attractor collapse and uniform repetition
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        loss = ce_loss - (getattr(cfg, "entropy_bonus_weight", 0.01) * entropy)

        # Backpropagation via Straight-Through Estimators (scaled for fp16 frozen weights)
        scaler.scale(loss).backward()

        # Gradient clipping prevents ternary quantization instability.
        # Only trainable params are clipped; frozen ones have no gradients.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Telemetry Logging
        is_log_step = (step % args.log_interval == 0) or (step == 1) or (step == args.steps)
        if is_log_step:
            step_count = (step % args.log_interval) if (step % args.log_interval != 0 and step > 1) else (args.log_interval if step > 1 else 1)
            avg_loss = running_loss / max(1, step_count)
            running_loss = 0.0
            elapsed = time.time() - start_time
            tok_per_sec = (step * args.batch_size * args.seq_len) / max(1.0, elapsed)
            LOGGER.info(
                "Step %4d/%d | Loss: %.4f | PPL: %.2f | LR: %.2e | Speed: %.1f tok/s",
                step, args.steps, avg_loss, math.exp(min(20.0, avg_loss)), curr_lr, tok_per_sec
            )

            # ── Checkpoint selection ─────────────────────────────────────────
            # Prefer the held-out metric. Training loss can fall while the model
            # only memorises; a val split is the only signal that separates the two.
            if val_iter is not None:
                if step % args.val_interval == 0 or step == args.steps:
                    v = evaluate_val()
                    LOGGER.info(
                        "            | VAL loss: %.4f | VAL PPL: %.2f | best_val: %.4f",
                        v, math.exp(min(20.0, v)), best_val,
                    )
                    if v < best_val:
                        best_val = v
                        save_atomic_checkpoint(out_ckpt, model, cfg, step, v)
                        LOGGER.info("            >>> New best by VALIDATION loss -> %s", out_ckpt.name)
            elif avg_loss < best_loss and step > args.warmup_steps:
                best_loss = avg_loss
                save_atomic_checkpoint(out_ckpt, model, cfg, step, avg_loss)

        # Periodic Periodic Save
        if step % args.save_interval == 0:
            step_ckpt = ckpt_dir / f"quillan_{args.model}_cpt_step_{step}.pt"
            save_atomic_checkpoint(step_ckpt, model, cfg, step, loss.item())
            save_atomic_checkpoint(out_ckpt, model, cfg, step, loss.item())

    # Final Save — must NOT overwrite the best checkpoint.
    # The original wrote to out_ckpt here, which SILENTLY CLOBBERED the best
    # weights with the last step's state. That is why quillan_6l_cpt_best.pt
    # reports step=500 / loss=6.797 instead of the best value seen.
    final_ckpt = ckpt_dir / f"quillan_{args.model}_cpt_final.pt"
    save_atomic_checkpoint(final_ckpt, model, cfg, args.steps, loss.item())
    LOGGER.info(
        "[DONE] Continued Pre-Training completed successfully in %.1f minutes. "
        "Best val: %.4f | best weights: %s | final: %s",
        (time.time() - start_time) / 60, best_val, out_ckpt.name, final_ckpt.name,
    )


# ── CLI Interface ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan-Ronin Continued Pre-Training (CPT) Engine")
    parser.add_argument("--model", type=str, default="mini", choices=["mini", "main"], help="Model size to train")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps. NOTE: the previous 500-step run consumed only ~2.6%% of one epoch of the 10M-token corpus, which is the main reason PPL plateaued near 14.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per step")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--lr", type=float, default=8e-5, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warm-up steps")
    parser.add_argument("--replay-ratio", type=float, default=0.15, help="Fraction of batches drawn from gold replay")
    parser.add_argument("--raw-corpus", type=str, default=None, help="Path to secondary continuous text .pt file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (cpu, cuda)")
    parser.add_argument("--init-ckpt", type=str, default=None, help="Explicit baseline checkpoint path (overrides auto-resolution)")
    parser.add_argument(
        "--unfreeze-set", type=str, default="wide", choices=["wide", "legacy", "full"],
        help="wide = experts+routers+norms+ATTENTION+embeddings+prism/bridge (default); "
             "legacy = old GPU set (no attention/embeddings); full = everything",
    )
    parser.add_argument(
        "--half-frozen", action=argparse.BooleanOptionalAction, default=True,
        help="Store frozen weights in fp16 to free VRAM (default: on). Requires autocast, "
             "which is enabled automatically. Use --no-half-frozen to disable.",
    )
    parser.add_argument(
        "--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
        help="Enable activation gradient checkpointing (default: on). Trades ~30%% step time "
             "for a large activation-memory reduction.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Held-out validation fraction (default 0.05)")
    parser.add_argument("--val-interval", type=int, default=50, help="Steps between validation passes")
    parser.add_argument("--val-batches", type=int, default=8, help="Batches per validation pass")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log outputs")
    parser.add_argument("--save-interval", type=int, default=50, help="Steps between periodic checkpoint saves")

    cli_args = parser.parse_args()
    run_training(cli_args)
