#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN STABILIZED SFT & MEMORY ATTENTION TRAINING ENGINE
===================================================================
Solves:
  1. High-variance gradient collapse by using smoothed micro-batching + gradient accumulation.
  2. Integrated support for ArXiv:2609.28399 Memory Attention (zero-FLOP value projection).
  3. Proper training for both Mini-6L and Main-12L architectures without layer disconnection.
  4. Deterministic evaluation and atomic checkpoint persistence.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(r"C:\02_QUILLAN")
for p in [str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(REPO_ROOT / "03 - Training & Model")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_memory_attention import MemoryAttention

DATASET_PATH = REPO_ROOT / "training_data" / "canonical_standardized" / "quillan_gold_sft_master.pt"
CKPT_DIR = REPO_ROOT / "checkpoints" / "checkpoints_oni"

PROMPTS = [
    "User: What is 17 * 19? Give the number directly.\n\nAssistant:",
    "User: What is photosynthesis in one sentence?\n\nAssistant:",
    "User: Write a Python function to check if a string is a palindrome.\n\nAssistant:",
    "User: What HTTP header prevents clickjacking attacks?\n\nAssistant:",
]


def generate_probe(model: nn.Module, tok: QuillanBPETokenizer, prompt: str, max_tokens: int = 15) -> str:
    """Generates sample tokens using greedy decoding for deterministic quality monitoring."""
    model.eval()
    device = next(model.parameters()).device
    tokens = tok.encode(prompt)
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([generated[-128:]], dtype=torch.long, device=device)
            out = model(x, use_cache=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            nxt = torch.argmax(logits[0, -1, :tok.vocab_size], dim=-1).item()
            if nxt in [50256, 50257]:
                break
            generated.append(nxt)

    res = tok.decode(generated[len(tokens):]).strip()
    for st in ["<|endoftext|>", "<|end|>", "User:"]:
        if st in res:
            res = res.split(st)[0].strip()
    return res


def train():
    parser = argparse.ArgumentParser(description="Stabilized SFT Training Pipeline")
    parser.add_argument("--layers", type=int, default=6, choices=[6, 12], help="Model depth: 6 (Mini) or 12 (Main)")
    parser.add_argument("--steps", type=int, default=150, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=4.0e-5, help="Peak learning rate")
    parser.add_argument("--eval_interval", type=int, default=25, help="Validation interval")
    parser.add_argument("--eval_samples", type=int, default=24, help="Number of validation samples to evaluate")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH), help="Path to SFT dataset .pt file")
    parser.add_argument("--use_ma", action="store_true", help="Enable Memory Attention (ArXiv:2609.28399)")
    parser.add_argument("--freeze_layers", type=int, default=6, help="Number of bottom layers to freeze on 12L (default 6)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    safe_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(safe_threads)

    print("=" * 72, flush=True)
    print(f"  [QUILLAN-RONIN] STABILIZED SFT ENGINE ({args.layers}-LAYER {'MA' if args.use_ma else 'DENSE'})", flush=True)
    print(f"  PyTorch threads: {safe_threads} | Effective Batch: {args.batch_size * args.grad_accum}", flush=True)
    print("=" * 72, flush=True)

    tok = QuillanBPETokenizer()

    target_data_path = Path(args.dataset)
    if not target_data_path.exists():
        raise FileNotFoundError(f"Missing master SFT dataset: {target_data_path}")

    raw_data = torch.load(target_data_path, map_location="cpu", weights_only=True)
    input_ids = raw_data["input_ids"].long()
    labels = raw_data["labels"].long()
    n_samples = len(input_ids)

    val_count = int(n_samples * 0.10)
    train_x, train_y = input_ids[:-val_count], labels[:-val_count]
    val_x, val_y = input_ids[-val_count:], labels[-val_count:]

    print(f"Dataset: {target_data_path.name} | {n_samples} total | {len(train_x)} train | {len(val_x)} val", flush=True)

    cfg = QuillanOniConfig(
        vocab_size=50257,
        hidden_dim=1024,
        ffn_dim=2048,
        n_layer=args.layers,
        num_experts=34,
        top_k=4,
        max_seq_len=512,
        router_mode="topk",
    )
    if args.use_ma:
        setattr(cfg, "use_memory_attention", True)

    model = QuillanRoninOni(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model instantiated: {total_params / 1e6:.1f}M params ({args.layers} layers)", flush=True)

    # Base checkpoint resolution
    target_ckpt = CKPT_DIR / (f"quillan_{args.layers}l_ma_best.pt" if args.use_ma else f"quillan_{args.layers}l_stabilized_best.pt")
    base_candidates = []
    if args.use_ma:
        base_candidates.append(CKPT_DIR / f"quillan_{args.layers}l_ma_best.pt")
    base_candidates.extend([
        CKPT_DIR / f"quillan_{args.layers}l_best.pt",
        CKPT_DIR / "quillan_6l_dense_gold_best.pt",
    ])
    loaded = False
    for bc in base_candidates:
        if bc.exists():
            try:
                ckpt_data = torch.load(bc, map_location="cpu", weights_only=False)
                sd = ckpt_data.get("model", ckpt_data.get("model_state_dict", ckpt_data))
                if args.use_ma:
                    adapted_sd = {}
                    wte = sd.get("wte.weight", None)
                    for k, v in sd.items():
                        if ".attn.c_attn.weight" in k and v.shape[0] == 3072:
                            adapted_sd[k] = v[:2048, :].clone()
                        elif ".attn.c_attn.bias" in k and v.shape[0] == 3072:
                            adapted_sd[k] = v[:2048].clone()
                        else:
                            adapted_sd[k] = v
                    if wte is not None:
                        for l_idx in range(args.layers):
                            tm_key = f"h.{l_idx}.attn.token_memory.weight"
                            if tm_key not in adapted_sd:
                                adapted_sd[tm_key] = wte.clone()
                            norm_key = f"h.{l_idx}.attn.mem_norm.weight"
                            if norm_key not in adapted_sd:
                                adapted_sd[norm_key] = torch.ones(cfg.head_dim)
                    sd = adapted_sd
                prev_best_val = ckpt_data.get("val_loss", float("inf")) if isinstance(ckpt_data, dict) else float("inf")
                m, u = model.load_state_dict(sd, strict=False)
                if args.use_ma and "adapted_sd" in locals():
                    del adapted_sd
                del ckpt_data, sd
                import gc; gc.collect()
                print(f"Loaded weights from {bc.name} (Missing: {len(m)}, Unexpected: {len(u)})", flush=True)
                if prev_best_val < float("inf"):
                    print(f"Resuming with existing best val loss: {prev_best_val:.4f}", flush=True)
                loaded = True
                break
            except Exception as e:
                print(f"Note: Could not load {bc.name}: {e}", flush=True)

    if not loaded:
        print("Starting training with standard Gaussian initialization.", flush=True)

    device = torch.device(args.device)
    if args.layers == 12:
        # Dual-tier alignment: freeze foundational layers (default 6 or 10 on 4GB GPU), train deeper reasoning layers
        n_freeze = min(args.freeze_layers, 11)
        for l_idx in range(n_freeze):
            for p in model.h[l_idx].parameters():
                p.requires_grad = False
    elif args.layers == 6 and device.type == "cuda":
        n_freeze = min(args.freeze_layers, 5)
        for l_idx in range(n_freeze):
            for p in model.h[l_idx].parameters():
                p.requires_grad = False

    # Freeze static token memory projections and embeddings (ArXiv:2609.28399 zero-FLOP projection)
    if hasattr(model, "wte"):
        model.wte.weight.requires_grad = False
    for l_idx in range(args.layers):
        if hasattr(model.h[l_idx].attn, "token_memory"):
            model.h[l_idx].attn.token_memory.weight.requires_grad = False

    model.to(device)
    if device.type == "cuda":
        print(f"Model placed on {device} ({torch.cuda.get_device_name(0)}) | Initial VRAM: {torch.cuda.memory_allocated() / 1e6:.1f} MB", flush=True)
    else:
        print(f"Model placed on {device}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)/1e6:.1f}M / {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    steps = args.steps
    warmup_steps = 15
    lr_min = args.lr * 0.1
    best_val_loss = prev_best_val if "prev_best_val" in locals() else float("inf")
    start_time = time.time()

    # Sequential permutation shuffling ensures 100% unique sample coverage per epoch
    perm = torch.randperm(len(train_x))
    perm_ptr = 0

    model.train()
    optimizer.zero_grad()

    for step in range(1, steps + 1):
        if step <= warmup_steps:
            cur_lr = args.lr * (step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = lr_min + 0.5 * (args.lr - lr_min) * (1.0 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        step_losses = []
        for _ in range(args.grad_accum):
            if perm_ptr + args.batch_size > len(train_x):
                perm = torch.randperm(len(train_x))
                perm_ptr = 0
            idx = perm[perm_ptr : perm_ptr + args.batch_size]
            perm_ptr += args.batch_size
            bx = train_x[idx].to(device)
            by = train_y[idx].to(device)

            out = model(bx, labels=by, return_aux=False)
            total_loss = out[1] if isinstance(out, tuple) else out

            loss_scaled = total_loss / args.grad_accum
            loss_scaled.backward()
            step_losses.append(total_loss.item())
            del out, total_loss, loss_scaled

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_train_loss = sum(step_losses) / len(step_losses)

        if step == 1 or step % 5 == 0:
            elapsed = time.time() - start_time
            rate = (step * args.batch_size * args.grad_accum * 256) / max(1e-4, elapsed)
            print(f"Step {step:3d}/{steps} | Train Loss: {avg_train_loss:.4f} | LR: {cur_lr:.2e} | Speed: {rate:.1f} tok/s | Elapsed: {elapsed/60:.1f}m", flush=True)

        if step % args.eval_interval == 0 or step == steps:
            model.eval()
            val_ce_losses = []
            val_tot_losses = []
            with torch.no_grad():
                n_eval = min(args.eval_samples, len(val_x))
                for v_step in range(0, n_eval, 2):
                    v_bx = val_x[v_step:v_step + 2].to(device)
                    v_by = val_y[v_step:v_step + 2].to(device)
                    v_out = model(v_bx, labels=v_by, return_aux=True, deliberation=False)
                    v_ce = v_out[1].item()
                    v_aux = model.total_aux_loss(v_out[2]).item()
                    val_ce_losses.append(v_ce)
                    val_tot_losses.append(v_ce + v_aux)

            mean_val_ce = sum(val_ce_losses) / max(1, len(val_ce_losses))
            mean_val_tot = sum(val_tot_losses) / max(1, len(val_tot_losses))
            ppl = math.exp(min(20.0, mean_val_ce))
            print(f"\n--- VALIDATION EVALUATION [Step {step}] ---", flush=True)
            print(f"Val CE Loss: {mean_val_ce:.4f} (Perplexity: {ppl:.2f}) | Total Loss: {mean_val_tot:.4f} | Previous Best: {best_val_loss:.4f}", flush=True)

            if mean_val_ce < best_val_loss or mean_val_tot < best_val_loss:
                best_val_loss = min(mean_val_ce, best_val_loss)
                target_ckpt.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "step": step,
                    "val_loss": mean_val_ce,
                    "val_total_loss": mean_val_tot,
                    "perplexity": ppl,
                    "config": cfg.__dict__,
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "layers": args.layers,
                    "use_ma": args.use_ma,
                }
                tmp_ckpt = target_ckpt.with_suffix(".pt.tmp")
                torch.save(payload, tmp_ckpt)
                if tmp_ckpt.exists():
                    os.replace(tmp_ckpt, target_ckpt)
                print(f"[+] Saved new best checkpoint to {target_ckpt.name} (Val CE: {mean_val_ce:.4f}, PPL: {ppl:.2f})!", flush=True)

            if args.use_ma and hasattr(model, "fold_memory_attention_weights"):
                model.fold_memory_attention_weights()
            for p_idx, pr in enumerate(PROMPTS[:3]):
                p_probe = generate_probe(model, tok, pr, max_tokens=25)
                p_labels = ["Math [17*19]", "Science [Photosynthesis]", "Python [Palindrome]"]
                print(f"Probe {p_labels[p_idx]}: {p_probe}", flush=True)
            if args.use_ma and hasattr(model, "unfold_memory_attention_weights"):
                model.unfold_memory_attention_weights()
            print("--------------------------------------------------\n", flush=True)
            model.train()

    print(f"\n[+] SFT Training Complete! Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"Final Artifact: {target_ckpt}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        train()
    except BaseException as e:
        print(f"\n[FATAL ERROR IN TRAIN]: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
