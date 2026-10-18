#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN GOLD INSTRUCTION SFT ENGINE
============================================
Executes high-density instruction fine-tuning on the 2,355 pristine gold Q&A samples:
  - Dataset: training_data/canonical_standardized/quillan_gold_sft_master.pt
  - Strict prompt masking: ignore_index = -100 on prompts and padding
  - Hardware ceiling: torch.set_num_threads(3) reserving Core 0 for host OS
  - Automatic atomic saves to quillan_6l_dense_gold_best.pt and mini_sft_best.pt
"""

import math
import os
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

torch.set_num_threads(3)
torch.set_num_interop_threads(1)

REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO), str(REPO / "scripts"), str(REPO / "03 - Training & Model"), str(REPO / "09 - Projects" / "projects" / "oni")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from quillan_bpe_tokenizer import QuillanBPETokenizer

DATASET_PATH = REPO / "training_data" / "canonical_standardized" / "quillan_gold_sft_master.pt"
CKPT_DIR = REPO / "checkpoints" / "checkpoints_oni"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = CKPT_DIR / "quillan_6l_dense_gold_best.pt"
GATEWAY_CKPT = REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt"

PROMPTS = [
    "User: What is 17 * 19?\n\nAssistant:",
    "User: Explain the chemical process of photosynthesis in plants.\n\nAssistant:",
    "User: Write a Python function to check if a string is a palindrome.\n\nAssistant:",
    "User: What is the difference between SIGTERM and SIGKILL in Linux?\n\nAssistant:"
]

def generate_probe(model, tok, prompt, max_tokens=50):
    model.eval()
    ids = tok.encode(prompt)
    gen = list(ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([gen[-256:]], dtype=torch.long)
            out = model(x)
            logits = (out[0] if isinstance(out, tuple) else out)[0, -1, :tok.vocab_size]
            probs = F.softmax(logits / 0.5, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            if nxt in (50256, 50257):
                break
            gen.append(nxt)
    model.train()
    return tok.decode(gen[len(ids):]).strip()

def main():
    print("=" * 72, flush=True)
    print("  👑 QUILLAN-RONIN GOLD INSTRUCTION SFT ENGINE", flush=True)
    print("=" * 72, flush=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Master SFT dataset not found: {DATASET_PATH}")

    tok = QuillanBPETokenizer()
    print(f"Loaded tokenizer with {tok.vocab_size} tokens.", flush=True)

    print(f"Loading packed master SFT dataset from {DATASET_PATH.name}...", flush=True)
    raw = torch.load(DATASET_PATH, map_location="cpu", weights_only=True)
    input_ids = raw["input_ids"].long()
    labels = raw["labels"].long()
    n_samples = len(input_ids)
    print(f"Dataset loaded: {n_samples} samples | Shape: {list(input_ids.shape)}", flush=True)

    # 90/10 Train / Val split
    val_split = int(n_samples * 0.10)
    train_x = input_ids[:-val_split]
    train_y = labels[:-val_split]
    val_x = input_ids[-val_split:]
    val_y = labels[-val_split:]
    print(f"Split: {len(train_x)} training samples | {len(val_x)} validation samples", flush=True)

    import argparse
    parser = argparse.ArgumentParser(description="Quillan-Ronin Gold SFT Engine")
    parser.add_argument("--steps", type=int, default=200, help="Total SFT steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6.0e-5, help="Base learning rate")
    parser.add_argument("--eval_interval", type=int, default=25, help="Validation interval")
    args = parser.parse_args()

    cfg = QuillanOniConfig(
        vocab_size=50257,
        hidden_dim=1024,
        ffn_dim=2048,
        n_layer=6,
        num_experts=34,
        top_k=4,
        max_seq_len=512,
        router_mode="topk"
    )
    model = QuillanRoninOni(cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {param_count/1e6:.1f}M parameters.", flush=True)

    best_val_loss = float("inf")

    if BEST_CKPT.exists():
        print(f"Loading base weights from {BEST_CKPT.name}...", flush=True)
        try:
            ckpt_data = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state_dict"], strict=True)
                prev_loss = ckpt_data.get("val_loss", float("inf"))
                print(f"✅ Successfully loaded checkpoint (Previous Val Loss: {prev_loss:.4f})!", flush=True)
            elif isinstance(ckpt_data, dict) and "model" in ckpt_data:
                model.load_state_dict(ckpt_data["model"], strict=True)
                print(f"✅ Successfully loaded model weights!", flush=True)
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint ({e}), initializing fresh.", flush=True)

    steps = args.steps
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    lr_base = args.lr
    lr_min = lr_base * 0.1
    warmup_steps = 15
    eval_interval = args.eval_interval
    aux_alpha = 0.01

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    print(f"SFT Training Plan: 1 -> {steps} steps | Effective Batch: {batch_size*grad_accum}", flush=True)
    print(f"Learning Rate: {lr_base:.2e} -> {lr_min:.2e} (Cosine Annealed)", flush=True)

    start_time = time.time()
    optimizer.zero_grad()

    for step in range(1, steps + 1):
        if step <= warmup_steps:
            cur_lr = lr_base * (step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            cur_lr = lr_min + 0.5 * (lr_base - lr_min) * (1.0 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        batch_losses = []
        for _ in range(grad_accum):
            idx = torch.randint(0, len(train_x), (batch_size,))
            bx = train_x[idx]
            by = train_y[idx]

            out = model(bx, labels=by, return_aux=True)
            if isinstance(out, tuple) and len(out) == 3:
                logits, ce_loss, aux = out
                aux_val = model.total_aux_loss(aux) if (isinstance(aux, dict) and hasattr(model, "total_aux_loss")) else (aux if isinstance(aux, torch.Tensor) else torch.tensor(0.0))
                total_loss = ce_loss + aux_alpha * aux_val
            elif isinstance(out, tuple):
                logits, total_loss = out
            else:
                total_loss = out

            loss_scaled = total_loss / grad_accum
            loss_scaled.backward()
            batch_losses.append(total_loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        avg_train_loss = sum(batch_losses) / len(batch_losses)

        if step == 1 or step % 5 == 0:
            elapsed = time.time() - start_time
            rate = (step * batch_size * grad_accum * 256) / max(1e-4, elapsed)
            print(f"SFT Step {step:4d}/{steps} | Train Loss: {avg_train_loss:.4f} | LR: {cur_lr:.2e} | Speed: {rate:.1f} tok/s | Elapsed: {elapsed/60:.1f}m", flush=True)

        if step % eval_interval == 0 or step == steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(12):
                    v_idx = torch.randint(0, len(val_x), (batch_size,))
                    v_bx = val_x[v_idx]
                    v_by = val_y[v_idx]
                    v_out = model(v_bx, labels=v_by, return_aux=False)
                    v_loss = v_out[1] if isinstance(v_out, tuple) else v_out
                    val_losses.append(v_loss.item())
            val_mean = sum(val_losses) / len(val_losses)
            print(f"\n--- SFT EVALUATION [Step {step}] ---", flush=True)
            print(f"Validation Loss: {val_mean:.4f} (Previous Best: {best_val_loss:.4f})", flush=True)

            if val_mean < best_val_loss:
                best_val_loss = val_mean
                payload = {
                    "step": step,
                    "val_loss": val_mean,
                    "config": cfg.__dict__,
                    "model_state_dict": model.state_dict(),
                }
                BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)
                GATEWAY_CKPT.parent.mkdir(parents=True, exist_ok=True)
                tmp_best = BEST_CKPT.with_suffix(".pt.tmp")
                tmp_gw = GATEWAY_CKPT.with_suffix(".pt.tmp")
                torch.save(payload, tmp_best)
                torch.save(payload, tmp_gw)
                if tmp_best.exists():
                    os.replace(tmp_best, BEST_CKPT)
                if tmp_gw.exists():
                    os.replace(tmp_gw, GATEWAY_CKPT)
                print(f"🌟 NEW SFT BEST ATOMICALLY SAVED to {BEST_CKPT.name} & {GATEWAY_CKPT.name}!", flush=True)

            # Probe generations
            p_idx = (step // eval_interval) % len(PROMPTS)
            sample_1 = generate_probe(model, tok, PROMPTS[0], max_tokens=40)
            sample_2 = generate_probe(model, tok, PROMPTS[p_idx], max_tokens=40)
            print(f"Probe (17*19): {sample_1}", flush=True)
            if p_idx != 0:
                print(f"Probe ({PROMPTS[p_idx][:25]}...): {sample_2}", flush=True)
            print("--------------------------------", flush=True)
            model.train()

    print("\n" + "=" * 72, flush=True)
    print(f"  🎉 SFT FINE-TUNING COMPLETE! Best Val Loss: {best_val_loss:.4f}", flush=True)
    print(f"  Saved checkpoint: {BEST_CKPT}", flush=True)
    print("=" * 72, flush=True)

if __name__ == "__main__":
    main()
