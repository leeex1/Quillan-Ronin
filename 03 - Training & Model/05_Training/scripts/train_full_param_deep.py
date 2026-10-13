#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — FULL-PARAMETER Deep Conversational Alignment
Unfreezes ALL 453.9M parameters (not just 191.2M adapters) for maximum fluency.
Uses gradient accumulation, warm-up LR scheduling, and repetition penalty.
"""

import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_dev'))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from _dev.quillan_bpe_tokenizer import QuillanBPETokenizer

def parse_line_to_messages(data):
    """Multi-schema parser supporting messages, text, prompt/response, and conversations formats."""
    if 'messages' in data and data['messages']:
        return data['messages']
    if 'prompt' in data and 'response' in data:
        return [
            {"role": "user", "content": data['prompt']},
            {"role": "assistant", "content": data['response']}
        ]
    # full_dataset.jsonl: original_input/model_response schema  
    if 'original_input' in data and 'model_response' in data:
        user_text = data['original_input']
        if isinstance(user_text, dict):
            user_text = user_text.get('text', '') or user_text.get('concept', '') or str(user_text)
        assistant_text = data['model_response']
        # Prepend model_thoughts as <think> block if available
        if 'model_thoughts' in data and data['model_thoughts']:
            assistant_text = f"<think>\n{data['model_thoughts']}\n</think>\n\n{assistant_text}"
        msgs = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text}
        ]
        return msgs
    # Quillan seed dataset: question/final_output schema
    if 'question' in data and 'final_output' in data:
        assistant_text = data['final_output']
        if 'reasoning_trace' in data and data['reasoning_trace']:
            assistant_text = f"<think>\n{data['reasoning_trace']}\n</think>\n\n{assistant_text}"
        return [
            {"role": "user", "content": data['question']},
            {"role": "assistant", "content": assistant_text}
        ]
    if 'conversations' in data and data['conversations']:
        msgs = []
        for c in data['conversations']:
            role = "user" if c.get("from") in ["human", "user"] else "assistant"
            msgs.append({"role": role, "content": c.get("value", "")})
        return msgs
    if 'text' in data and data['text']:
        text = data['text']
        msgs = []
        parts = text.split("<|")
        for part in parts:
            if not part.strip():
                continue
            if part.startswith("user|>"):
                content = part[len("user|>"):].strip()
                if content.endswith("<|end|>"):
                    content = content[:-len("<|end|>")].strip()
                msgs.append({"role": "user", "content": content})
            elif part.startswith("assistant|>"):
                content = part[len("assistant|>"):].strip()
                if content.endswith("<|end|>"):
                    content = content[:-len("<|end|>")].strip()
                msgs.append({"role": "assistant", "content": content})
            elif part.startswith("system|>"):
                content = part[len("system|>"):].strip()
                if content.endswith("<|end|>"):
                    content = content[:-len("<|end|>")].strip()
                msgs.append({"role": "system", "content": content})
        if msgs:
            return msgs
    return []

def load_all_conversational_datasets(training_dir, tokenizer, max_seq_len=512, max_samples=80000):
    """Load and pack all conversational datasets into target-masked dialogue samples."""
    samples = []
    dataset_files = [
        # HIGH-QUALITY REASONING + FLUENCY (prioritize these)
        "full_train.jsonl",                                   # 8,707 PhD-level Claude reasoning turns
        "full_dataset.jsonl",                                 # 3,151 expert domain reasoning turns
        "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",  # 7 Quillan identity seed turns
        # LARGE CORPORA
        "instruct_train.jsonl",                               # 7,215 multi-turn instruction turns
        "GPT_5.5_Distilled.jsonl",                            # 18,000 reasoning and dialogue turns
        "quillan_12mb_training_dataset.jsonl",
        "train.jsonl",
        "code_train.jsonl",
        "quillan_science_absolute.jsonl",
        "quillan_science_additional.jsonl",
        "quillan_corpus_CLEAN_V7.jsonl",
    ]

    print("=" * 65)
    print("  LOADING ALL CONVERSATIONAL DATASETS INTO UNIFIED DIALOGUE CORPUS")
    print("=" * 65)

    for fname in dataset_files:
        fpath = Path(training_dir) / fname
        if not fpath.exists():
            continue

        loaded_file_samples = 0
        print(f"[DATASET] Parsing {fname}...", flush=True)

        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    messages = parse_line_to_messages(data)
                    if not messages:
                        continue

                    full_token_ids = []
                    full_target_ids = []

                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if not content:
                            continue

                        prompt_text = f"<|{role}|>\n{content}\n"
                        t_ids = tokenizer.encode(prompt_text)

                        if role in ['system', 'user']:
                            mask_ids = [-100] * len(t_ids)
                            full_token_ids.extend(t_ids)
                            full_target_ids.extend(mask_ids)
                        elif role == 'assistant':
                            full_token_ids.extend(t_ids)
                            full_target_ids.extend(t_ids)

                    if len(full_token_ids) > 1:
                        input_ids = full_token_ids[:max_seq_len - 1]
                        target_ids = full_target_ids[1:max_seq_len]
                        input_ids = input_ids[:len(target_ids)]

                        if any(t != -100 for t in target_ids):
                            samples.append((input_ids, target_ids))
                            loaded_file_samples += 1

                            if len(samples) >= max_samples:
                                break
                except Exception:
                    continue

        print(f"  --> Loaded {loaded_file_samples} packed turns from {fname}", flush=True)
        if len(samples) >= max_samples:
            break

    print(f"\n[UNIFIED CORPUS] Total packed target-masked dialogue samples: {len(samples)}", flush=True)
    return samples

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quillan FULL-PARAMETER Deep Conversational Alignment")
    parser.add_argument("--steps", type=int, default=10000, help="Number of SFT steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--warmup-steps", type=int, default=200, help="LR warmup steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    print("=" * 65)
    print("  QUILLAN-RONIN v5.3.1 — FULL-PARAMETER DEEP CONVERSATIONAL SFT")
    print(f"  ALL 453.9M PARAMETERS UNFROZEN — MAXIMUM FLUENCY TRAINING")
    print(f"  Target: {args.steps} steps | Peak LR: {args.lr} | Warmup: {args.warmup_steps}")
    print(f"  Grad Accum: {args.grad_accum} | Effective Batch: {args.grad_accum}")
    print("=" * 65)

    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    print(f"[HARDWARE] Compute device: {device}")

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg).to(device)

    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer = QuillanBPETokenizer()
    tokenizer.load(str(tok_path))

    # 1. Load Base Checkpoint
    base_path = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_full_base_final.pt'
    print(f"\n[1/2 BASE LOAD] {base_path.name}...")
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    base_sd = base_ckpt.get('model_state_dict', base_ckpt)
    model_sd = model.state_dict()
    base_loaded = sum(1 for k, v in base_sd.items() if k in model_sd and v.shape == model_sd[k].shape)
    for k, v in base_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
    print(f"[1/2 BASE LOAD] Loaded {base_loaded} / 662 base keys.")

    # 2. Overlay Previous SFT Checkpoint
    sft_path = ROOT / 'checkpoints' / 'checkpoints_sft' / 'quillan_causal_aligned.pt'
    if sft_path.exists():
        print(f"[2/2 SFT LOAD] {sft_path.name}...")
        sft_ckpt = torch.load(str(sft_path), map_location=device, weights_only=False)
        sft_sd = sft_ckpt.get('model_state_dict', sft_ckpt)
        sft_loaded = sum(1 for k, v in sft_sd.items() if k in model_sd and v.shape == model_sd[k].shape)
        for k, v in sft_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                model_sd[k].copy_(v)
        print(f"[2/2 SFT LOAD] Overlaid {sft_loaded} SFT keys.")
    else:
        print("[2/2 SFT LOAD] No previous SFT checkpoint found, training from base.")

    # Load Unified All-Dataset Corpus
    training_dir = ROOT / 'training_data'
    samples = load_all_conversational_datasets(training_dir, tokenizer, max_seq_len=args.seq_len, max_samples=80000)
    if not samples:
        print("[ERROR] No target-masked dialogue samples loaded!")
        sys.exit(1)

    # *** FULL-PARAMETER: UNFREEZE ALL 453.9M PARAMETERS ***
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] *** FULL-PARAMETER MODE ***")
    print(f"[MODEL] Total parameters: {total_params / 1e6:.1f}M")
    print(f"[MODEL] Trainable parameters: {trainable_params / 1e6:.1f}M (ALL UNFROZEN)")
    print(f"[MODEL] Frozen parameters: 0M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # Warm-up + Cosine Annealing LR Schedule
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = ROOT / 'checkpoints' / 'checkpoints_sft'
    save_dir.mkdir(parents=True, exist_ok=True)
    aligned_ckpt_path = save_dir / 'quillan_causal_aligned.pt'

    print(f"\n[TRAIN] Starting {args.steps}-step FULL-PARAMETER Deep Conversational Alignment...", flush=True)
    t0 = time.time()
    ema_loss = None
    optimizer.zero_grad()

    for step in range(1, args.steps + 1):
        input_ids_list, target_ids_list = samples[(step - 1) % len(samples)]
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        target_ids = torch.tensor([target_ids_list], dtype=torch.long, device=device)

        out = model(input_ids)
        logits = out["logits"]

        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target_ids.reshape(-1), ignore_index=-100)
        loss = loss / args.grad_accum
        loss.backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_val = loss.item() * args.grad_accum
        ema_loss = loss_val if ema_loss is None else 0.95 * ema_loss + 0.05 * loss_val

        if step % 50 == 0 or step == args.steps:
            elapsed = time.time() - t0
            sec_per_step = elapsed / step
            eta_hours = (args.steps - step) * sec_per_step / 3600.0
            current_lr = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}/{args.steps}  loss={ema_loss:.4f}  resp_ce={loss_val:.4f}  lr={current_lr:.2e}  {sec_per_step:.3f}s/st  ETA:{eta_hours:.1f}h", flush=True)

        if step % 200 == 0 or step == args.steps:
            torch.save({"model_state_dict": model.state_dict(), "step": step, "loss": ema_loss}, str(aligned_ckpt_path))

    print("\n" + "=" * 65)
    print("  FULL-PARAMETER DEEP CONVERSATIONAL ALIGNMENT COMPLETE")
    print(f"  Saved final checkpoint: {aligned_ckpt_path.name}")
    print("=" * 65, flush=True)

if __name__ == '__main__':
    main()
