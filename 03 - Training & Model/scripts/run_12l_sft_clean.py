#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_12l_sft_clean.py  —  Quillan-Ronin 12L CLEAN SFT (Monolithic BPE Edition)
============================================================================
Trains the 12-Layer Main model (1.0B) on clean question-and-answer pairs:
  1. Monolithic BPE Encoding: Prompt + Answer encoded as ONE continuous string.
     Eliminates subword splicing bugs.
  2. Loss Masking: Computes loss strictly on assistant response tokens (-100 on prompt).
  3. Selective Unfreezing: Unfreezes alignment parameters (~65M params) to fit
     100% within physical 4GB VRAM without PCIe paging stalls.
  4. Micro-batch 1 with 8-step Gradient Accumulation (Effective Batch Size = 8).
  5. 512-Token Context Anchor.
"""
import os, sys, gc, json, time, math, functools
from pathlib import Path

print = functools.partial(print, flush=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

REPO_ROOT   = Path(r"C:\02_QUILLAN")

def resolve_path(rel_p):
    p1 = REPO_ROOT / "03 - Training & Model" / rel_p
    if p1.exists():
        return p1
    return REPO_ROOT / rel_p

SCRIPTS_DIR = resolve_path("scripts")
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model" / "scripts"))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

CKPT_IN  = resolve_path("checkpoints/checkpoints_oni/quillan_12l_ma_best.pt")
CKPT_OUT = resolve_path("checkpoints/checkpoints_oni/quillan_12l_clean_sft.pt")
CKPT_DIR = resolve_path("checkpoints/checkpoints_oni/sft_steps_12l")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class MonolithicCleanSFTDataset(Dataset):
    def __init__(self, tok: Tokenizer, seq_len: int = 512):
        self.seq_len = seq_len
        self.samples = []
        self.tok = tok
        self.EOS_ID = 0

        sources = [
            (resolve_path("training_data/Quillan_Universal_Sovereign_Gold_1000.jsonl"), "question", "response"),
            (resolve_path("training_data/Quillan_Direct_Answers_Gold.jsonl"), "prompt", "response"),
            (resolve_path("training_data/sovereign_thinking_gold.jsonl"), "question", "response"),
        ]

        total_read = 0
        skipped_long = 0

        for path, q_key, a_key in sources:
            if not path.exists():
                print(f"  [SKIP] Not found: {path.name}")
                continue
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        q = item.get(q_key, "").strip()
                        a = item.get(a_key, "").strip()
                        if not q or not a:
                            continue
                        total_read += 1

                        prompt_prefix = f"User: {q}\n\nAssistant:"
                        full_text     = f"{prompt_prefix} {a}"

                        prompt_ids = self.tok.encode(prompt_prefix).ids
                        full_ids   = self.tok.encode(full_text).ids

                        if len(full_ids) + 1 > self.seq_len:
                            skipped_long += 1
                            continue

                        prefix_len = len(prompt_ids)
                        if full_ids[:prefix_len] != prompt_ids:
                            continue

                        input_ids = full_ids + [self.EOS_ID]
                        labels    = [-100] * prefix_len + full_ids[prefix_len:] + [self.EOS_ID]

                        # Keep exact unpadded sequence length (3.8x faster on CPU)
                        self.samples.append((
                            torch.tensor(input_ids, dtype=torch.long),
                            torch.tensor(labels,    dtype=torch.long),
                        ))
                        count += 1
                    except Exception:
                        pass
            print(f"  Loaded {count:4d} intact samples from {path.name}")

        print(f"  Total intact 12L samples: {len(self.samples)} (skipped {skipped_long} over-length)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def probe(model, tok, prompt, device, max_new=60):
    model.eval()
    fmt  = f"User: {prompt}\n\nAssistant:"
    ids  = tok.encode(fmt).ids
    gen  = list(ids)
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new):
            out    = model(x, use_cache=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out
            nxt    = int(logits[0, -1, :].argmax().item())
            if nxt in (0, 50256):
                break
            gen.append(nxt)
            x = torch.tensor([gen[-512:]], dtype=torch.long, device=device)
    model.train()
    return tok.decode(gen[len(ids):]).strip()


def run():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _probe_t = torch.zeros(1, device="cuda") + 1
            device = torch.device("cuda")
        except Exception:
            print("  [DEVICE] GPU sm_61 detected without binary kernels in PyTorch wheel; utilizing optimized CPU threading.")
            device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print("=" * 70)
    print("  QUILLAN 12L — MONOLITHIC SFT (Attention, Prism, LoRA & Gate Unfrozen)")
    print(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))
    print("=" * 70)

    print(f"\nBase checkpoint: {CKPT_IN.name}")
    data   = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    cfg_dict = dict(data["config"])
    cfg_dict["device"] = str(device)
    cfg_dict["max_seq_len"] = 512
    cfg    = QuillanOniConfig(**cfg_dict)
    print(f"  12L base step={data.get('step', '?')} val_loss={data.get('val_loss', 0.0):.4f}")

    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(data["model_state_dict"], strict=False)

    trainable, frozen = [], []
    # Comprehensive unfreeze: Self-Attention (QKV/out), Semantic Prism, 34 Expert LoRAs, Gates, Norms, Bridges, Embeddings
    ALIGNMENT_KEYS = ("wte", "ln", "norm", "gate", "lora", "finalizer", "c_attn", "c_proj", "prism", "bridge")
    for name, p in model.named_parameters():
        if any(k in name for k in ALIGNMENT_KEYS):
            p.requires_grad = True
            trainable.append(p)
        else:
            p.requires_grad = False

    print(f"\n  ======================================================================")
    print(f"  [ATTENTION + PRISM + EXPERT LORA + GATE UNFREEZE ACTIVATED]")
    print(f"  Trainable Parameters: {sum(p.numel() for p in trainable):,} | Frozen: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"  All Attention Heads, Semantic Prism, Expert Adapters, and Routing Gates Active")
    print(f"  ======================================================================\n")

    tok = Tokenizer.from_file(str(resolve_path("quillan_bpe_tokenizer_hf/tokenizer.json")))

    print("\nBuilding monolithic clean dataset (seq_len=512)...")
    dataset    = MonolithicCleanSFTDataset(tok, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    LR           = 2.5e-5
    WARMUP_STEPS = 25
    MAX_STEPS    = 500
    ACCUM        = 4        # effective batch 4
    EARLY_STOP   = 0.35     # Deep convergence floor
    SAVE_EVERY   = 25
    PROBE_EVERY  = 25

    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.98), weight_decay=0.01)

    print(f"\n  Max steps    : {MAX_STEPS}")
    print(f"  LR           : {LR:.1e} (cosine w/ {WARMUP_STEPS} warmup steps)")
    print(f"  Effective BS : {1*ACCUM}")
    print(f"  Early stop   : loss <= {EARLY_STOP}")
    print(f"  Save every   : {SAVE_EVERY} steps")
    print("=" * 70 + "\n")

    step         = 0
    running_loss = 0.0
    best_loss    = float("inf")
    loader_iter  = iter(dataloader)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    PROBE_PROMPTS = [
        "What is the exact speed of light in a vacuum?",
        "Who are you and what is your architecture?",
    ]

    while step < MAX_STEPS:
        step += 1

        if step < WARMUP_STEPS:
            curr_lr = LR * step / WARMUP_STEPS
        else:
            prog    = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
            curr_lr = 2e-6 + 0.5 * (LR - 2e-6) * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        accum_loss = 0.0
        for _ in range(ACCUM):
            try:
                b_inp, b_tgt = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                b_inp, b_tgt = next(loader_iter)

            b_inp = b_inp.to(device)
            b_tgt = b_tgt.to(device)

            out    = model(b_inp, use_cache=False, deliberation=False)
            logits = out[0] if isinstance(out, tuple) else out

            shift_logits  = logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size)
            shift_targets = b_tgt[:, 1:].contiguous().view(-1)

            valid = (shift_targets != -100)
            if valid.any():
                loss = F.cross_entropy(shift_logits[valid], shift_targets[valid])
            else:
                loss = shift_logits.sum() * 0.0
            scaled      = loss / ACCUM
            scaled.backward()
            accum_loss += loss.item() / ACCUM

            del out, logits, shift_logits, shift_targets, b_inp, b_tgt, loss, scaled

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += accum_loss

        if step == 1 or step % 10 == 0:
            count = 1.0 if step == 1 else 10.0
            avg = running_loss / count
            ppl = math.exp(min(avg, 20.0))
            print(f"[{step:4d}/{MAX_STEPS}] loss={avg:.4f}  PPL={ppl:6.2f}  lr={curr_lr:.2e}")
            running_loss = 0.0

            if step % SAVE_EVERY == 0:
                step_ckpt = CKPT_DIR / f"step_{step:04d}_loss{avg:.3f}.pt"
                torch.save({
                    "step": step, "loss": avg, "ppl": ppl,
                    "config": cfg.__dict__,
                    "model_state_dict": model.state_dict(),
                    "timestamp": time.time(),
                }, step_ckpt)

            if avg < best_loss:
                best_loss = avg
                torch.save({
                    "step": step, "loss": best_loss, "ppl": ppl,
                    "config": cfg.__dict__,
                    "model_state_dict": model.state_dict(),
                    "timestamp": time.time(),
                    "engine": "Quillan-12L Monolithic SFT",
                }, CKPT_OUT)
                print(f"  >>> [BEST] Saved: {CKPT_OUT.name}  loss={best_loss:.4f}")

            if avg <= EARLY_STOP:
                print(f"\n  [EARLY STOP] Balanced loss floor reached: {avg:.4f} <= target {EARLY_STOP}")
                print(f"  Best checkpoint: {CKPT_OUT.name}  loss={best_loss:.4f}")
                break

        if step % PROBE_EVERY == 0:
            print(f"\n{'─'*70}")
            print(f"  [GENERATION PROBE @ step {step}]")
            for pp in PROBE_PROMPTS:
                resp = probe(model, tok, pp, device)
                print(f"  Q: {pp}")
                print(f"  A: {resp[:200]}")
            print(f"{'─'*70}\n")

    print("\n" + "=" * 70)
    print(f"  DONE | Best loss: {best_loss:.4f} | Saved: {CKPT_OUT.name}")
    print("=" * 70)


if __name__ == "__main__":
    run()
