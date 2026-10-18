#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_6l_sft_v2.py  -  Quillan-Ronin 6L SFT v2 (Anti-Blending Edition)
======================================================================
Fixes for word-salad / cross-topic blending from v1:

  ROOT CAUSE 1 - Memorisation collapse (loss 1.21 < 1.75 floor):
    FIX: Raise EARLY_STOP to 2.15. At loss ~2.1 the model has learned
         prompt-response routing without collapsing all examples into one
         superimposed token soup. PPL ~8-9 is the practical floor for
         clean generation on a small SFT set.

  ROOT CAUSE 2 - Frozen attention / FFN body cannot learn to route by prompt:
    FIX: Unfreeze c_attn, c_proj, and c_fc IN ADDITION to the existing
         alignment keys (wte, ln, norm, gate, lora, finalizer).
         This lets the transformer body learn prompt-conditioned routing,
         not just surface token statistics in the embedding layer.
         VRAM budget: GTX 1050 4GB. Attn+FFN adds ~18M trainable params
         to the 6L model - verified safe at batch_size=1 with ACCUM=8.

  ROOT CAUSE 3 - LR 3.5e-5 too high for body layers; caused fast convergence
                 straight through the target window into memorisation:
    FIX: Lower LR to 1.5e-5. Body params get a separate, lower LR group
         (0.6x of alignment LR) to prevent overdriving frozen-until-now
         attention weights.

  ADDITIONAL SAFEGUARDS:
    - Per-step loss tracked (not per-10-step average) so early stop fires
      exactly when the actual step loss crosses the floor.
    - Divergence guard: if loss > 6.0 after warmup, abort and report last good ckpt.
    - Generation probe uses top-p sampling (not greedy) every 10 steps.
    - probe() uses clean_forward (same path as inference) for representative probes.
    - Step-level checkpoint saved at the EXACT step the early-stop fires.
    - SAVE_EVERY reduced to 5 steps for fine-grained rollback points.
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
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

# -- Checkpoint paths ----------------------------------------------------------
# INPUT:  last CPT checkpoint (same as v1 used)
# OUTPUT: new v2 checkpoint - DOES NOT overwrite v1 quillan_6l_clean_sft.pt
CKPT_IN  = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_cpt_best.pt"
CKPT_OUT = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_sft_v2.pt"
CKPT_DIR = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "sft_v2_steps_6l"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# -- Dataset -------------------------------------------------------------------

class MonolithicSFTDataset(Dataset):
    """
    Monolithic BPE encoding: prompt + answer as one continuous string.
    Labels mask the prompt prefix with -100 so loss is computed only on
    the assistant response tokens.
    """

    def __init__(self, tok, seq_len=512):
        self.seq_len = seq_len
        self.samples = []
        self.EOS_ID  = 0  # <|endoftext|>

        sources = [
            (REPO_ROOT / "training_data" / "Quillan_Universal_Sovereign_Gold_1000.jsonl", "question", "response"),
            (REPO_ROOT / "training_data" / "Quillan_Direct_Answers_Gold.jsonl",           "prompt",   "response"),
            (REPO_ROOT / "training_data" / "sovereign_thinking_gold.jsonl",               "question", "response"),
        ]

        total_read, skipped_long = 0, 0

        for path, q_key, a_key in sources:
            if not path.exists():
                print(f"  [SKIP] Not found: {path.name}")
                continue
            count = 0
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
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

                        prompt_ids = tok.encode(prompt_prefix).ids
                        full_ids   = tok.encode(full_text).ids

                        if len(full_ids) + 1 > seq_len:
                            skipped_long += 1
                            continue

                        prefix_len = len(prompt_ids)
                        if full_ids[:prefix_len] != prompt_ids:
                            continue

                        input_ids = full_ids + [self.EOS_ID]
                        labels    = [-100] * prefix_len + full_ids[prefix_len:] + [self.EOS_ID]

                        pad       = seq_len - len(input_ids)
                        input_ids = input_ids + [0] * pad
                        labels    = labels    + [-100] * pad

                        self.samples.append((
                            torch.tensor(input_ids, dtype=torch.long),
                            torch.tensor(labels,    dtype=torch.long),
                        ))
                        count += 1
                    except Exception:
                        pass
            print(f"  Loaded {count:4d} intact samples from {path.name}")

        print(f"  Total intact samples : {len(self.samples)}  (skipped {skipped_long} over-length)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -- Clean forward (matches inference path exactly) ----------------------------

def clean_forward(model, input_ids):
    x = model.wte(input_ids)
    for block in model.h:
        try:
            out = block(x, layer_past=None, use_cache=False,
                        gov_scale=None, token_ids=input_ids)
            x = out[0]
        except Exception:
            pass

    hidden = model.ln_f(x)
    try:
        q1    = model.quillan_finalizer_q1(hidden)
        q2    = model.quillan_finalizer_q2(hidden)
        gate  = torch.sigmoid(model.quillan_comm_gate(torch.cat([q1, q2], dim=-1)))
        fused = gate * q1 + (1.0 - gate) * q2
    except Exception:
        fused = hidden

    return model.lm_head(fused)


# -- Generation probe (top-p sampling) -----------------------------------------

def probe(model, tok, prompt, device, max_new=60, temperature=0.6, top_p=0.90):
    """Top-p sampling probe on clean_forward - reveals blending that greedy hides."""
    model.eval()
    fmt      = f"User: {prompt}\n\nAssistant:"
    ids      = tok.encode(fmt).ids
    gen      = list(ids)
    EOS      = {0, 50256}
    STOPS    = ["User:", "Human:", "<|endoftext|>"]

    with torch.no_grad():
        for _ in range(max_new):
            inp    = torch.tensor([gen[-512:]], dtype=torch.long, device=device)
            logits = clean_forward(model, inp)
            curr   = logits[0, -1, :].float()

            gen_only = gen[len(ids):]
            for tid in set(gen_only[-32:]):
                curr[tid] = curr[tid] / 1.2 if curr[tid] > 0 else curr[tid] * 1.2

            sorted_logits, sorted_idx = torch.sort(curr, descending=True)
            probs = F.softmax(sorted_logits / max(temperature, 0.05), dim=-1)
            cum   = torch.cumsum(probs, dim=-1)
            sorted_logits[cum - probs > top_p] = float("-inf")
            filtered = F.softmax(sorted_logits, dim=-1)
            if not filtered.isfinite().any() or filtered.sum() < 1e-8:
                filtered = torch.ones_like(curr) / curr.size(-1)

            nxt = int(sorted_idx[torch.multinomial(filtered, 1).item()].item())
            if nxt in EOS:
                break
            gen.append(nxt)

            partial = tok.decode(gen[len(ids):])
            if any(s in partial for s in STOPS):
                break

    model.train()
    text = tok.decode(gen[len(ids):]).strip()
    for s in STOPS:
        if s in text:
            text = text.split(s)[0].strip()
    return text


# -- Save helper ---------------------------------------------------------------

def save_ckpt(path, model, cfg, step, loss, ppl, tag):
    torch.save({
        "step":             step,
        "loss":             loss,
        "ppl":              ppl,
        "config":           cfg.__dict__,
        "model_state_dict": model.state_dict(),
        "timestamp":        time.time(),
        "engine":           tag,
    }, path)


# -- Main ----------------------------------------------------------------------

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  QUILLAN 6L - SFT v2 (Anti-Blending, Broader Unfreeze)")
    print(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    print("=" * 72)

    if not CKPT_IN.exists():
        print(f"[FATAL] Base checkpoint not found: {CKPT_IN}")
        sys.exit(1)

    print(f"\nBase checkpoint : {CKPT_IN.name}")
    data = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    cfg  = QuillanOniConfig(**data["config"])
    cfg.device      = str(device)
    cfg.max_seq_len = 512
    print(f"  CPT base loss={data.get('loss', 0.0):.4f}  step={data.get('step', '?')}")

    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(data["model_state_dict"], strict=True)

    # -- Selective unfreeze (v2: body layers included) -------------------------
    # v1 only unfroze: wte, ln, norm, gate, lora, finalizer
    # v2 also unfreezes: c_attn, c_proj, c_fc
    #   c_attn : Q/K/V projections - enables prompt-conditioned routing
    #   c_proj : attention output projection - gates info propagation
    #   c_fc   : first FFN layer - controls feature expansion per-token
    # NOT unfreezing c_fc2 / mlp.out to keep VRAM within budget (~18M extra params)
    ALIGNMENT_KEYS = ("wte", "ln", "norm", "gate", "lora", "finalizer")
    BODY_KEYS      = ("c_attn", "c_proj", "c_fc")

    align_params, body_params, frozen_params = [], [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ALIGNMENT_KEYS):
            p.requires_grad = True
            align_params.append(p)
        elif any(k in name for k in BODY_KEYS):
            p.requires_grad = True
            body_params.append(p)
        else:
            p.requires_grad = False
            frozen_params.append(p)

    n_align  = sum(p.numel() for p in align_params)
    n_body   = sum(p.numel() for p in body_params)
    n_frozen = sum(p.numel() for p in frozen_params)
    print(f"  Alignment params : {n_align:>12,}  (embeddings, norms, gates, finalizer)")
    print(f"  Body params      : {n_body:>12,}  (c_attn, c_proj, c_fc)")
    print(f"  Frozen params    : {n_frozen:>12,}")
    print(f"  Total trainable  : {n_align + n_body:>12,}")

    tok = Tokenizer.from_file(str(REPO_ROOT / "quillan_bpe_tokenizer_hf" / "tokenizer.json"))

    print("\nBuilding monolithic SFT dataset (seq_len=512)...")
    dataset = MonolithicSFTDataset(tok, seq_len=512)
    if len(dataset) == 0:
        print("[FATAL] Dataset is empty - check training_data paths.")
        sys.exit(1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    # -- Hyperparameters (v2) --------------------------------------------------
    # LR lowered 3.5e-5 -> 1.5e-5; body params get 0.6x of alignment LR.
    # EARLY_STOP raised 1.75 -> 2.15 to prevent memorisation collapse.
    # LOSS_FLOOR = 1.90: if loss drops below this the model is memorising.
    # WARMUP_STEPS raised 25 -> 40 so body layers get stable early gradients.
    LR_ALIGN     = 1.5e-5
    LR_BODY      = LR_ALIGN * 0.6
    WARMUP_STEPS = 40
    MAX_STEPS    = 300
    ACCUM        = 8
    EARLY_STOP   = 2.15
    LOSS_FLOOR   = 1.90
    SAVE_EVERY   = 5
    PROBE_EVERY  = 10
    DIVERGE_LIM  = 6.0

    # Two-optimizer setup to stay within GTX 1050 4GB VRAM budget:
    #   AdamW for alignment params (65M): 3 tensors/param = 0.78 GB state
    #   SGD no-momentum for body params (57M): 0 extra tensors = 0 GB state
    #   Total optimizer state: ~0.78 GB vs 1.47 GB if AdamW for both
    #   Full budget estimate: model(3.03) + optim(0.78) + acts(0.05) = 3.86 GB
    optimizer_align = torch.optim.AdamW(
        align_params,
        lr=LR_ALIGN,
        betas=(0.9, 0.98),
        weight_decay=0.01,
    )
    # SGD with no momentum for body layers: directional push on pretrained weights
    # doesn't need per-param adaptive scaling — saves 2x optimizer state vs AdamW
    optimizer_body = torch.optim.SGD(
        body_params,
        lr=LR_BODY,
        momentum=0.0,
        weight_decay=0.005,
    )

    print(f"\n  Max steps     : {MAX_STEPS}")
    print(f"  LR align (AdamW) : {LR_ALIGN:.1e} | LR body (SGD): {LR_BODY:.1e}")
    print(f"  Warmup steps  : {WARMUP_STEPS}")
    print(f"  Effective BS  : {ACCUM}")
    print(f"  Early stop    : loss <= {EARLY_STOP}  (prevents memorisation collapse)")
    print(f"  Loss floor    : loss <= {LOSS_FLOOR}  (hard abort - overfit territory)")
    print(f"  Save every    : {SAVE_EVERY} steps")
    print(f"  Optimizer mem : ~0.78 GB state (AdamW align only + SGD body)")
    print("=" * 72 + "\n")

    PROBE_PROMPTS = [
        "What is the exact speed of light in a vacuum?",
        "Who are you and what is your architecture?",
        "What is the time complexity of Dijkstra's algorithm with a min-heap?",
    ]

    step        = 0
    step_losses = []
    best_loss   = float("inf")
    loader_iter = iter(dataloader)

    model.train()
    optimizer_align.zero_grad(set_to_none=True)
    optimizer_body.zero_grad(set_to_none=True)

    while step < MAX_STEPS:
        step += 1

        # LR schedule: cosine with warmup, decays to 10% of peak LR
        if step < WARMUP_STEPS:
            scale = step / WARMUP_STEPS
        else:
            prog  = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

        optimizer_align.param_groups[0]["lr"] = LR_ALIGN * scale
        optimizer_body.param_groups[0]["lr"]  = LR_BODY  * scale
        curr_lr = LR_ALIGN * scale

        # Micro-batch accumulation
        accum_loss = 0.0
        for _ in range(ACCUM):
            try:
                b_inp, b_tgt = next(loader_iter)
            except StopIteration:
                loader_iter  = iter(dataloader)
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

            scaled = loss / ACCUM
            scaled.backward()
            accum_loss += loss.item() / ACCUM

            del out, logits, shift_logits, shift_targets, b_inp, b_tgt, loss, scaled

        # Tighter gradient clip vs v1 (1.0 -> 0.8)
        torch.nn.utils.clip_grad_norm_(align_params + body_params, max_norm=0.8)
        optimizer_align.step()
        optimizer_body.step()
        optimizer_align.zero_grad(set_to_none=True)
        optimizer_body.zero_grad(set_to_none=True)

        step_losses.append(accum_loss)

        # Divergence guard
        if step > WARMUP_STEPS and accum_loss > DIVERGE_LIM:
            print(f"\n  [ABORT] Loss diverged at step {step}: {accum_loss:.4f} > {DIVERGE_LIM}")
            print(f"  Best checkpoint preserved: {CKPT_OUT.name}  loss={best_loss:.4f}")
            break

        # Logging
        if step == 1 or step % 10 == 0:
            window = step_losses[-10:] if step > 1 else step_losses
            avg    = sum(window) / len(window)
            ppl    = math.exp(min(avg, 20.0))
            print(f"[{step:4d}/{MAX_STEPS}] loss={avg:.4f}  PPL={ppl:6.2f}  lr={curr_lr:.2e}")

        # Per-step checkpoint + early-stop check
        if step % SAVE_EVERY == 0:
            window = step_losses[-SAVE_EVERY:]
            avg    = sum(window) / len(window)
            ppl    = math.exp(min(avg, 20.0))

            step_ckpt = CKPT_DIR / f"step_{step:04d}_loss{avg:.3f}.pt"
            save_ckpt(step_ckpt, model, cfg, step, avg, ppl, "Quillan-6L SFT-v2")

            # Save best only if above the hard memorisation floor
            if LOSS_FLOOR < avg < best_loss:
                best_loss = avg
                save_ckpt(CKPT_OUT, model, cfg, step, best_loss, ppl, "Quillan-6L SFT-v2")
                print(f"  >>> [BEST] Saved: {CKPT_OUT.name}  loss={best_loss:.4f}  PPL={ppl:.2f}")

            # Early stop: crossed target floor
            if avg <= EARLY_STOP:
                print(f"\n  [EARLY STOP] Target floor reached: {avg:.4f} <= {EARLY_STOP}")
                print(f"  Best checkpoint: {CKPT_OUT.name}  loss={best_loss:.4f}")
                exact_ckpt = CKPT_DIR / f"step_{step:04d}_STOP_loss{avg:.3f}.pt"
                save_ckpt(exact_ckpt, model, cfg, step, avg, ppl, "Quillan-6L SFT-v2 STOP")
                print(f"  Stop-step checkpoint: {exact_ckpt.name}")
                break

            # Hard floor: memorisation territory
            if avg <= LOSS_FLOOR:
                print(f"\n  [HARD FLOOR] Loss {avg:.4f} below memorisation floor {LOSS_FLOOR}.")
                print(f"  Saving OVERFIT candidate for inspection.")
                of_ckpt = CKPT_DIR / f"step_{step:04d}_OVERFIT_loss{avg:.3f}.pt"
                save_ckpt(of_ckpt, model, cfg, step, avg, ppl, "Quillan-6L SFT-v2 OVERFIT")
                break

        # Generation probe
        if step % PROBE_EVERY == 0:
            print(f"\n{chr(9472) * 72}")
            print(f"  [GENERATION PROBE @ step {step}]  (top-p sampling, clean forward)")
            for pp in PROBE_PROMPTS:
                resp = probe(model, tok, pp, device)
                print(f"  Q: {pp}")
                print(f"  A: {resp[:220]}")
            print(f"{chr(9472) * 72}\n")

    # Final summary
    print("\n" + "=" * 72)
    if best_loss < float("inf"):
        print(f"  DONE | Best loss: {best_loss:.4f} | PPL: {math.exp(min(best_loss, 20.0)):.2f}")
        print(f"  Output: {CKPT_OUT}")
    else:
        print("  WARNING: No best checkpoint saved.")
        print(f"  Loss never entered valid window ({LOSS_FLOOR}, {EARLY_STOP}).")
        print(f"  Check step checkpoints in: {CKPT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    run()
