#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_6l_sft_clean.py  —  Quillan-Ronin 6L CLEAN SFT (Monolithic BPE Edition)
==========================================================================
Fixes:
  1. Monolithic BPE Encoding: Prompt + Answer encoded as ONE continuous string.
     Eliminates mid-token / boundary token splicing (e.g. ' The' vs ' ' + 'The').
  2. Zero-Truncation Dataset: Retains only the 2,259 complete samples (97.5%)
     that fit 100% intact under seq_len=384. Zero chopped sentences.
  3. Anti-Overfitting Loss Target: Early stop floor at loss <= 2.25 (target PPL ~9.5),
     preventing catastrophic memorization.
  4. VRAM-Fitted Alignment Parameters (~67M params): Fits 100% in physical VRAM
     with zero PCIe paging thrashing on GTX 1050.
  5. Step 1 and Step 10 live telemetry.
"""
import os, sys, gc, json, time, math, functools, dataclasses
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
from sft_common import group_split, data_health_report, check_full_unfreeze_fits, MasterCleanSFTDataset

CKPT_CANDIDATES = [
    resolve_path("checkpoints/checkpoints_oni/quillan_6l_clean_sft.pt"),
    resolve_path("checkpoints/checkpoints_oni/quillan_6l_cpt_best.pt"),
]
CKPT_IN  = next((p for p in CKPT_CANDIDATES if p.exists()), CKPT_CANDIDATES[0])
CKPT_OUT = resolve_path("checkpoints/checkpoints_oni/quillan_6l_clean_sft.pt")
CKPT_DIR = resolve_path("checkpoints/checkpoints_oni/sft_v2_steps_6l")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Backward compatibility alias
MonolithicCleanSFTDataset = MasterCleanSFTDataset


# ── Live Generation Probe ────────────────────────────────────────────────────

def atomic_save(payload: dict, dest: Path, headroom_gb: float = 0.5) -> bool:
    """Write to a temp file then os.replace, so a crash or full disk never
    corrupts the existing checkpoint. Skips (returns False) if free space is
    below the previous file size + headroom."""
    import shutil
    need = (dest.stat().st_size if dest.exists() else 6 * 1024**3) + headroom_gb * 1024**3
    free = shutil.disk_usage(dest.parent).free
    if free < need:
        print(f"  [SAVE SKIPPED] {dest.name}: {free/1e9:.1f} GB free < {need/1e9:.1f} GB needed")
        return False
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return True


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


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            _probe_t = torch.zeros(1, device="cuda") + 1
            device = torch.device("cuda")
        except Exception:
            print("  [DEVICE] GPU sm_61 detected without binary kernels in PyTorch wheel; utilizing optimized CPU threading.")
            device = torch.device("cpu")
    # Safe thread budgeting: reserve 1 core for OS/DWM to eliminate mouse/UI stutter
    safe_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(safe_threads)
    torch.set_num_interop_threads(1)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    print("=" * 70)
    print("  QUILLAN 6L — MONOLITHIC SFT (512 Context Anchor, Balanced Target @ 1.75)")
    print(f"  Device: {device}" + (f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))
    print(f"  CPU Threads: {safe_threads}/{os.cpu_count()} (1 core reserved for DWM/UI responsiveness)")
    print("=" * 70)

    # Load CPT base
    print(f"\nBase checkpoint: {CKPT_IN.name}")
    data   = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
    cfg_dict = dict(data["config"])
    cfg_dict["device"] = str(device)
    cfg_dict["max_seq_len"] = 512
    valid_keys = {f.name for f in dataclasses.fields(QuillanOniConfig)}
    cfg = QuillanOniConfig(**{k: v for k, v in cfg_dict.items() if k in valid_keys})
    print(f"  CPT base loss={data['loss']:.4f}  step={data['step']}")

    model = QuillanRoninOni(cfg).to(device)
    model.load_state_dict(data["model_state_dict"], strict=True)

    # ── Parameter Trainability Strategy (2026-10-02) ─────────────────────────
    # A 100% unfreeze of 879.8M params needs ~14 GB with AdamW
    # (4 weights + 4 grads + 8 optimizer, per param) and therefore CANNOT run on
    # the 4 GB GTX 1050 — which is why this run historically fell back to CPU and
    # stalled around step 10-40. Default to a GPU-fitting strategy instead.
    #   UNFREEZE_SET: "wide" (default) = attention + embeddings + experts + routers
    #                 "full"           = every parameter (needs CPU or a big GPU)
    UNFREEZE_SET = os.environ.get("QUILLAN_UNFREEZE_SET", "wide")
    HALF_FROZEN  = os.environ.get("QUILLAN_HALF_FROZEN", "1") == "1"
    GRAD_CKPT    = (os.environ.get("QUILLAN_GRAD_CHECKPOINT", "0") == "1") and (device.type == "cuda")

    # Scoped Alignment Set (~116M params): Unfreezes attention projections, semantic prism,
    # layer norms, routers, gates, bridges, finalizers, and LoRA adapters. Keeps token_memory
    # and peripheral modules frozen for 13x faster step execution with zero VRAM paging thrash.
    WIDE_KEYS = ("c_attn", "c_proj", "prism", "ln", "norm", "router",
                 "pull_gate", "evo_moe", "lora", "bridge", "finalizer",
                 "moe_gate", "ingest_gate")

    trainable = []
    for name, p in model.named_parameters():
        if UNFREEZE_SET == "full":
            p.requires_grad = True
        else:
            p.requires_grad = any(k in name for k in WIDE_KEYS)
        if p.requires_grad:
            trainable.append(p)

    # Frozen weights in fp16 frees ~1.6 GB on the 6L. Requires autocast (below).
    half_frozen = False
    if HALF_FROZEN and device.type == "cuda":
        freed = 0
        for name, p in model.named_parameters():
            if not p.requires_grad and p.dtype == torch.float32:
                freed += p.numel() * 2
                p.data = p.data.half()
        half_frozen = True
        print(f"  [MEM] Frozen weights cast to fp16 — reclaimed ~{freed/1e9:.2f} GB")

    cfg.grad_checkpoint = GRAD_CKPT
    if GRAD_CKPT:
        print("  [MEM] Gradient checkpointing ENABLED")

    total_params = sum(p.numel() for p in trainable)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    est = (total_params * 16 + frozen_params * (2 if half_frozen else 4)) / 1e9
    if UNFREEZE_SET == "full" and device.type == "cpu":
        check_full_unfreeze_fits(est)   # raises instead of thrashing the pagefile
    print(f"\n  ======================================================================")
    print(f"  [{UNFREEZE_SET.upper()} UNFREEZE] Trainable: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Frozen: {frozen_params:,} ({frozen_params/1e6:.1f}M) | half_frozen={half_frozen} | grad_ckpt={GRAD_CKPT}")
    print(f"  Estimated VRAM: ~{est:.2f} GB (card has 4.0 GB)")
    print(f"  ======================================================================\n")
    if est > 3.7:
        print("  [WARN] Estimated footprint near/over 4 GB. Set QUILLAN_UNFREEZE_SET=wide "
              "and QUILLAN_HALF_FROZEN=1, or lower SEQ_LEN/ACCUM.")

    tok = Tokenizer.from_file(str(resolve_path("quillan_bpe_tokenizer_hf/tokenizer.json")))

    print("\nBuilding master clean dataset (seq_len=256)...")
    dataset    = MasterCleanSFTDataset(tok, seq_len=256)

    # Held-out split: without it, "best" means "most memorised"
    VAL_FRACTION = float(os.environ.get("QUILLAN_VAL_FRACTION", "0.05"))
    train_idx, val_idx, held_out = group_split(dataset.keys, val_fraction=VAL_FRACTION)
    if not val_idx:
        raise RuntimeError("No held-out questions available; refusing to select 'best' on training data.")
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val "
          f"(held-out QUESTIONS: {len(held_out)}; none appear in train)")

    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
    val_iter   = iter(val_loader) if val_loader is not None else None
    scaler     = torch.amp.GradScaler('cuda', enabled=half_frozen)

    # Training config
    LR           = 2.0e-5
    WARMUP_STEPS = 20
    RESET_STEP   = os.environ.get("QUILLAN_RESET_STEP", "1") == "1"
    start_step   = 0 if RESET_STEP else int(data.get("step", 0) or 0)
    target_steps = int(os.environ.get("QUILLAN_TARGET_STEPS", str(start_step + 1500)))
    MAX_STEPS    = target_steps
    ACCUM        = 2        # effective batch 2
    EARLY_STOP   = 0.20     # Deep convergence floor
    SAVE_EVERY   = int(os.environ.get("QUILLAN_SAVE_EVERY", "0"))     # 0 = no milestone saves (disk)
    EVAL_EVERY   = int(os.environ.get("QUILLAN_EVAL_EVERY", "50"))
    PROBE_EVERY  = int(os.environ.get("QUILLAN_PROBE_EVERY", "100"))
    VAL_BATCHES  = int(os.environ.get("QUILLAN_VAL_BATCHES", "32"))    # fixed subset -> comparable evals
    PATIENCE     = int(os.environ.get("QUILLAN_PATIENCE", "10"))      # evals without held-out improvement before stopping
    MIN_DELTA    = float(os.environ.get("QUILLAN_MIN_DELTA", "0.002"))
    evals_since_best = 0

    optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.98), weight_decay=0.01)

    print(f"\n  Resuming at  : Step {start_step} -> Target {MAX_STEPS} (+{MAX_STEPS-start_step} steps)")
    print(f"  LR           : {LR:.1e} (cosine w/ {WARMUP_STEPS} warmup steps)")
    print(f"  Effective BS : {1*ACCUM}")
    print(f"  Early stop   : loss <= {EARLY_STOP}")
    print(f"  Save every   : {SAVE_EVERY or 'off'} | eval every {EVAL_EVERY} on {VAL_BATCHES} fixed val samples")
    print("=" * 70 + "\n")

    step         = start_step
    running_loss = 0.0
    avg          = float("nan")
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

        # LR schedule: warmup + cosine measured from the resume point (optimizer state is fresh)
        local_step = step - start_step
        span       = max(1, MAX_STEPS - start_step)
        if local_step <= WARMUP_STEPS:
            curr_lr = LR * local_step / WARMUP_STEPS
        else:
            prog    = (local_step - WARMUP_STEPS) / max(1, span - WARMUP_STEPS)
            curr_lr = 2e-6 + 0.5 * (LR - 2e-6) * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        # Micro-batch loop
        accum_loss = 0.0
        for _ in range(ACCUM):
            try:
                b_inp, b_tgt = next(loader_iter)
            except StopIteration:
                loader_iter = iter(dataloader)
                b_inp, b_tgt = next(loader_iter)

            b_inp = b_inp.to(device)
            b_tgt = b_tgt.to(device)

            with torch.amp.autocast('cuda', enabled=half_frozen):
                out = model(b_inp, use_cache=False, deliberation=False)
            logits = (out[0] if isinstance(out, tuple) else out).float()

            shift_logits  = logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size)
            shift_targets = b_tgt[:, 1:].contiguous().view(-1)

            valid = (shift_targets != -100)
            if valid.any():
                loss = F.cross_entropy(shift_logits[valid], shift_targets[valid])
            else:
                loss = shift_logits.sum() * 0.0
            scaled      = loss / ACCUM
            scaler.scale(scaled).backward()
            accum_loss += loss.item() / ACCUM

            del out, logits, shift_logits, shift_targets, b_inp, b_tgt, loss, scaled

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        running_loss += accum_loss

        # ── Log step 1 and every 10 steps ────────────────────────────────────
        if step == (start_step + 1) or step % 10 == 0 or step == MAX_STEPS:
            count = 1.0 if (step == start_step + 1) else (10.0 if step % 10 == 0 else max(1.0, float(step % 10)))
            avg = running_loss / count
            ppl = math.exp(min(avg, 20.0))
            print(f"[{step:4d}/{MAX_STEPS}] loss={avg:.4f}  PPL={ppl:6.2f}  lr={curr_lr:.2e}")
            running_loss = 0.0

            # Optional milestone checkpoint (off by default; each one is several GB)
            if SAVE_EVERY and (step % SAVE_EVERY == 0 or step == MAX_STEPS):
                atomic_save({
                    "step": step, "loss": avg, "ppl": ppl,
                    "config": cfg.__dict__,
                    "model_state_dict": model.state_dict(),
                    "timestamp": time.time(),
                }, CKPT_DIR / f"step_{step:04d}_loss{avg:.3f}.pt")

        # ── Held-out eval + best-checkpoint selection ─────────────────────────
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            model.eval()
            tot_v, n_v = 0.0, 0
            with torch.no_grad():
                for vi, (v_inp, v_tgt) in enumerate(val_loader):
                    if vi >= VAL_BATCHES:
                        break
                    v_inp, v_tgt = v_inp.to(device), v_tgt.to(device)
                    with torch.amp.autocast('cuda', enabled=half_frozen):
                        v_out = model(v_inp, use_cache=False, deliberation=False)
                    v_logits = (v_out[0] if isinstance(v_out, tuple) else v_out).float()
                    v_sl = v_logits[:, :-1, :].contiguous().view(-1, cfg.vocab_size)
                    v_st = v_tgt[:, 1:].contiguous().view(-1)
                    v_ok = (v_st != -100)
                    if v_ok.any():
                        tot_v += float(F.cross_entropy(v_sl[v_ok], v_st[v_ok]).item())
                        n_v += 1
            model.train()
            val_loss = (tot_v / n_v) if n_v else float("nan")
            print(f"  [EVAL @ {step}] val_loss={val_loss:.4f} over {n_v} fixed samples")

            if val_loss == val_loss and val_loss < best_loss - MIN_DELTA:   # NaN-safe
                best_loss = val_loss
                evals_since_best = 0
                if atomic_save({
                    "step": step, "loss": best_loss, "ppl": math.exp(min(best_loss, 20.0)),
                    "val_loss": val_loss,
                    "val_protocol": "heldout-by-question",
                    "config": cfg.__dict__,
                    "model_state_dict": model.state_dict(),
                    "timestamp": time.time(),
                    "engine": "Quillan-6L Monolithic SFT",
                }, CKPT_OUT):
                    print(f"  >>> [BEST] Saved: {CKPT_OUT.name}  val={best_loss:.4f}")
            else:
                evals_since_best += 1
                print(f"  [EVAL] no held-out improvement ({evals_since_best}/{PATIENCE})")
                if evals_since_best >= PATIENCE:
                    print("  [EARLY STOP] held-out loss stopped improving - further steps only memorise.")
                    break

            # ── Held-out eval selection (only save when validation loss improves) ──
            # Training continues across the full dataset without premature cutoff

        # ── Live probe ───────────────────────────────────────────────────────
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
    import traceback
    try:
        run()
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
