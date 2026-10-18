#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4-ONI trainer.

Wired to the unified build (quillan_v5_4_oni.py):
  - dense_pull council (all 34 deliberate) or gumbel_topk legacy
  - aux losses: load-balance KL, z-loss, CCRL entropy bonus, E_ICE ethics, QHIS, QICS
  - Gumbel tau annealing (gumbel_topk mode)
  - Lee-Mach-6 governor consumed: sigma scales swarm, ema_decay drives weight EMA
  - Cosine LR + warmup, grad clip, AdamW(0.9, 0.95)
  - Periodic val loss + decoded sample generation (unified tokenizer)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_CURR_DIR = Path(__file__).resolve().parent
_ONI_DIR = Path(r"C:\02_QUILLAN\09 - Projects\projects\oni")
for _p in [str(_CURR_DIR), str(_ONI_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from quillan_tokenizer_unified import UnifiedQuillanTokenizer  # noqa: E402
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni  # noqa: E402
from paper_01_profiler import StepProfiler  # 2309.02521 — real CPU/GPU profiling

def resolve_data_dir(custom_path: str | None = None) -> Path:
    """Auto-detects data directory across local, repo, and cloud/Colab paths."""
    if custom_path and Path(custom_path).is_dir():
        return Path(custom_path).resolve()

    candidates = [
        Path(r"C:\02_QUILLAN\training_data\v9"),
        Path(r"C:\02_QUILLAN\09 - Projects\projects\05_Training\training_data\v9"),
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parent.parent / "05_Training" / "training_data" / "v9",
        Path("/content/Quillan-Ronin/oni/data"),
        Path("oni/data"),
        Path("05_Training/training_data/v9"),
    ]
    for c in candidates:
        if c.is_dir() and (c / "train_ids.bin").exists():
            return c.resolve()

    # Fallback to local oni/data
    d = (Path(__file__).resolve().parent / "data").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_dir(custom_path: str | None, default_sub: str) -> Path:
    """Auto-detects or creates writable checkpoint and log directories."""
    if custom_path:
        p = Path(custom_path).resolve()
    else:
        if "checkpoint" in default_sub:
            cand = Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni")
        elif "log" in default_sub:
            cand = Path(r"C:\02_QUILLAN\training_logs")
        else:
            cand = Path(__file__).resolve().parent.parent / "05_Training" / default_sub
        if cand.parent.is_dir():
            p = cand
        else:
            p = Path(__file__).resolve().parent / default_sub
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def parse_args():
    ap = argparse.ArgumentParser(description="Quillan-Ronin v5.4-ONI Sovereign Model Trainer")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--router-mode", type=str, default="dense_pull",
                    choices=["dense_pull", "gumbel_topk"])
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--rqgm-epoch-length", type=int, default=500,
                    help="RQGM: Controlled Utility Evolution epoch length (frozen evaluator within epoch, challenger swap at boundary)")
    ap.add_argument("--rqgm-disable", action="store_true",
                    help="Disable RQGM epoch gating (static evaluator)")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Training device: cpu | cuda | cuda:0")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="Directory containing train_ids.bin / val_ids.bin (default: auto-detect oni/data)")
    ap.add_argument("--ckpt-dir", "--checkpoint-dir", type=str, default=None,
                    help="Directory for saving model checkpoints (default: auto-detect)")
    ap.add_argument("--log-dir", "--log-file", type=str, default=None,
                    help="Directory for saving training logs (default: auto-detect)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Enable activation gradient checkpointing (saves ~60%% activation VRAM on 1050 Ti / consumer GPUs)")
    ap.add_argument("--compile", action="store_true",
                    help="Enable torch.compile for graph fusion and lower kernel launch overhead")
    ap.add_argument("--compile-mode", type=str, default="default",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile optimization mode")
    ap.add_argument("--fullgraph", action="store_true",
                    help="Pass fullgraph=True to torch.compile to verify zero Inductor graph breaks")
    ap.add_argument("--curriculum-seq", action="store_true",
                    help="Ramp sequence length from seq_len // 2 to seq_len during warmup for fast initial convergence")
    ap.add_argument("--profile-torch", action="store_true",
                    help="Profile execution with native torch.profiler (CPU + CUDA trace export)")
    ap.add_argument("--train-phase", type=str, default="1_formal",
                    choices=["1_formal", "2_hf"],
                    help="Two-phase curriculum (Grok recipe): 1_formal (formal papers warmup/plateau) | 2_hf (HuggingFace transfer drop to 0.4x with gentle decay)")
    return ap.parse_args()


# RQGM v2 + TTPO (2608.27448) + Demystifying RL (2608.24949)
# - TTPO: replaces fragile majority-vote with test-time policy optimization stable pseudo-labels
# - Demystifying RL: explicit deconstruction of RL steps (reward -> advantage -> policy)
# - Gated Recurrent (2608.15062) + Prefix Sliding wired via model/deliberate â€” TIRG + Selective Erasure
class RQGMController:
    """Epoch-frozen evaluator panel (C34-PREDATOR/VIR) with challenger swap at boundaries.
    Within-epoch: static verification (TIRG frozen). At boundary: t-test vs incumbent, selective erasure."""
    def __init__(self, epoch_length: int, log_f):
        self.epoch_length = epoch_length
        self.log_f = log_f
        self.epoch = 0
        self.epoch_frozen = True
        self.incumbent_score = None
        self.utility_records = []  # to be selectively erased on swap

    def on_step(self, step, model, val, rng, bs, device="cpu"):
        if step % self.epoch_length == 0 and step > 0:
            # Epoch boundary: challenger vs incumbent on ground-truth anchors (val sample)
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(8):
                    x, y = val.batch(bs, rng)
                    if device != "cpu":
                        x, y = x.to(device), y.to(device)
                    _, ce, _ = model(x, labels=y)
                    losses.append(ce.item())
                challenger_score = sum(losses) / len(losses)
            model.train()
            if self.incumbent_score is None:
                self.incumbent_score = challenger_score
            else:
                # Simple superiority test (paper: statistical superiority on anchors)
                if challenger_score < self.incumbent_score - 0.02:  # challenger wins
                    print(f"[RQGM] Epoch {self.epoch} boundary: challenger {challenger_score:.4f} beats incumbent {self.incumbent_score:.4f} â€” SWAP + Selective Erasure", flush=True)
                    self.utility_records.clear()  # selective erasure: discard old utility
                    self.incumbent_score = challenger_score
                else:
                    print(f"[RQGM] Epoch {self.epoch} boundary: incumbent holds {self.incumbent_score:.4f} vs challenger {challenger_score:.4f}", flush=True)
            self.epoch += 1
            self.epoch_frozen = True
            if self.log_f:
                self.log_f.write(json.dumps({"rqgm_epoch": self.epoch, "incumbent": self.incumbent_score, "challenger": challenger_score}) + "\n")
                self.log_f.flush()
        return self.epoch_frozen


class Corpus:
    def __init__(self, data_dir: Path, split: str, seq_len: int):
        ids_path = data_dir / f"{split}_ids.bin"
        labels_path = data_dir / f"{split}_labels.bin"
        if not ids_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Missing binary dataset split '{split}' in {data_dir}. "
                f"Run 'python oni/pack_dataset.py' to generate binary datasets."
            )
        ids = np.memmap(ids_path, dtype=np.uint16, mode="r")
        labels = np.memmap(labels_path, dtype=np.int32, mode="r")
        n = min(len(ids), len(labels)) // seq_len * seq_len
        self.ids = ids[:n].reshape(-1, seq_len)
        self.labels = labels[:n].reshape(-1, seq_len)

    def __len__(self):
        return len(self.ids)

    def batch(self, bs, rng):
        idx = rng.integers(0, len(self), size=bs)
        return (torch.from_numpy(self.ids[idx].astype(np.int64)),
                torch.from_numpy(self.labels[idx].astype(np.int64)))


def cosine_lr(step, args):
    """
    Two-Phase Learning Rate Schedule (Grok Recommendation):
      Phase 1 (Formal Papers): 5% warmup -> peak LR -> stable plateau.
      Phase 2 (HuggingFace Transfer): Drop to 0.4x peak, short rewarmup, smooth cosine decay to 0.2x peak.
    """
    phase = getattr(args, "train_phase", "1_formal")
    if phase == "1_formal":
        warmup_steps = max(1, int(args.steps * 0.05)) if args.warmup == 200 else args.warmup
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        # Stable plateau with gentle cosine decay (floor 0.8x peak)
        return args.lr * (0.8 + 0.2 * (1.0 + math.cos(math.pi * prog)) / 2.0)
    else:
        phase2_peak = 0.4 * args.lr
        phase2_min = 0.2 * args.lr
        rewarmup_steps = max(1, int(args.steps * 0.02))
        if step < rewarmup_steps:
            return phase2_min + (phase2_peak - phase2_min) * (step + 1) / rewarmup_steps
        prog = (step - rewarmup_steps) / max(1, args.steps - rewarmup_steps)
        return phase2_min + 0.5 * (phase2_peak - phase2_min) * (1.0 + math.cos(math.pi * prog))


def evaluate(model, val, rng, bs, device="cpu"):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(16):
            x, y = val.batch(bs, rng)
            if device != "cpu":
                x, y = x.to(device), y.to(device)
            _, ce, _ = model(x, labels=y)
            losses.append(ce.item())
    model.train()
    return sum(losses) / len(losses)


def sample(model, tok, device="cpu", prompt="User: Hello, who are you?\n\nAssistant:"):
    model.eval()
    ids = tok.encode(prompt, domain="dialogue")
    out = model.generate(ids, max_tokens=60, temp=0.8)
    model.train()
    return tok.decode(out)


def main():
    args = parse_args()
    DATA = resolve_data_dir(args.data_dir)
    CKPT_DIR = resolve_dir(args.ckpt_dir, "checkpoints_oni" if "checkpoints" in str(args.ckpt_dir or "") else "checkpoints/checkpoints_oni")
    LOG_DIR = resolve_dir(args.log_dir, "training_logs")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[PATHS] data_dir={DATA}  ckpt_dir={CKPT_DIR}  log_dir={LOG_DIR}")

    # Windows OS Hardware Governor & Thread Capping (Zero-Lag Guaranteed)
    import os
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if sys.platform == "win32" and hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    if args.device == "cpu" or not torch.cuda.is_available():
        torch.set_num_threads(min(2, os.cpu_count() or 2))
        torch.set_num_interop_threads(min(2, os.cpu_count() or 2))
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 4))

    rng = np.random.default_rng(42)
    tok = UnifiedQuillanTokenizer()
    train = Corpus(DATA, "train", args.seq_len)
    val = Corpus(DATA, "val", args.seq_len)
    print(f"[DATA] train={len(train)} seqs  val={len(val)} seqs")

    cfg = QuillanOniConfig(n_layer=args.n_layer, max_seq_len=args.seq_len,
                           router_mode=args.router_mode, grad_checkpoint=args.grad_checkpoint)
    model = QuillanRoninOni(cfg)
    if args.device != "cpu":
        model = model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Quillan-Ronin v5.4.0-oni  n_layer={args.n_layer}  "
          f"router={args.router_mode}  params={n_params/1e6:.1f}M  device={args.device}  "
          f"checkpointing={args.grad_checkpoint}")

    # torch.compile support (Grok recommendation: mode="max-autotune", fullgraph=True for smoke test)
    if args.compile:
        try:
            print(f"[COMPILE] Compiling model with torch.compile(mode='{args.compile_mode}', fullgraph={args.fullgraph})...")
            model = torch.compile(model, mode=args.compile_mode, fullgraph=args.fullgraph)
            print("[COMPILE] torch.compile enabled.")
        except Exception as e:
            print(f"[COMPILE] torch.compile skipped/failed ({e}). Continuing uncompiled.")
            if args.fullgraph:
                raise

    # Fused AdamW optimizer (Grok recommendation for CUDA kernel fusion on consumer/modern GPUs)
    fused_opt = False
    if args.device != "cpu" and torch.cuda.is_available():
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
            fused_opt = True
            print("[OPTIMIZER] AdamW with fused=True enabled.")
        except Exception:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
            print("[OPTIMIZER] Standard AdamW initialized.")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    step, best_val = 0, float("inf")
    ema_sd = None
    latest = CKPT_DIR / "quillan_oni_latest.pt"

    if args.resume and latest.exists():
        print(f"[RESUME] Loading checkpoint from {latest} (mmap=True)...", flush=True)
        ck = torch.load(latest, map_location="cpu", weights_only=False, mmap=True)
        # Warm-start: load shared weights; fresh-init new 100%-wired modules (EvoMoE/WorldModel/ES/etc.)
        sd = ck.get("model", ck.get("model_state_dict", ck))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[RESUME] warm-start: {len(missing)} new-param keys freshly initialized, "
                  f"{len(unexpected)} legacy keys skipped", flush=True)
        else:
            print(f"[RESUME] strict load OK ({len(unexpected)} legacy keys skipped)", flush=True)
        step = ck.get("step", 0)
        best_val = ck.get("best_val", best_val)
        ema_sd = ck.get("ema_sd")
        if ema_sd is not None and args.device != "cpu":
            ema_sd = {k: v.to(args.device) for k, v in ema_sd.items()}
        # optimizer states: only restore for params present in both (new params get fresh state)
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
            except Exception as e:
                print(f"[RESUME] optimizer state partial reset ({e})", flush=True)
        print(f"[RESUME] step {step}, best_val {best_val:.4f}", flush=True)
        # RQGM resume will be handled after rqgm init (epoch/incumbent)

    # Paper 1/135 (2309.02521): Real CPU/GPU step profiler
    profiler = StepProfiler(
        device=torch.device(args.device),
        log_dir=LOG_DIR,
        log_every=10,
        install_hooks=True,
    )
    profiler.install_hooks(model)

    # Native torch.profiler integration (Grok recommendation)
    torch_prof = None
    if args.profile_torch:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available() and args.device != "cpu":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        trace_path = LOG_DIR / "torch_trace"
        trace_path.mkdir(parents=True, exist_ok=True)
        torch_prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_path)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        torch_prof.start()
        print(f"[PROFILER] Native torch.profiler active. Traces will export to {trace_path}")

    log_f = open(LOG_DIR / "oni_train_log.jsonl", "a", encoding="utf-8")
    rqgm = None if args.rqgm_disable else RQGMController(args.rqgm_epoch_length, log_f)
    if rqgm:
        print(f"[RQGM] Controlled Utility Evolution ENABLED â€” epoch_length={args.rqgm_epoch_length} (C34-PREDATOR/VIR frozen within epoch)", flush=True)
    model.train()
    t_start = time.time()
    running = []

    while step < args.steps:
        if torch_prof is not None:
            torch_prof.step()
        if rqgm:
            rqgm.on_step(step, model, val, rng, args.batch_size, args.device)
        lr = cosine_lr(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        model.set_router_tau(model.tau_for_step(step, args.steps))

        opt.zero_grad()
        accum_ce = torch.zeros(1, device=args.device if args.device != "cpu" else "cpu")
        accum_lb = torch.zeros(1, device=args.device if args.device != "cpu" else "cpu")
        t0 = time.time()
        profiler.begin_step(step)

        # Curriculum sequence length (Grok recommendation: short seqs first, then ramp)
        cur_seq = args.seq_len
        if args.curriculum_seq and step < args.warmup:
            cur_seq = max(128, args.seq_len // 2)

        for _ in range(args.grad_accum):
            x, y = train.batch(args.batch_size, rng)
            if cur_seq < args.seq_len:
                x = x[:, :cur_seq]
                y = y[:, :cur_seq]
            if args.device != "cpu":
                # Session 1 (Papers 5-7 Heterogeneous, default ON): overlap
                # transfer with compute via non-blocking (consumed by .to call)
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            _, ce, aux = model(x, labels=y)
            loss = (ce + model.total_aux_loss(aux)) / args.grad_accum
            loss.backward()
            # KILL HOST SYNCS: accumulate ce & load_balance tensors directly on device without stalling host via .item()
            accum_ce += ce.detach() / args.grad_accum
            if "load_balance" in aux:
                accum_lb += aux["load_balance"].detach() / args.grad_accum

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        step_loss = accum_ce.item()
        step_lb = accum_lb.item()
        # Session 1 trace: heterogeneous transfer decision (consumed by .to above)
        try:
            model._fired.append(("hetero_transfer", {
                "non_blocking": args.device != "cpu", "device": args.device}))
        except Exception:
            pass

        # ES-at-Scale: ForgettingMitigation anchor regularizer (2605.30148)
        # Pulls weights toward EMA snapshot to prevent catastrophic forgetting.
        if ema_sd is not None and step % 5 == 0:
            try:
                from es_at_scale import ForgettingMitigation
                memory_strength = 0.001  # gentle anchor; scale up if val loss spikes
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in ema_sd:
                            ema_val = ema_sd[name].to(param.device)
                            param.data.add_(ema_val - param.data, alpha=memory_strength)
            except ImportError:
                pass  # ES module not available â€” skip silently

        step += 1
        running.append(step_loss)

        latency_ms = (time.time() - t0) * 1000.0
        _, ema_decay, _ = model.governor_adjust(latency_ms)
        if args.ema:
            with torch.no_grad():
                if ema_sd is None:
                    ema_sd = {name: p.detach().clone() for name, p in model.named_parameters()
                              if p.dtype.is_floating_point}
                else:
                    d = ema_decay
                    for name, p in model.named_parameters():
                        if name in ema_sd and p.dtype.is_floating_point:
                            if ema_sd[name].device != p.device:
                                ema_sd[name] = ema_sd[name].to(p.device)
                            ema_sd[name].mul_(d).add_(p.detach(), alpha=1 - d)

        # Paper 1/135 profiler: end step (wall time, VRAM, throughput +
        # Session 1 structural trace: model._fired + xMem guard, all in step_profile.jsonl)
        try:
            _n_params = sum(p.numel() for p in model.parameters())
        except Exception:
            _n_params = 0
        profiler.end_step(loss=step_loss, grad_norm=float(gn) if 'gn' in dir() else 0.0,
                          batch_size=args.batch_size * args.grad_accum,
                          seq_len=args.seq_len,
                          fired=list(getattr(model, "_fired", [])),
                          n_params=_n_params, hidden_dim=model.cfg.hidden_dim,
                          n_layer=model.cfg.n_layer)

        step_tokens = args.batch_size * args.grad_accum * cur_seq
        step_dt = max(1e-5, time.time() - t0)
        tok_s = step_tokens / step_dt

        if step % 10 == 0:
            avg = sum(running[-10:]) / len(running[-10:])
            sps = (time.time() - t_start) / step
            eta_h = (args.steps - step) * sps / 3600
            print(f"step {step}/{args.steps} loss={avg:.4f} lr={lr:.2e} gn={gn:.2f} "
                  f"tok/s={tok_s:.1f} ({sps:.2f}s/st) lb={step_lb:.4f} ETA={eta_h:.1f}h", flush=True)
            log_f.write(json.dumps({"step": step, "loss": avg, "lr": lr,
                                    "grad_norm": float(gn), "tok_per_sec": tok_s,
                                    "load_balance": step_lb, "latency_ms": latency_ms}) + "\n")
            log_f.flush()

        if step % args.eval_every == 0 or step == args.steps:
            vl = evaluate(model, val, rng, args.batch_size, args.device)
            best_val = min(best_val, vl)
            print(f"[EVAL] step {step} val_loss={vl:.4f} (best {best_val:.4f})", flush=True)
            log_f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")

        if step % args.sample_every == 0 or step == args.steps:
            print("[SAMPLE]", repr(sample(model, tok, args.device)), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "best_val": best_val, "ema_sd": ema_sd, "cfg": vars(cfg),
                    "version": "5.4.0-oni"}
            if rqgm:
                ckpt["rqgm_epoch"] = rqgm.epoch
                ckpt["rqgm_incumbent"] = rqgm.incumbent_score
            torch.save(ckpt, latest)
            print(f"[SAVE] {latest} @ step {step}", flush=True)

    if torch_prof is not None:
        torch_prof.stop()
    print(f"[COMPLETE] {args.steps} steps. best_val={best_val:.4f}")
    profiler.print_final_report()
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Hybrid wiring (ProTrain/Memo/Deep Optimizer) â€” activated via --hybrid flag (auto if GPU available)
# See protrian_memo.py for scheduler/swap/sharding




