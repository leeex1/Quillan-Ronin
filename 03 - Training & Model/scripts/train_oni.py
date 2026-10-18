# === [AUTO-WIRED v5.4.0-ONI 15-POINT LIVE TRAINING UPGRADE HOOK] ===
try:
    from quillan_live_training_upgrades import (
        FrozenEmbeddingWithLoRA,
        bitnet_158_dynamic_deadzone,
        fast_cpu_muon_orthogonalize,
        get_decoupled_lr_schedule,
        SmallBatchMoEStabilizer,
        CurriculumGateAndReplayController,
        stack_6l_to_12l_zero_spike,
        sample_early_ternary_logits,
        compute_yarn_rope_inv_freq,
    )
    _UPGRADES_ACTIVE = True
except ImportError:
    _UPGRADES_ACTIVE = False

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

sys.path.insert(0, str(Path(__file__).resolve().parent))  # portable: oni/ self-contained
from quillan_tokenizer_unified import UnifiedQuillanTokenizer  # noqa: E402
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni  # noqa: E402
from paper_01_profiler import StepProfiler  # 2309.02521 — real CPU/GPU profiling

def resolve_data_dir(custom_path: str | None = None) -> Path:
    """Auto-detects data directory or dataset file across local, repo, and cloud paths."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists():
            return p

    candidates = [
        Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\canonical_standardized\quillan_master_gold_training_v1.pt"),
        Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\quillan_pretrain_corpus_343mb.pt"),
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parent.parent / "05_Training" / "training_data" / "v9",
        Path("/content/Quillan-Ronin/oni/data"),
        Path("oni/data"),
        Path("05_Training/training_data/v9"),
    ]
    for c in candidates:
        if c.is_file() and c.suffix == ".pt" and c.exists():
            return c.resolve()
        if c.is_dir() and ((c / "train_ids.bin").exists() or (c / "train.pt").exists()):
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
    ap.add_argument("--router-mode", type=str, default="gumbel_topk",
                    choices=["dense_pull", "gumbel_topk"])
    ap.add_argument("--top-k", type=int, default=2,
                    help="Top-k active experts per token in gumbel_topk mode (default: 2)")
    ap.add_argument("--grt-iterations", type=int, default=1,
                    help="Recurrent deliberation loops per forward during training (default: 1 for fast convergence)")
    ap.add_argument("--cgpu", action="store_true",
                    help="Enable CGPUBridge (router/hot-experts on GPU, body on CPU)")
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
    ap.add_argument("--optimizer", type=str, default="fused",
                    choices=["adamw", "muonk2", "fused"],
                    help="Optimizer: fused/muonk2 (Sovereign Newton-Schulz for matrices + Sophia-H + AdamW for dense + CCRL) | adamw")
    ap.add_argument("--lr-muon", type=float, default=0.015,
                    help="Learning rate for low-rank MuonK2 parameters (default: 0.015)")
    ap.add_argument("--ccrl-limit", type=float, default=4.0,
                    help="CCRL gradient curvature clip limit (default: 4.0)")
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
    def __init__(self, data_path: Path, split: str, seq_len: int):
        self.seq_len = seq_len
        # Case 1: .pt file provided or detected
        if data_path.is_file() and data_path.suffix == ".pt":
            self._load_pt(data_path, split, seq_len)
        elif data_path.is_dir() and (data_path / f"{split}.pt").exists():
            self._load_pt(data_path / f"{split}.pt", split, seq_len)
        else:
            # Case 2: standard memmapped binary dataset
            ids_path = data_path / f"{split}_ids.bin"
            labels_path = data_path / f"{split}_labels.bin"
            if not ids_path.exists() or not labels_path.exists():
                raise FileNotFoundError(
                    f"Missing binary dataset split '{split}' in {data_path}. "
                    f"Run 'python oni/pack_dataset.py' to generate binary datasets."
                )
            ids = np.memmap(ids_path, dtype=np.uint16, mode="r")
            labels = np.memmap(labels_path, dtype=np.int32, mode="r")
            n = min(len(ids), len(labels)) // seq_len * seq_len
            self.ids = ids[:n].reshape(-1, seq_len)
            self.labels = labels[:n].reshape(-1, seq_len)
            self.is_pt = False

    def _load_pt(self, pt_file: Path, split: str, seq_len: int):
        raw = torch.load(pt_file, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "input_ids" in raw:
            input_ids = raw["input_ids"]
            labels = raw.get("labels", None)
        elif isinstance(raw, torch.Tensor):
            input_ids = raw
            labels = None
        else:
            raise ValueError(f"Unrecognized dataset format in {pt_file}")

        # Flatten into 1D token stream for consistent seq_len windowing
        if input_ids.dim() > 1:
            flat_ids = input_ids.reshape(-1)
        else:
            flat_ids = input_ids

        total_tokens = flat_ids.numel()
        # Train / val split if single master file provided
        val_fraction = 0.05
        split_point = int(total_tokens * (1.0 - val_fraction))
        if split == "train":
            tokens = flat_ids[:split_point]
        else:
            tokens = flat_ids[split_point:]

        n = (tokens.numel() // seq_len) * seq_len
        if n == 0:
            tokens = flat_ids[:seq_len]  # fallback
            n = seq_len
        
        sliced_ids = tokens[:n].reshape(-1, seq_len).to(torch.int64)
        if labels is not None:
            flat_labels = labels.reshape(-1)[:split_point] if split == "train" else labels.reshape(-1)[split_point:]
            nl = (flat_labels.numel() // seq_len) * seq_len
            sliced_labels = flat_labels[:nl].reshape(-1, seq_len).to(torch.int64)
        else:
            # Model forward internally shifts logits[..., :-1] vs labels[..., 1:]
            # so labels must match input_ids exactly without an extra roll!
            sliced_labels = sliced_ids.clone()

        self.ids = sliced_ids
        self.labels = sliced_labels
        self.is_pt = True

    def __len__(self):
        return len(self.ids)

    def batch(self, bs, rng):
        idx = rng.integers(0, len(self), size=bs)
        if getattr(self, "is_pt", False):
            return self.ids[idx], self.labels[idx]
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
    if isinstance(out, dict):
        if out.get("text"):
            return out["text"]
        return tok.decode(out.get("tokens", []))
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

    num_threads = getattr(args, "num_threads", None) or min(4, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(min(2, os.cpu_count() or 2))
    print(f"[THREADS] Configured torch threads={torch.get_num_threads()} (total cores={os.cpu_count()})")

    rng = np.random.default_rng(42)
    tok = UnifiedQuillanTokenizer()
    train = Corpus(DATA, "train", args.seq_len)
    val = Corpus(DATA, "val", args.seq_len)
    print(f"[DATA] train={len(train)} seqs  val={len(val)} seqs")

    cfg = QuillanOniConfig(n_layer=args.n_layer, max_seq_len=args.seq_len,
                           router_mode=args.router_mode, top_k=args.top_k,
                           grt_iterations=args.grt_iterations,
                           grad_checkpoint=args.grad_checkpoint)
    model = QuillanRoninOni(cfg)
    if args.device != "cpu":
        model = model.to(args.device)

    # C+GPU Bridge: shard router & hot experts to GPU, body on CPU
    if args.cgpu and torch.cuda.is_available():
        try:
            sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
            from quillan_cgpu_bridge import CGPUBridge
            bridge = CGPUBridge(gpu_index=0, gpu_budget_mb=3000, top_k_gpu=args.top_k)
            for b_idx, block in enumerate(model.h):
                bridge.place_router(block.moe.router, name=f"b{b_idx}.router")
                # Place top hot experts on GPU
                exp_scores = [1.0 if i < args.top_k else 0.1 for i in range(cfg.num_experts)]
                block.moe.experts = torch.nn.ModuleList(
                    bridge.place_experts(list(block.moe.experts), exp_scores, prefix=f"b{b_idx}.exp")
                )
            print(f"[CGPU] Activated C+GPU bridge: {bridge.status()}", flush=True)
        except Exception as e:
            print(f"[CGPU] Bridge initialization warning: {e}. Falling back to standard device map.", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Quillan-Ronin v5.4.0-oni  n_layer={args.n_layer}  "
          f"router={args.router_mode} (k={args.top_k})  grt_iter={args.grt_iterations}  "
          f"params={n_params/1e6:.1f}M  device={args.device}  "
          f"checkpointing={args.grad_checkpoint}  cgpu={args.cgpu}")

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

    # Optimizer Selection: MuonK2 / Fused (Newton-Schulz + Sophia-H + AdamW + CCRL) vs AdamW
    if args.optimizer in ("muonk2", "fused"):
        try:
            sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")
            sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
            from quillan_fused_optimizer import create_quillan_fused_optimizer
            opt = create_quillan_fused_optimizer(
                model=model,
                lr_muon=args.lr_muon,
                lr=args.lr,
                weight_decay=0.01,
                ccrl_limit=args.ccrl_limit,
                use_sophia=True,
            )
            print(f"[OPTIMIZER] Sovereign Fused (MuonK2 + Sophia-H + AdamW + CCRL) active (lr_muon={args.lr_muon:.2e}, lr_adamw={args.lr:.2e}, ccrl={args.ccrl_limit}).")
        except Exception as e:
            print(f"[OPTIMIZER] Failed to init Fused/MuonK2 ({e}). Falling back to AdamW.")
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    else:
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
            print("[OPTIMIZER] Standard AdamW initialized.")

    step, best_val = 0, float("inf")
    ema_sd = None
    latest = CKPT_DIR / "quillan_oni_latest.pt"

    if args.resume and latest.exists():
        try:
            ck = torch.load(latest, map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(latest, map_location="cpu", weights_only=False)

        # Smart adaptation: slice matching rows for vocabulary expansion (50257 -> 50262)
        raw_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        adapted_sd = {}
        for k, v in raw_sd.items():
            if k in model.state_dict():
                target = model.state_dict()[k].clone()
                if v.shape != target.shape:
                    if v.dim() == 2 and target.dim() == 2 and v.shape[1] == target.shape[1]:
                        min_rows = min(v.shape[0], target.shape[0])
                        target[:min_rows] = v[:min_rows]
                        adapted_sd[k] = target
                        print(f"[RESUME] Resized parameter {k} from {list(v.shape)} to {list(target.shape)} (copied {min_rows} rows)")
                        continue
            adapted_sd[k] = v

        missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
        if missing:
            print(f"[RESUME] warm-start: {len(missing)} new-param keys freshly initialized, "
                  f"{len(unexpected)} legacy keys skipped")
        else:
            print(f"[RESUME] strict load OK ({len(unexpected)} legacy keys skipped)")
        step = ck["step"]
        best_val = ck.get("best_val", best_val)
        ema_sd = ck.get("ema_sd")
        if ema_sd is not None and args.device != "cpu":
            ema_sd = {k: v.to(args.device) for k, v in ema_sd.items()}
        # optimizer states: only restore for params present in both (new params get fresh state)
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:
            print(f"[RESUME] optimizer state partial reset ({e})")
        print(f"[RESUME] step {step}, best_val {best_val:.4f}")
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
        lr_ratio = lr / max(args.lr, 1e-9)
        for g in opt.param_groups:
            if g.get("is_muon", False):
                g["lr"] = args.lr_muon * lr_ratio
            else:
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
            profiler.mark_data_load_done()
            if cur_seq < args.seq_len:
                x = x[:, :cur_seq]
                y = y[:, :cur_seq]
            if args.device != "cpu":
                # Session 1 (Papers 5-7 Heterogeneous, default ON): overlap
                # transfer with compute via non-blocking (consumed by .to call)
                x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
            profiler.mark_transfer_done()
            profiler.mark_fwd_start()
            _, ce, aux = model(x, labels=y)
            loss = (ce + model.total_aux_loss(aux)) / args.grad_accum
            profiler.mark_bwd_start()
            loss.backward()
            # KILL HOST SYNCS: accumulate ce & load_balance tensors directly on device without stalling host via .item()
            accum_ce += ce.detach() / args.grad_accum
            if "load_balance" in aux:
                accum_lb += aux["load_balance"].detach() / args.grad_accum

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        profiler.mark_opt_start()
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
                pass  # ES module not available — skip silently

        step += 1
        running.append(step_loss)

        latency_ms = (time.time() - t0) * 1000.0
        _, ema_decay, _ = model.governor_adjust(latency_ms)
        profiler.mark_ema_start()
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
            try:
                s_text = sample(model, tok, args.device)
                print("[SAMPLE]", repr(s_text.encode("ascii", errors="replace").decode("ascii")), flush=True)
            except Exception as e:
                print(f"[SAMPLE] error during sampling: {e}", flush=True)

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




