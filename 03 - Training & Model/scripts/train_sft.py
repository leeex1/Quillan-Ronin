#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — SUPERVISED FINE-TUNING (SFT) SCRIPT (2026)
==================================================================
Phase 2 of training: Takes the pre-trained checkpoint (step 2500, loss ~0.12)
and fine-tunes on chat-formatted instruction data.

Key differences from pre-training (train_sovereign.py):
  1. Chat-formatted JSONL data with <|system|>, <|user|>, <|assistant|> markers
  2. Loss masking: only compute loss on ASSISTANT tokens (not prompts/system)
  3. Lower LR: 1e-5 base / 5e-5 LoRA (vs 5e-5 / 2e-4 in pre-training)
  4. Shorter warmup (2% of steps)
  5. Converges much faster — 500-1000 steps is typical

Usage:
  cd C:\\Users\\Admin\\Quillan-Ronin
  python scripts\\train_sft.py [--steps 1000] [--resume] [--dry-run]
"""
import os, sys, gc, time, json, math, argparse, warnings

# ── CRITICAL: Disable CUDA entirely before torch import ──────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── Force UTF-8 on Windows ────────────────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, str(ROOT / '_dev'))
sys.path.insert(0, str(ROOT))

from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from quillan_bpe_tokenizer import QuillanBPETokenizer

# ── Hardware detection ────────────────────────────────────────────────────────
def detect_device():
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 7:
            print(f"  [DEVICE] CUDA sm_{cap[0]}{cap[1]} — GPU training enabled")
            return 'cuda'
        else:
            name = torch.cuda.get_device_name(0)
            print(f"  [DEVICE] {name} (sm_{cap[0]}{cap[1]}) is below sm_70 — forcing CPU")
    else:
        print("  [DEVICE] No CUDA available — CPU mode")
    return 'cpu'

# ── LanceDB training guard ────────────────────────────────────────────────────
def patch_lancedb_training_guard():
    try:
        from quillan_v8_saturated import QuillanAgenticExecutor
        _orig_flush = QuillanAgenticExecutor._flush_to_persistent
        def _safe_flush(self, *args, **kwargs):
            if getattr(self, '_training_guard', False): return
            return _orig_flush(self, *args, **kwargs)
        QuillanAgenticExecutor._flush_to_persistent = _safe_flush
        _orig_recall = QuillanAgenticExecutor._tool_memory_recall
        def _safe_recall(self, payload, sovereign):
            if getattr(self, '_training_guard', False):
                return {"recalled_memories": [], "historical_prism_avg": {}, "count": 0}
            return _orig_recall(self, payload, sovereign)
        QuillanAgenticExecutor._tool_memory_recall = _safe_recall
        print("  [PATCH] LanceDB training guard applied")
    except Exception as e:
        print(f"  [PATCH] Warning: LanceDB patch failed ({e})")

patch_lancedb_training_guard()

# ── Router Z-Loss ─────────────────────────────────────────────────────────────
def router_z_loss(router_logits: torch.Tensor, coeff: float = 1e-3) -> torch.Tensor:
    if router_logits is None or router_logits.numel() == 0:
        return torch.tensor(0.0, device='cpu')
    z = torch.logsumexp(router_logits.float(), dim=-1)
    return coeff * (z ** 2).mean()

# ── SFT Chat Dataset ─────────────────────────────────────────────────────────
class SFTChatDataset(Dataset):
    """
    Loads chat-formatted JSONL files and produces (input_ids, target_ids, loss_mask).
    
    Format expected per line:
      {"messages": [{"role": "system", "content": "..."}, 
                    {"role": "user", "content": "..."}, 
                    {"role": "assistant", "content": "..."}]}
    
    Loss masking: only assistant tokens contribute to the loss.
    System and user tokens are present in input but masked from loss.
    """
    def __init__(self, jsonl_paths: list, tokenizer, seq_len: int = 128, max_samples: int = None):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []  # list of (input_ids, loss_mask) pairs
        
        total_loaded = 0
        total_skipped = 0
        
        for path_str in jsonl_paths:
            p = Path(path_str)
            if not p.exists():
                print(f"  [SFT-DATA] Skipping (not found): {p.name}")
                continue
            
            size_mb = p.stat().st_size / 1e6
            print(f"  [SFT-DATA] Loading {p.name} ({size_mb:.1f}MB)...", end=' ', flush=True)
            
            file_count = 0
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            total_skipped += 1
                            continue
                        
                        messages = obj.get('messages', [])
                        if not messages:
                            # Try raw text format
                            text = obj.get('text', '')
                            if text and len(text) > 50:
                                tokens = self.tokenizer.encode(text)
                                if len(tokens) > 10:
                                    # For raw text, all tokens are trainable
                                    mask = [1] * len(tokens)
                                    self.samples.append((tokens, mask))
                                    file_count += 1
                            continue
                        
                        # Build chat-formatted token sequence with loss mask
                        all_tokens = []
                        all_mask = []
                        
                        for msg in messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            
                            # Format: <|role|>\ncontent\n
                            header = f"<|{role}|>\n"
                            header_tokens = self.tokenizer.encode(header)
                            content_tokens = self.tokenizer.encode(content + "\n")
                            
                            # Only assistant tokens contribute to loss
                            if role == 'assistant':
                                # Header is not loss-masked, content IS
                                all_tokens.extend(header_tokens)
                                all_mask.extend([0] * len(header_tokens))
                                all_tokens.extend(content_tokens)
                                all_mask.extend([1] * len(content_tokens))
                            else:
                                # System/user: present but masked from loss
                                all_tokens.extend(header_tokens)
                                all_mask.extend([0] * len(header_tokens))
                                all_tokens.extend(content_tokens)
                                all_mask.extend([0] * len(content_tokens))
                        
                        # Add EOS
                        eos = self.tokenizer.eos_token_id
                        all_tokens.append(eos)
                        all_mask.append(1)  # EOS is part of the assistant response
                        
                        if len(all_tokens) > 10:
                            self.samples.append((all_tokens, all_mask))
                            file_count += 1
                        
                        if max_samples and total_loaded + file_count >= max_samples:
                            break
                            
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            
            total_loaded += file_count
            print(f"{file_count:,} conversations")
            
            if max_samples and total_loaded >= max_samples:
                break
        
        print(f"  [SFT-DATA] Total: {len(self.samples):,} conversations | Skipped: {total_skipped}")
        if not self.samples:
            raise RuntimeError("No SFT data loaded. Check file paths.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens, mask = self.samples[idx]
        
        # Truncate or pad to seq_len + 1 (need +1 for input/target shift)
        total_len = self.seq_len + 1
        
        if len(tokens) > total_len:
            # Truncate from the end (keep the beginning including system prompt)
            tokens = tokens[:total_len]
            mask = mask[:total_len]
        elif len(tokens) < total_len:
            # Pad with zeros
            pad_len = total_len - len(tokens)
            tokens = tokens + [0] * pad_len
            mask = mask + [0] * pad_len  # Padding is NOT loss-contributing
        
        tokens_t = torch.tensor(tokens, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        
        # Input: tokens[:-1], Target: tokens[1:], Mask: mask[1:]
        return tokens_t[:-1], tokens_t[1:], mask_t[1:]


# ── Also load pre-tokenized .pt data for continued pre-training blend ─────────
class PretrainBlendDataset(Dataset):
    """Small fraction of raw pre-training data to prevent catastrophic forgetting."""
    def __init__(self, file_paths: list, seq_len: int = 128, max_tokens: int = 5_000_000):
        self.seq_len = seq_len
        self.stride = seq_len // 2
        self.file_tensors = []
        self.chunk_map = []
        
        total_tokens = 0
        for path_str in file_paths:
            p = Path(path_str)
            if not p.exists(): continue
            try:
                raw = torch.load(str(p), weights_only=True, map_location='cpu')
                if isinstance(raw, dict):
                    raw = raw.get('input_ids', raw.get('tokens', next(iter(raw.values()))))
                t = raw.reshape(-1)
                if len(t) > max_tokens:
                    t = t[:max_tokens]
                t = t.to(torch.int32)
                n_chunks = max(0, (len(t) - seq_len - 1) // self.stride)
                if n_chunks == 0: continue
                fid = len(self.file_tensors)
                self.file_tensors.append(t)
                self.chunk_map.extend([(fid, i * self.stride) for i in range(n_chunks)])
                total_tokens += len(t)
                print(f"  [BLEND] {p.name}: {len(t):,} tokens → {n_chunks:,} chunks")
            except Exception as e:
                print(f"  [BLEND] Error loading {p.name}: {e}")
        
        print(f"  [BLEND] Total: {total_tokens:,} tokens | {len(self.chunk_map):,} chunks")
    
    def __len__(self):
        return len(self.chunk_map)
    
    def __getitem__(self, idx):
        fid, start = self.chunk_map[idx]
        t = self.file_tensors[fid]
        end = start + self.seq_len + 1
        if end > len(t):
            end = len(t)
            start = max(0, end - self.seq_len - 1)
        chunk = t[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
        inp = chunk[:-1].long().clone()
        tgt = chunk[1:].long().clone()
        # Full mask — all tokens contribute to loss in pre-training blend
        mask = torch.ones(self.seq_len, dtype=torch.float32)
        return inp, tgt, mask


# ── Checkpoint I/O (same as pre-training) ─────────────────────────────────────
def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    model_sd = model.state_dict()
    ckpt = {
        'model_state_dict': model_sd,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'loss': loss,
        'version': 'v5.3.1-sft-2026',
        'phase': 'sft',
        'ts': time.time(),
    }
    tmp = str(path) + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, str(path))
    size_gb = os.path.getsize(str(path)) / 1e9
    print(f"  [CKPT] Saved → {Path(path).name} ({size_gb:.2f}GB)")

def load_checkpoint(model, path):
    p = Path(path)
    if not p.exists():
        print(f"  [CKPT] Not found: {path}")
        return
    print(f"  [CKPT] Loading: {p.name} ({p.stat().st_size/1e9:.2f}GB)")
    ckpt = torch.load(str(p), map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    
    model_sd = model.state_dict()
    loaded, skipped = 0, 0
    for k in list(sd.keys()):
        v = sd.pop(k)
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    del sd, ckpt
    gc.collect()
    print(f"  [CKPT] Loaded {loaded} keys, skipped {skipped}")

# ── JSONL Logger ─────────────────────────────────────────────────────────────
class TrainingLogger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(str(path), 'a', encoding='utf-8')
    def log(self, d: dict):
        self._fh.write(json.dumps(d) + '\n')
        self._fh.flush()
    def close(self):
        self._fh.close()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',   type=int,  default=1000)
    parser.add_argument('--seq',     type=int,  default=128)
    parser.add_argument('--accum',   type=int,  default=4)
    parser.add_argument('--resume',  action='store_true')
    parser.add_argument('--ckpt',    type=str,  default=None)
    parser.add_argument('--blend-ratio', type=float, default=0.1,
                        help='Fraction of pre-train data to blend (prevents forgetting)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 5 steps then exit (smoke test)')
    args = parser.parse_args()

    STEPS       = 5 if args.dry_run else args.steps
    SEQ_LEN     = args.seq
    ACCUM_STEPS = args.accum

    # ── Device ────────────────────────────────────────────────────────────────
    device = detect_device()

    print(f"\n{'='*62}")
    print(f"  QUILLAN-RONIN v5.3.1  |  SFT PHASE  |  2026")
    print(f"  Device:{device:>6}  |  Steps:{STEPS:>6}  |  Seq:{SEQ_LEN}")
    print(f"{'='*62}\n")

    if device == 'cpu':
        n_threads = 2
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(1)
        os.environ.setdefault('OMP_NUM_THREADS', str(n_threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(n_threads))
        print(f"  CPU threads capped at: {n_threads}")

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    print("  Building model...")
    model = QuillanRoninSovereign(cfg)
    model.txt_dec.weight = model.ingestion.txt_emb.weight

    if hasattr(model, 'agentic_executor'):
        model.agentic_executor._training_guard = True

    # ── Load pre-trained checkpoint ───────────────────────────────────────────
    ckpt_dir = Path('checkpoints_v2')
    sft_ckpt_dir = Path('checkpoints_sft')
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt:
        load_checkpoint(model, args.ckpt)
    elif args.resume and (sft_ckpt_dir / 'quillan_sft_latest.pt').exists():
        load_checkpoint(model, sft_ckpt_dir / 'quillan_sft_latest.pt')
    else:
        # Load the pre-trained checkpoint
        candidates = [
            ckpt_dir / 'quillan_sovereign_step2000.pt',
            ckpt_dir / 'quillan_sovereign_latest.pt',
            ckpt_dir / 'quillan_finetuned.pt',
        ]
        load_path = next((str(c) for c in candidates if c.exists()), None)
        if load_path:
            load_checkpoint(model, load_path)
        else:
            print("  [WARN] No pre-trained checkpoint found! Starting from random init.")

    # ── Parameter freeze strategy for SFT ─────────────────────────────────────
    # SFT: Train the same components as pre-training, but with lower LR.
    # The idea is that all language-producing components need to adapt to
    # the instruction-following format.
    TRAINABLE = [
        'moe.router',
        'moe.w1_lora', 'moe.wgate_lora', 'moe.w2_lora',
        'moe.expert_swarms',
        'moe.output_norm',
        'quillan_finalizer',
        'quillan_finalizer2',
        'quillan_gate',
        'pre_final_norm',
        'txt_dec.',
        'ingestion.',
        'decomposition.',
        'diffusion_core.',
    ]

    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in TRAINABLE)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_train/1e6:.1f}M trainable / {n_total/1e6:.1f}M total")

    model.to(device)
    model.train()

    # ── Optimizer (lower LR for SFT) ──────────────────────────────────────────
    lora_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(k in name for k in ('lora_', 'expert_swarms', 'decomposition',
                                    'thought_paths', 'path_projector')):
            lora_params.append(p)
        else:
            base_params.append(p)

    # SFT uses lower LR to avoid catastrophic forgetting
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-5,  'weight_decay': 0.01},
        {'params': lora_params, 'lr': 5e-5,  'weight_decay': 0.0},
    ], betas=(0.9, 0.95), eps=1e-8)

    # Cosine decay with 2% warmup
    warmup = max(20, int(STEPS * 0.02))
    def lr_lambda(s):
        if s < warmup:
            return s / warmup
        prog = (s - warmup) / max(STEPS - warmup, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\n  Loading tokenizer...")
    tokenizer = QuillanBPETokenizer()
    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    if not tok_path.exists():
        tok_path = ROOT / 'training_data' / 'tokenizer.json'
    if not tok_path.exists():
        tok_path = ROOT / 'scripts' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer.load(str(tok_path))
    print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # ── SFT Dataset ───────────────────────────────────────────────────────────
    print("\n  Loading SFT data...")
    data_dir = ROOT / 'training_data'
    
    sft_files = [
        str(data_dir / 'instruct_train.jsonl'),         # 60MB — primary SFT data
        str(data_dir / 'GPT_5.5_Distilled.jsonl'),      # 47MB — distilled conversations
        str(data_dir / 'code_train.jsonl'),              # 12MB — code instructions
        str(data_dir / 'quillan_12mb_training_dataset.jsonl'),  # 13MB — curated
        str(data_dir / 'quillan_science_absolute.jsonl'),       # 2.7MB — science
        str(data_dir / 'quillan_science_additional.jsonl'),     # 4.4MB — more science
    ]
    
    sft_dataset = SFTChatDataset(sft_files, tokenizer, seq_len=SEQ_LEN, max_samples=50000)
    
    # Optional: blend in some raw pre-training data to prevent forgetting
    blend_dataset = None
    if args.blend_ratio > 0:
        print("\n  Loading pre-train blend data...")
        blend_files = [
            str(data_dir / 'quillan_corpus_CLEAN_V7.pt'),
        ]
        try:
            blend_dataset = PretrainBlendDataset(blend_files, seq_len=SEQ_LEN, max_tokens=2_000_000)
        except Exception as e:
            print(f"  [BLEND] Failed to load blend data: {e}")
            blend_dataset = None

    sft_loader = DataLoader(sft_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    blend_loader = None
    if blend_dataset and len(blend_dataset) > 0:
        blend_loader = DataLoader(blend_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir = ROOT / 'training_logs'
    log_dir.mkdir(exist_ok=True)
    logger = TrainingLogger(log_dir / 'sft_v2026.jsonl')

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"\n  Starting SFT training. Target: {STEPS} steps.\n")

    step = 0
    micro_step = 0
    ema_loss = None
    ema_sft_loss = None
    t0 = time.time()
    sft_iter = iter(sft_loader)
    blend_iter = iter(blend_loader) if blend_loader else None
    last_save_time = time.time()
    save_interval_seconds = 1800  # Auto-save every 30 min during SFT

    optimizer.zero_grad()

    # Emergency save handlers
    import signal, atexit
    def _emergency_save(reason="unknown"):
        try:
            save_checkpoint(model, optimizer, scheduler, step, ema_loss,
                            sft_ckpt_dir / 'quillan_sft_latest.pt')
            print(f"\n  [EMERGENCY] Saved at step {step} (reason: {reason})")
        except Exception as ex:
            print(f"\n  [EMERGENCY] Save failed: {ex}")
    
    signal.signal(signal.SIGTERM, lambda s, f: (_emergency_save(f"signal {s}"), sys.exit(0)))
    signal.signal(signal.SIGINT, lambda s, f: (_emergency_save(f"signal {s}"), sys.exit(0)))
    atexit.register(lambda: _emergency_save("atexit"))

    while step < STEPS:
        # ── Decide: SFT batch or blend batch ──────────────────────────────────
        use_blend = (blend_iter is not None and 
                     torch.rand(1).item() < args.blend_ratio)
        
        if use_blend:
            try:
                inp, tgt, loss_mask = next(blend_iter)
            except StopIteration:
                blend_iter = iter(blend_loader)
                inp, tgt, loss_mask = next(blend_iter)
        else:
            try:
                inp, tgt, loss_mask = next(sft_iter)
            except StopIteration:
                sft_iter = iter(sft_loader)
                inp, tgt, loss_mask = next(sft_iter)

        inp = inp.to(device)
        tgt = tgt.to(device)
        loss_mask = loss_mask.to(device)

        # ── Forward ───────────────────────────────────────────────────────────
        try:
            out = model(inp)
        except Exception as e:
            print(f"  [FWD ERR] step {step}: {e}")
            optimizer.zero_grad()
            gc.collect()
            continue

        logits = out.get('logits', out) if isinstance(out, dict) else out
        if isinstance(logits, dict):
            logits = logits.get('logits')

        # Per-token cross-entropy (no reduction)
        loss_per_token = F.cross_entropy(
            logits.float().reshape(-1, cfg.vocab_size),
            tgt.reshape(-1),
            reduction='none',
        )
        
        # Apply loss mask: only assistant tokens contribute
        mask_flat = loss_mask.reshape(-1)
        masked_loss = (loss_per_token * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        if torch.isnan(masked_loss) or torch.isinf(masked_loss):
            optimizer.zero_grad()
            continue

        # Auxiliary losses
        routing_loss = torch.tensor(0.0, device=device)
        if isinstance(out, dict):
            routing_loss = out.get('routing_loss', routing_loss)

        z_loss = torch.tensor(0.0, device=device)
        if hasattr(model, 'moe') and hasattr(model.moe, '_last_logits'):
            ll = model.moe._last_logits
            if ll is not None:
                z_loss = router_z_loss(ll, coeff=1e-3)

        loss = masked_loss + 0.01 * routing_loss + z_loss

        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        # ── Backward ──────────────────────────────────────────────────────────
        (loss / ACCUM_STEPS).backward()
        micro_step += 1
        
        ce_val = masked_loss.item()
        ema_loss = ce_val if ema_loss is None else 0.95 * ema_loss + 0.05 * ce_val
        if not use_blend:
            ema_sft_loss = ce_val if ema_sft_loss is None else 0.95 * ema_sft_loss + 0.05 * ce_val

        if micro_step % ACCUM_STEPS != 0:
            continue

        # Grad clip
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )

        if torch.isnan(grad_norm):
            optimizer.zero_grad()
            continue

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        step += 1

        # ── Console output ────────────────────────────────────────────────────
        elapsed = time.time() - t0
        sps = step / max(elapsed, 1e-6)
        cur_lr = optimizer.param_groups[0]['lr']
        eta_h = (STEPS - step) / max(sps * 3600, 1e-6)
        batch_type = "BLEND" if use_blend else "SFT"

        if step % 5 == 0 or step <= 10:
            sft_str = f"  sft_ce={ema_sft_loss:.4f}" if ema_sft_loss else ""
            print(
                f"  step {step:>5}/{STEPS}"
                f"  loss={ema_loss:.4f}"
                f"  ce={ce_val:.3f}"
                f"  z={z_loss.item():.5f}"
                f"  gn={grad_norm.item():.3f}"
                f"  lr={cur_lr:.1e}"
                f"  {sps:.3f}st/s"
                f"  ETA:{eta_h:.1f}h"
                f"  [{batch_type}]"
                f"{sft_str}"
            )
            sys.stdout.flush()
            import psutil
            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f"  [RAM] {mem:.1f} MB")
            sys.stdout.flush()

        logger.log({
            'step': step,
            'ema_loss': ema_loss,
            'sft_loss': ema_sft_loss,
            'loss_ce': ce_val,
            'routing_loss': routing_loss.item(),
            'z_loss': z_loss.item(),
            'grad_norm': grad_norm.item(),
            'lr': cur_lr,
            'sps': sps,
            'batch_type': batch_type,
            'ts': time.time(),
        })

        # ── Periodic checkpoints ──────────────────────────────────────────────
        if (step <= 50 and step % 10 == 0) or (step % 100 == 0):
            save_checkpoint(model, optimizer, scheduler, step, ema_loss,
                            sft_ckpt_dir / 'quillan_sft_latest.pt')
            print(f"  [CKPT] Saved at step {step}")

        # Time-based auto-save
        if time.time() - last_save_time >= save_interval_seconds:
            save_checkpoint(model, optimizer, scheduler, step, ema_loss,
                            sft_ckpt_dir / 'quillan_sft_latest.pt')
            print(f"  [SENTINEL] Auto-saved at {time.strftime('%H:%M:%S')}")
            last_save_time = time.time()

        if step % 5 == 0:
            gc.collect()

        if args.dry_run and step >= 5:
            print("\n  [DRY-RUN] 5 steps OK — SFT script is healthy!")
            break

    # ── Final save ────────────────────────────────────────────────────────────
    logger.close()
    save_checkpoint(model, optimizer, scheduler, step, ema_loss,
                    sft_ckpt_dir / 'quillan_sft_final.pt')
    
    # Also save a standalone state_dict for inference
    torch.save(model.state_dict(), sft_ckpt_dir / 'quillan_sft_weights.pt')

    elapsed_h = (time.time() - t0) / 3600
    print(f"\n{'='*62}")
    print(f"  SFT TRAINING COMPLETE")
    ema_str = f"{ema_loss:.4f}" if ema_loss is not None else "N/A"
    sft_str = f"{ema_sft_loss:.4f}" if ema_sft_loss is not None else "N/A"
    print(f"  Steps: {step} | Final EMA loss: {ema_str} | SFT loss: {sft_str}")
    print(f"  Total time: {elapsed_h:.2f}h | "
          f"Avg: {step/max(elapsed_h,1e-6):.0f} steps/hr")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] SFT training stopped by user.")
    except Exception as e:
        print(f"\n  [FATAL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
