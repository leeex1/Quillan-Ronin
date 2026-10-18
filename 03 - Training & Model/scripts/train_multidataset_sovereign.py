#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — MULTI-DATASET FULL-CORPUS PRE-TRAINING
============================================================
Streams through ALL pre-tokenized .pt datasets in training_data/:
  - quillan_corpus_CLEAN_V7.pt (1.33 GB, ~330M tokens)
  - full_train.pt (128 MB, ~32M tokens)
  - instruct_train.pt (108 MB, ~27M tokens)
  - GPT_5.5_Distilled.pt (85 MB, ~21M tokens)
  - quillan_tokenized.pt (55 MB, ~14M tokens)
  - train.pt & code_train.pt (50 MB, ~12M tokens)
  - quillan_science_absolute.pt & quillan_science_additional.pt (13 MB, ~3.5M tokens)

TOTAL: ~440 Million Tokens across all local datasets!

Usage:
  python scripts/train_multidataset_sovereign.py [--steps 20000] [--seq-len 256] [--resume]
"""
import os, sys, gc, time, json, math, argparse, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, str(ROOT / '_dev'))
sys.path.insert(0, str(ROOT))

from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from quillan_bpe_tokenizer import QuillanBPETokenizer

def detect_device():
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 7: return 'cuda'
    return 'cpu'

# LanceDB training guard
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

class MultiDatasetStreamer(Dataset):
    """
    Streams across ALL .pt pre-tokenized files in training_data/.
    Loads tensors dynamically to fit comfortably in RAM.
    """
    def __init__(self, data_dir: Path, seq_len: int = 256):
        self.seq_len = seq_len
        self.stride = seq_len // 2
        self.file_tensors = []
        self.chunk_map = []

        pt_files = sorted([
            'quillan_corpus_CLEAN_V7.pt',
            'full_train.pt',
            'instruct_train.pt',
            'GPT_5.5_Distilled.pt',
            'quillan_tokenized.pt',
            'train.pt',
            'code_train.pt',
            'quillan_science_absolute.pt',
            'quillan_science_additional.pt',
            'quillan_12mb_training_dataset.pt',
            'full_dataset.pt',
        ])

        total_tokens = 0
        for name in pt_files:
            p = data_dir / name
            if not p.exists(): continue
            try:
                size_mb = p.stat().st_size / 1e6
                print(f"  [CORPUS] Loading {name} ({size_mb:.1f} MB)...", end=' ', flush=True)
                raw = torch.load(str(p), weights_only=True, map_location='cpu')
                if isinstance(raw, dict):
                    raw = raw.get('input_ids', raw.get('tokens', next(iter(raw.values()))))
                t = raw.reshape(-1).to(torch.int32)
                n_chunks = max(0, (len(t) - seq_len - 1) // self.stride)
                if n_chunks == 0:
                    print("skipped (too small)")
                    continue
                fid = len(self.file_tensors)
                self.file_tensors.append(t)
                self.chunk_map.extend([(fid, i * self.stride) for i in range(n_chunks)])
                total_tokens += len(t)
                print(f"{len(t):,} tokens ({n_chunks:,} chunks)")
            except Exception as e:
                print(f"ERROR: {e}")

        print(f"\n  [CORPUS-TOTAL] {total_tokens:,} tokens (~{total_tokens/1e6:.1f}M tokens) across {len(self.chunk_map):,} sequence chunks!")

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        fid, start = self.chunk_map[idx]
        t = self.file_tensors[fid]
        chunk = t[start : start + self.seq_len + 1].long()
        return chunk[:-1], chunk[1:]

def main():
    parser = argparse.ArgumentParser(description="Quillan Multi-Dataset Pre-Training Streamer")
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lora-lr', type=float, default=2e-4)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = detect_device()
    print("\n" + "=" * 66)
    print(f"  QUILLAN-RONIN v5.3.1 | MULTI-DATASET FULL-CORPUS PRE-TRAINING")
    print(f"  Device: {device} | Target Steps: {args.steps} | SeqLen: {args.seq_len}")
    print("=" * 66 + "\n")

    if not torch.cuda.is_available():
        torch.set_num_threads(2)
        print("  CPU threads capped at: 2")

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg)
    model.txt_dec.weight = model.ingestion.txt_emb.weight

    ckpt_dir = ROOT / 'checkpoints' / 'checkpoints_v2'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / 'quillan_sovereign_latest.pt'
    step2000_ckpt = ckpt_dir / 'quillan_sovereign_step2000.pt'

    start_step = 0
    if args.resume and latest_ckpt.exists():
        load_path = latest_ckpt
        print(f"  [CKPT] Resuming pre-training from: {load_path.name}")
    elif latest_ckpt.exists():
        load_path = latest_ckpt
        print(f"  [CKPT] Continuing pre-training from latest: {load_path.name}")
    else:
        load_path = step2000_ckpt
        print(f"  [CKPT] Continuing pre-training from: {load_path.name}")

    ckpt = torch.load(str(load_path), map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    model_sd = model.state_dict()
    loaded = 0
    for k, v in list(sd.items()):
        if k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k].copy_(v)
            loaded += 1
    del sd, ckpt
    print(f"  [CKPT] Loaded {loaded} keys into Sovereign model")

    # Freeze base MoE weights
    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable = []
    base_params = []
    lora_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['lora_', 'router', 'w1_lora', 'w2_lora', 'finalizer', 'q1_brain', 'q2_brain']):
            param.requires_grad = True
            trainable.append(param)
            if 'lora' in name: lora_params.append(param)
            else: base_params.append(param)

    print(f"  Params: {sum(p.numel() for p in trainable)/1e6:.1f}M trainable / {sum(p.numel() for p in model.parameters())/1e6:.1f}M total")

    tokenizer = QuillanBPETokenizer()
    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer.load(str(tok_path))

    data_dir = ROOT / 'training_data'
    ds = MultiDatasetStreamer(data_dir, seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr, 'weight_decay': 0.01},
        {'params': lora_params, 'lr': args.lora_lr, 'weight_decay': 0.001},
    ])

    data_iter = iter(loader)
    total_steps = 5 if args.dry_run else args.steps
    ema_loss = 2.5
    start_time = time.time()
    last_save_time = time.time()

    model.train()
    print(f"\n  Starting Full Multi-Dataset Pre-Training. Target: {total_steps} steps.\n")

    for step in range(1, total_steps + 1):
        try:
            inp, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, tgt = next(data_iter)

        inp = inp.to(device)
        tgt = tgt.to(device)

        # LR Warmup + Cosine Decay
        if step <= args.warmup_steps:
            lr_scale = step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            initial_lr = args.lora_lr if 'weight_decay' in param_group and param_group['weight_decay'] == 0.001 else args.lr
            param_group['lr'] = initial_lr * lr_scale

        optimizer.zero_grad()
        out = model(inp)
        logits = out['logits']

        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        ema_loss = 0.95 * ema_loss + 0.05 * loss_val

        if step % 50 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            s_per_st = elapsed / step
            eta_h = (total_steps - step) * s_per_st / 3600
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  step {step:5d}/{total_steps}  loss={ema_loss:.4f}  ce={loss_val:.3f}  lr={current_lr:.1e}  {s_per_st:.3f}s/st  ETA:{eta_h:.1f}h")

        now = time.time()
        if step % 250 == 0 or (now - last_save_time) > 1800 or step == total_steps:
            save_dict = {'model_state_dict': {k: v for k, v in model.state_dict().items() if v.requires_grad or 'lora' in k or 'router' in k or 'finalizer' in k}}
            torch.save(save_dict, str(latest_ckpt))
            last_save_time = now
            print(f"  [CKPT] Saved → {latest_ckpt.name} at step {step}")

    final_path = ckpt_dir / 'quillan_sovereign_final.pt'
    save_dict = {'model_state_dict': {k: v for k, v in model.state_dict().items() if v.requires_grad or 'lora' in k or 'router' in k or 'finalizer' in k}}
    torch.save(save_dict, str(final_path))
    print(f"\n==================================================")
    print(f"  MULTI-DATASET FULL-CORPUS PRE-TRAINING COMPLETE")
    print(f"  Saved final checkpoint: {final_path.name}")
    print(f"==================================================")

if __name__ == '__main__':
    main()
