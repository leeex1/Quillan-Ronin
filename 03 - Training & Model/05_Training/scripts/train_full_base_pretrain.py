#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — FULL BASE BITNET PRE-TRAINING (STE Direct Backprop)
==========================================================================
Trains ALL 453.9M BitNet weights (37.4M active) directly via STE backprop.
Unfreezes main BitLinear matrices so self-attention heads and expert MLPs
learn full ternary representations across 223M+ tokens!

Usage:
  python scripts/train_full_base_pretrain.py [--steps 5000] [--seq-len 256]
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

class MultiDatasetPretrainStreamer(Dataset):
    def __init__(self, file_paths: list, seq_len: int = 256):
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
                t = raw.reshape(-1).to(torch.int32)
                n_chunks = max(0, (len(t) - seq_len - 1) // self.stride)
                if n_chunks == 0: continue
                fid = len(self.file_tensors)
                self.file_tensors.append(t)
                self.chunk_map.extend([(fid, i * self.stride) for i in range(n_chunks)])
                total_tokens += len(t)
                print(f"  [CORPUS] Loading {p.name} ({p.stat().st_size/1e6:.1f} MB)... {len(t):,} tokens ({n_chunks:,} chunks)")
            except Exception as e:
                print(f"  [CORPUS] Error loading {p.name}: {e}")

        print(f"\n  [CORPUS-TOTAL] {total_tokens:,} tokens across {len(self.chunk_map):,} sequence chunks!\n")

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        fid, start = self.chunk_map[idx]
        t = self.file_tensors[fid]
        chunk = t[start : start + self.seq_len + 1].long()
        return chunk[:-1], chunk[1:]

def main():
    parser = argparse.ArgumentParser(description="Full Base BitNet Model Pre-Training")
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=300)
    args = parser.parse_args()

    device = detect_device()
    print("\n" + "=" * 68)
    print(f"  QUILLAN-RONIN v5.3.1 | FULL BASE BITNET PRE-TRAINING (ALL WEIGHTS UNFROZEN)")
    print(f"  Device: {device} | Target Steps: {args.steps} | SeqLen: {args.seq_len}")
    print("=" * 68 + "\n")

    if not torch.cuda.is_available():
        # Set max CPU threads for accelerated BitNet backprop
        max_t = os.cpu_count() or 4
        torch.set_num_threads(max_t)
        print(f"  CPU threads set to: {max_t}")

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg)
    model.txt_dec.weight = model.ingestion.txt_emb.weight

    # UNFREEZE ALL MODEL PARAMETERS FOR FULL BASE PRE-TRAINING
    for name, param in model.named_parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: ALL {trainable_params/1e6:.1f}M parameters trainable / {total_params/1e6:.1f}M total")

    ckpt_dir = ROOT / 'checkpoints' / 'checkpoints_v2'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / 'quillan_full_base_latest.pt'

    start_step = 1
    # Load existing base weights if available
    base_ckpt = latest_ckpt if latest_ckpt.exists() else (ckpt_dir / 'quillan_sovereign_step2000.pt')
    if base_ckpt.exists():
        print(f"  [CKPT] Loading baseline checkpoint: {base_ckpt.name}")
        b_data = torch.load(str(base_ckpt), map_location='cpu', weights_only=False)
        b_sd = b_data.get('model_state_dict', b_data)
        start_step = b_data.get('step', 1)
        m_sd = model.state_dict()
        for k, v in list(b_sd.items()):
            if k in m_sd and v.shape == m_sd[k].shape:
                m_sd[k].copy_(v)
        del b_sd, b_data
        print(f"  [CKPT] Resuming from step {start_step}")

    data_dir = ROOT / 'training_data'
    corpus_files = sorted(list(data_dir.glob('*.pt')))
    ds = MultiDatasetPretrainStreamer([str(f) for f in corpus_files], seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    loader_iter = iter(loader)
    ema_loss = 7.5
    start_time = time.time()
    last_save_time = time.time()

    model.train()
    print(f"  Starting Full Base BitNet Pre-Training across {len(ds):,} chunks. Target: {args.steps} steps.\n")

    for step in range(start_step + 1, args.steps + 1):
        try:
            inp, tgt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inp, tgt = next(loader_iter)

        inp = inp.to(device)
        tgt = tgt.to(device)

        # Learning rate schedule: Warmup + Cosine Decay
        if step <= args.warmup_steps:
            lr_scale = step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        current_lr = args.lr * lr_scale
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        out = model(inp)
        logits = out['logits']

        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        ema_loss = 0.95 * ema_loss + 0.05 * loss_val

        if step % 5 == 0 or step == args.steps:
            elapsed = time.time() - start_time
            s_per_st = elapsed / step
            eta_h = (args.steps - step) * s_per_st / 3600
            print(f"  step {step:5d}/{args.steps}  loss={ema_loss:.4f}  ce={loss_val:.3f}  lr={current_lr:.1e}  {s_per_st:.3f}s/st  ETA:{eta_h:.1f}h", flush=True)

        now = time.time()
        if step % 100 == 0 or (now - last_save_time) > 1200 or step == args.steps or step == 1500:
            # Save FULL model state dict including base weights
            save_dict = {'model_state_dict': model.state_dict(), 'step': step}
            torch.save(save_dict, str(latest_ckpt))
            last_save_time = now
            print(f"  [CKPT] Saved FULL base model → {latest_ckpt.name} at step {step}", flush=True)

        if step >= 1500:
            print(f"\n  [TARGET REACHED] Reached target step 1,500 as requested!")
            break

    final_path = ckpt_dir / 'quillan_full_base_final.pt'
    torch.save({'model_state_dict': model.state_dict(), 'step': args.steps}, str(final_path))
    print(f"\n==================================================")
    print(f"  FULL BASE BITNET PRE-TRAINING COMPLETE")
    print(f"  Saved final full base checkpoint: {final_path.name}")
    print(f"==================================================")

if __name__ == '__main__':
    main()
