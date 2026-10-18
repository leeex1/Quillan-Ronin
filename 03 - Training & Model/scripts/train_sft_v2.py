#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.1 — SFT REFINEMENT SCRIPT (Phase 2.5)
======================================================
Solves the truncation & alignment bottleneck of Phase 1 SFT:
  1. Sequence Length upgraded from 128 -> 256 tokens (enables multi-paragraph response learning)
  2. Strict Assistant Token Filtering (discards any chunk with < 15 assistant tokens, 0% ghost batches)
  3. Smart Assistant Windowing (slides window to capture assistant response if prompt is long)
  4. Anti-forgetting blend (10% raw pre-training tokens)
  5. Base LR: 2e-5 / LoRA LR: 8e-5 with 5% warmup & Cosine Decay

Usage:
  python scripts/train_sft_v2.py --steps 1000 [--resume] [--dry-run]
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
        if cap[0] >= 7:
            return 'cuda'
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

class SFTRefinementDataset(Dataset):
    """
    Loads chat JSONL conversations and formats them with loss-masked assistant responses.
    Ensures every output chunk has at least min_assistant_tokens (default: 15).
    """
    def __init__(self, jsonl_paths: list, tokenizer, seq_len: int = 256, min_assistant_tokens: int = 15):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []
        eos = tokenizer.eos_token_id or 0

        total_loaded = 0
        total_skipped = 0

        for path_str in jsonl_paths:
            p = Path(path_str)
            if not p.exists():
                continue

            size_mb = p.stat().st_size / 1e6
            print(f"  [SFT-V2] Loading {p.name} ({size_mb:.1f}MB)...", end=' ', flush=True)

            file_count = 0
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                    except:
                        total_skipped += 1
                        continue

                    messages = obj.get('messages', [])
                    if not messages:
                        text = obj.get('text', '')
                        if text and len(text) > 50:
                            tokens = tokenizer.encode(text)
                            if len(tokens) > 20:
                                mask = [1] * len(tokens)
                                self._add_valid_chunks(tokens, mask, min_assistant_tokens)
                                file_count += 1
                        continue

                    all_tokens = []
                    all_mask = []

                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')

                        header = f"<|{role}|>\n"
                        h_ids = tokenizer.encode(header)
                        c_ids = tokenizer.encode(content + "\n")

                        if role == 'assistant':
                            all_tokens.extend(h_ids)
                            all_mask.extend([0] * len(h_ids))
                            all_tokens.extend(c_ids)
                            all_mask.extend([1] * len(c_ids))
                        else:
                            all_tokens.extend(h_ids)
                            all_mask.extend([0] * len(h_ids))
                            all_tokens.extend(c_ids)
                            all_mask.extend([0] * len(c_ids))

                    all_tokens.append(eos)
                    all_mask.append(1)

                    added = self._add_valid_chunks(all_tokens, all_mask, min_assistant_tokens)
                    if added:
                        file_count += 1

            total_loaded += file_count
            print(f"{file_count:,} conversations")

        print(f"  [SFT-V2] Total valid chunks: {len(self.samples):,}")
        if not self.samples:
            raise RuntimeError("No SFT data loaded!")

    def _add_valid_chunks(self, tokens, mask, min_assistant_tokens):
        target_len = self.seq_len + 1
        added_any = False

        if len(tokens) <= target_len:
            # Single chunk
            ast_count = sum(mask[1:])
            if ast_count >= min_assistant_tokens:
                self.samples.append((tokens, mask))
                added_any = True
        else:
            # If long conversation, find assistant response start and slice around it
            # Primary chunk (start of conversation)
            c1_tokens = tokens[:target_len]
            c1_mask = mask[:target_len]
            if sum(c1_mask[1:]) >= min_assistant_tokens:
                self.samples.append((c1_tokens, c1_mask))
                added_any = True

            # Secondary chunk (centered on assistant response if available)
            first_ast_idx = next((i for i, m in enumerate(mask) if m == 1), -1)
            if first_ast_idx > 40:
                start_idx = max(0, first_ast_idx - 30)
                end_idx = start_idx + target_len
                c2_tokens = tokens[start_idx:end_idx]
                c2_mask = mask[start_idx:end_idx]
                if sum(c2_mask[1:]) >= min_assistant_tokens and len(c2_tokens) >= target_len:
                    self.samples.append((c2_tokens, c2_mask))
                    added_any = True

        return added_any

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, mask = self.samples[idx]
        target_len = self.seq_len + 1

        if len(tokens) > target_len:
            tokens = tokens[:target_len]
            mask = mask[:target_len]
        elif len(tokens) < target_len:
            pad_len = target_len - len(tokens)
            tokens = tokens + [0] * pad_len
            mask = mask + [0] * pad_len

        tokens_t = torch.tensor(tokens, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        return tokens_t[:-1], tokens_t[1:], mask_t[1:]

class PretrainBlendDataset(Dataset):
    def __init__(self, file_paths: list, seq_len: int = 256, max_tokens: int = 5_000_000):
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
                if len(t) > max_tokens: t = t[:max_tokens]
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
        chunk = t[start : start + self.seq_len + 1].long()
        mask = torch.ones(self.seq_len, dtype=torch.float32)
        return chunk[:-1], chunk[1:], mask

def main():
    parser = argparse.ArgumentParser(description="Quillan Phase 2.5 Long-Context SFT Refinement")
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--base-lr', type=float, default=2e-5)
    parser.add_argument('--lora-lr', type=float, default=8e-5)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    device = detect_device()
    print("\n" + "=" * 62)
    print(f"  QUILLAN-RONIN v5.3.1 | PHASE 2.5 SFT REFINEMENT")
    print(f"  Device: {device} | Steps: {args.steps} | SeqLen: {args.seq_len}")
    print("=" * 62 + "\n")

    if not torch.cuda.is_available():
        torch.set_num_threads(2)
        print("  CPU threads capped at: 2")

    cfg = QuillanArchConfig(device=device, text_only=True, eggroll_rank=16)
    model = QuillanRoninSovereign(cfg)
    model.txt_dec.weight = model.ingestion.txt_emb.weight

    ckpt_dir = ROOT / 'checkpoints' / 'checkpoints_sft'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_sft = ckpt_dir / 'quillan_sft_latest.pt'
    pretrain_ckpt = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_sovereign_latest.pt'
    if not pretrain_ckpt.exists():
        pretrain_ckpt = ROOT / 'checkpoints' / 'checkpoints_v2' / 'quillan_sovereign_step2000.pt'

    start_step = 0
    if args.resume and latest_sft.exists():
        load_path = latest_sft
        print(f"  [CKPT] Resuming SFT from: {load_path.name}")
    else:
        load_path = pretrain_ckpt
        print(f"  [CKPT] Initializing from pre-train checkpoint: {load_path.name}")

    ckpt = torch.load(str(load_path), map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    model_sd = model.state_dict()
    loaded = 0
    for k in list(sd.keys()):
        v = sd.pop(k)
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
            if 'lora' in name:
                lora_params.append(param)
            else:
                base_params.append(param)

    print(f"  Params: {sum(p.numel() for p in trainable)/1e6:.1f}M trainable / {sum(p.numel() for p in model.parameters())/1e6:.1f}M total")

    tokenizer = QuillanBPETokenizer()
    tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    tokenizer.load(str(tok_path))

    data_dir = ROOT / 'training_data'
    sft_files = [
        data_dir / 'instruct_train.jsonl',
        data_dir / 'GPT_5.5_Distilled.jsonl',
        data_dir / 'code_train.jsonl',
        data_dir / 'quillan_science_absolute.jsonl',
        data_dir / 'quillan_science_additional.jsonl',
    ]

    sft_ds = SFTRefinementDataset([str(f) for f in sft_files], tokenizer, seq_len=args.seq_len, min_assistant_tokens=15)
    sft_loader = DataLoader(sft_ds, batch_size=1, shuffle=True)

    blend_files = [data_dir / 'quillan_corpus_CLEAN_V7.pt']
    blend_ds = PretrainBlendDataset([str(f) for f in blend_files], seq_len=args.seq_len)
    blend_loader = DataLoader(blend_ds, batch_size=1, shuffle=True) if len(blend_ds) > 0 else None

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.base_lr, 'weight_decay': 0.01},
        {'params': lora_params, 'lr': args.lora_lr, 'weight_decay': 0.001},
    ])

    sft_iter = iter(sft_loader)
    blend_iter = iter(blend_loader) if blend_loader else None

    total_steps = 5 if args.dry_run else args.steps
    ema_loss = 0.5
    start_time = time.time()
    last_save_time = time.time()

    model.train()
    print(f"\n  Starting Phase 2.5 SFT Refinement. Target: {total_steps} steps.\n")

    for step in range(1, total_steps + 1):
        # 90% SFT, 10% blend
        use_blend = (blend_iter is not None) and (step % 10 == 0)

        if use_blend:
            try:
                inp, tgt, mask = next(blend_iter)
            except StopIteration:
                blend_iter = iter(blend_loader)
                inp, tgt, mask = next(blend_iter)
            b_type = "BLEND"
        else:
            try:
                inp, tgt, mask = next(sft_iter)
            except StopIteration:
                sft_iter = iter(sft_loader)
                inp, tgt, mask = next(sft_iter)
            b_type = "SFT"

        inp = inp.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)

        # LR schedule: Warmup + Cosine
        if step <= args.warmup_steps:
            lr_scale = step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            initial_lr = args.lora_lr if 'weight_decay' in param_group and param_group['weight_decay'] == 0.001 else args.base_lr
            param_group['lr'] = initial_lr * lr_scale

        optimizer.zero_grad()
        out = model(inp)
        logits = out['logits']

        loss_raw = F.cross_entropy(logits.view(-1, cfg.vocab_size), tgt.view(-1), reduction='none')
        masked_loss = (loss_raw * mask.view(-1)).sum() / (mask.sum() + 1e-8)

        total_loss = masked_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        loss_val = masked_loss.item()
        ema_loss = 0.95 * ema_loss + 0.05 * loss_val

        if step % 5 == 0 or step == total_steps:
            elapsed = time.time() - start_time
            s_per_st = elapsed / step
            eta_h = (total_steps - step) * s_per_st / 3600
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  step {step:5d}/{total_steps}  loss={ema_loss:.4f}  ce={loss_val:.3f}  lr={current_lr:.1e}  {s_per_st:.3f}s/st  ETA:{eta_h:.1f}h  [{b_type}]  sft_ce={loss_val:.4f}")

        # Checkpoint saving
        now = time.time()
        if step % 100 == 0 or (now - last_save_time) > 1800 or step == total_steps:
            save_dict = {'model_state_dict': {k: v for k, v in model.state_dict().items() if v.requires_grad or 'lora' in k or 'router' in k or 'finalizer' in k}}
            torch.save(save_dict, str(latest_sft))
            last_save_time = now
            print(f"  [CKPT] Saved → {latest_sft.name} at step {step}")

    final_path = ckpt_dir / 'quillan_sft_final.pt'
    save_dict = {'model_state_dict': {k: v for k, v in model.state_dict().items() if v.requires_grad or 'lora' in k or 'router' in k or 'finalizer' in k}}
    torch.save(save_dict, str(final_path))
    print(f"\n==================================================")
    print(f"  PHASE 2.5 SFT REFINEMENT COMPLETE")
    print(f"  Saved final checkpoint: {final_path.name}")
    print(f"==================================================")

if __name__ == '__main__':
    main()
