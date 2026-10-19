#!/usr/bin/env python3
"""Quillan-Ronin — VRAM-optimized training for GTX 1050 4GB.

All hacks:
  - Weight tying
  - Adafactor (1 optimizer state)
  - Strip quant buffers from checkpoints
  - Mixed precision (fp16)
  - CuDNN autotuner + TF32
  - Frozen core experts in fp16, only ~16M trainable
  - Gradient accumulation
  - NaN-safe
  - Load ALL data
"""

import os, sys, time, gc, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig


class Adafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=(1e-30, 1e-3),
                 weight_decay=0.0, scale_parameter=True, relative_step=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        scale_parameter=scale_parameter, relative_step=relative_step)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']
                lr = group['lr']
                if group['relative_step']:
                    lr = min(lr, 1.0 / (step ** 0.5))
                decay_rate = beta1 if step > 1 else 1.0
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'][1])
                if group['scale_parameter']:
                    rms = p.data.norm() / (p.data.numel() ** 0.5 + 1e-10)
                    denom = denom * max(1.0, rms / 1.0)
                p.data.addcdiv_(grad, denom, value=-lr * max(1.0, 1.0 - decay_rate))
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * lr)
        return loss


class TokDataset(Dataset):
    def __init__(self, paths, seq=128):
        if isinstance(paths, str):
            paths = [paths]
        tensors = []
        for p in paths:
            if not os.path.exists(p): continue
            t = torch.load(p, weights_only=True, map_location='cpu')
            if isinstance(t, torch.Tensor):
                tensors.append(t.to(torch.long))
            print(f'  {Path(p).name}: loaded')
        self.t = torch.cat(tensors)
        self.seq = seq
        self.stride = seq // 2
        self.num_chunks = (len(self.t) - seq - 1) // self.stride
        print(f"  Loaded concatenated size: {len(self.t)} tokens. Total chunks: {self.num_chunks}")

    def __len__(self): return self.num_chunks
    def __getitem__(self, i):
        start = i * self.stride
        c = self.t[start : start + self.seq + 1]
        return c[:-1], c[1:]


def save_lean_checkpoint(model, path, step, loss):
    state = model.state_dict()
    clean = {k: v for k, v in state.items()
             if '_quant_' not in k
             and k not in ('moe._w1_quant', 'moe._wgate_quant', 'moe._w2_quant')}
    torch.save(dict(state_dict=clean, step=step, loss=loss, version='v5.3.1-train'), path)
    print(f'  [SAVED] {Path(path).name} ({os.path.getsize(path)/1e9:.2f} GB, step {step})')
    sys.stdout.flush()


def main():
    STEPS = 2000
    if len(sys.argv) > 1:
        try: STEPS = int(sys.argv[1])
        except: pass

    print(f'QUILLAN-RONIN — VRAM-OPTIMIZED ({STEPS} steps)')
    sys.stdout.flush()

    SEQ, BS, ACCUM, LR = 128, 1, 8, 3e-5
    SAVE_EVERY, LOG_EVERY = 500, 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu = torch.cuda.get_device_properties(0)
        print(f'  GPU: {gpu.name} ({gpu.total_memory/1e9:.1f} GB)')
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f'  Device: {device}')

    # ── Model ────────────────────────────────────────────────────────────
    print('\n[1] Loading model...'); sys.stdout.flush()
    cfg = QuillanArchConfig(device=device, pascal_mode=(device == 'cuda' and torch.cuda.get_device_capability()[0] < 7))
    model = QuillanRoninSovereign(cfg)

    ckpt_path = Path('checkpoints/quillan_finetuned.pt')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        sd = ckpt if 'state_dict' not in ckpt else ckpt['state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'  Loaded (Missing: {len(missing)}, Unexpected: {len(unexpected)})')
        del sd, ckpt
    else:
        fb = Path('checkpoints/quillan_transplanted_v8.pt')
        if fb.exists():
            ckpt = torch.load(fb, map_location='cpu', weights_only=True)
            sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f'  From transplant (Missing: {len(missing)}, Unexpected: {len(unexpected)})')
            del sd, ckpt
    gc.collect()

    # Ensure txt_dec is NOT tied to txt_emb so gradients can flow
    # (Weight tying was blocking training — loss stuck at ln(vocab))
    for attr in ['lora_A', 'lora_B']:
        if hasattr(model.txt_dec, attr):
            getattr(model.txt_dec, attr).requires_grad_(False)

    # HACK: Only train critical path (~16M params)
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if any(k in name for k in [
            'moe.router.', 'quillan_finalizer', 'pre_final_norm',
            'w1_lora', 'w2_lora', 'wgate_lora',
        ]):
            p.requires_grad_(True)

    model.to(device)
    # Keep model in native dtypes (fp16 for expert weights, fp32 for others)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'  Trainable: {n_train/1e6:.1f}M | Frozen: {n_frozen/1e6:.1f}M')
    if device == 'cuda':
        print(f'  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    sys.stdout.flush()

    # ── Data ─────────────────────────────────────────────────────────────
    print('\n[2] Loading data...'); sys.stdout.flush()
    data_dir = Path(r'C:\Users\Admin\Quillan-Ronin\training_data')
    files = [str(data_dir / f) for f in [
        'GPT_5.5_Distilled.pt', 'instruct_train.pt', 'full_train.pt',
        'code_train.pt', 'quillan_science_absolute.pt', 'quillan_science_additional.pt',
        'quillan_corpus_CLEAN_V7.pt', 'full_dataset.pt', 'train.pt',
        'quillan_12mb_training_dataset.pt',
    ] if (data_dir / f).exists()]
    dataset = TokDataset(files, seq=SEQ)
    loader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=0,
                        pin_memory=(device == 'cuda'))
    gc.collect()

    # HACK: Adafactor (1 state instead of 2)
    print('  [HACK] Adafactor...'); sys.stdout.flush()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adafactor(params, lr=LR, weight_decay=0.01,
                          scale_parameter=False, relative_step=False)

    if device == 'cuda':
        print(f'  VRAM ready: {torch.cuda.memory_allocated()/1024**2:.0f}MB')

    # ── Training ─────────────────────────────────────────────────────────
    print(f'\n[3] Training {STEPS} steps...'); sys.stdout.flush()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = micro_step = 0
    nan_count = 0
    t0 = time.time()
    loss_ce_ema = None
    grad_norm_val = 0.0

    data_iter = iter(loader)
    while step < STEPS:
        try:
            inp, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, tgt = next(data_iter)

        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # No autocast — fp16 overflows in deep model stack
        out = model(inp)
        logits = out['logits'].float()
        routing_loss = out.get('routing_loss', torch.tensor(0.0, device=device))
        loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss_total = loss_ce + routing_loss

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            nan_count += 1
            if nan_count <= 5:
                print(f'  NaN loss ({nan_count}/5)')
            if nan_count >= 5:
                print('  FATAL: 5 NaN batches'); sys.exit(1)
            optimizer.zero_grad(set_to_none=True)
            continue
        nan_count = 0
        loss_scaled = loss_total / ACCUM

        loss_scaled.backward()
        micro_step += 1

        ce_val = loss_ce.item()
        if loss_ce_ema is None:
            loss_ce_ema = ce_val
        else:
            loss_ce_ema = 0.95 * loss_ce_ema + 0.05 * ce_val

        if micro_step % ACCUM == 0:
            grad_norm_val = torch.nn.utils.clip_grad_norm_(params, 50.0)

            if torch.isnan(grad_norm_val) or torch.isinf(grad_norm_val):
                print(f'  NaN grad at step {step}, resetting...')
                optimizer.zero_grad(set_to_none=True)
                nan_count += 1
                if nan_count >= 5:
                    print('  FATAL: 5 NaN grad steps'); sys.exit(1)
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                sps = step / max(time.time() - t0, 1e-6)
                vram = f' | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB' if device == 'cuda' else ''
                print(f'  step {step:>4}/{STEPS} | ce {loss_ce_ema:.4f} | grad {grad_norm_val:.2f} | {sps:.2f} st/s{vram}')
                sys.stdout.flush()

            if step % SAVE_EVERY == 0:
                save_lean_checkpoint(model, 'checkpoints/quillan_finetuned.pt', step, loss_ce_ema)
                if device == 'cuda':
                    torch.cuda.empty_cache()

            if step >= STEPS:
                break

    save_lean_checkpoint(model, 'checkpoints/quillan_finetuned.pt', step, loss_ce_ema)
    print(f'\nDONE — {step} steps, final loss {loss_ce_ema:.4f}')
    if device == 'cuda':
        print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')

if __name__ == '__main__':
    main()
