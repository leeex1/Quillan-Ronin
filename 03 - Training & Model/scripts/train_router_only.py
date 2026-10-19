#!/usr/bin/env python3
"""Quillan-Ronin — Router Isolation Training (Stage A)

Train ONLY the ComplexityRouter to learn input-dependent routing.
Freezes all 34 expert FFNs. Uses routing diversity loss instead of CE.

Why: The hidden state is identical for all inputs because the random-init
router sends every token to every expert uniformly. Training the router
to produce diverse expert assignments makes hidden states input-dependent.

Architecture:
  ComplexityRouter (trainable ~1.2M params):
    - complexity_classifier: Linear(2048->512) + Linear(512->3)
    - fast_router, balanced_router, diffusion_router: BitLinear(2048->34)
  All expert weights (w1/wgate/w2): frozen
  All EGGROLL swarm params: frozen
  txt_dec + quillan_finalizer: frozen (trained in Stage B)

Loss:
  L = -var(expert_assignments) + 0.01 * load_balance_loss
  Maximize routing diversity so different tokens hit different experts.
"""

import os, sys, time, gc, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig


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


def main():
    STEPS = 2000
    if len(sys.argv) > 1:
        try: STEPS = int(sys.argv[1])
        except: pass

    print(f'QUILLAN-ROININ — ROUTER ISOLATION TRAINING ({STEPS} steps)')
    sys.stdout.flush()

    SEQ, BS, ACCUM, LR = 128, 1, 8, 1e-4
    SAVE_EVERY, LOG_EVERY = 500, 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu = torch.cuda.get_device_properties(0)
        print(f'  GPU: {gpu.name} ({gpu.total_memory/1e9:.1f} GB)')
    print(f'  Device: {device}')

    # ── Model ────────────────────────────────────────────────────────────
    print('\n[1] Loading model...'); sys.stdout.flush()
    is_pascal = device == 'cuda' and torch.cuda.get_device_capability()[0] < 7
    cfg = QuillanArchConfig(device=device, pascal_mode=is_pascal)
    model = QuillanRoninSovereign(cfg)

    # Load prebuilt checkpoint (with zero-padded expert weights)
    ckpt_path = Path('checkpoints/quillan_fixed.pt')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        sd = ckpt if 'state_dict' not in ckpt else ckpt['state_dict']
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'  Loaded quillan_fixed.pt (Missing: {len(missing)}, Unexpected: {len(unexpected)})')
        del sd, ckpt
    else:
        print(f'  WARNING: No checkpoint found at {ckpt_path}. Using random init.')
    gc.collect()

    # ── Freeze everything except the ComplexityRouter ────────────────────
    for name, p in model.named_parameters():
        p.requires_grad_(False)

    for name, p in model.named_parameters():
        if 'moe.router.' in name:
            p.requires_grad_(True)
        if 'moe.router.complexity_classifier' in name:
            p.requires_grad_(True)

    # Convert model to half-precision to fit in 4GB VRAM
    # Keep router params in fp32 for stable training
    model.half()
    for name, p in model.named_parameters():
        if 'moe.router' in name:
            p.data = p.data.float()
            if p.requires_grad:
                p.requires_grad_(True)

    model.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'  Trainable: {n_train/1e6:.2f}M / {n_total/1e6:.1f}M total')
    if device == 'cuda':
        before = torch.cuda.memory_allocated() / 1024**2
        print(f'  VRAM before model: {before:.0f}MB')

    # ── Data ─────────────────────────────────────────────────────────────
    print('\n[2] Loading data...'); sys.stdout.flush()
    data_dir = Path(r'C:\Users\Admin\Quillan-Ronin\training_data')
    files = [str(data_dir / f) for f in [
        'GPT_5.5_Distilled.pt', 'instruct_train.pt', 'full_train.pt',
        'code_train.pt', 'quillan_science_absolute.pt', 'quillan_science_additional.pt',
        'full_dataset.pt', 'train.pt', 'quillan_12mb_training_dataset.pt',
    ] if (data_dir / f).exists()]
    dataset = TokDataset(files, seq=SEQ)
    loader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=0,
                        pin_memory=(device == 'cuda'))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01
    )

    if device == 'cuda':
        after = torch.cuda.memory_allocated() / 1024**2
        print(f'  VRAM ready: {after:.0f}MB (+{after-before:.0f}MB for model/data)')

    # ── Training ─────────────────────────────────────────────────────────
    print(f'\n[3] Training {STEPS} steps...'); sys.stdout.flush()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = micro_step = 0
    t0 = time.time()

    data_iter = iter(loader)
    while step < STEPS:
        try:
            inp, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            inp, tgt = next(data_iter)

        inp = inp.to(device, non_blocking=True)

        # Forward pass — we only need routing diversity, not CE loss
        with torch.no_grad():
            z = model.ingestion(inp)
            blueprint = model.decomposition(z)
            x_diff, _ = model.diffusion_core(
                blueprint,
                torch.ones(blueprint.shape[0], blueprint.shape[1], device=device, dtype=blueprint.dtype)
            )

        # Router forward — compute routing probabilities
        # Cast router input to fp32 (router params are fp32 for stable training)
        routing_weights, path_weights, path_indices = model.moe.router(x_diff.float())

        # Diversity loss: maximize variance of TOP-1 assignments across tokens
        # This directly encourages different tokens to route to different experts
        top1_idx = routing_weights.argmax(dim=-1)  # [N] — which expert each token prefers
        counts = torch.bincount(top1_idx, minlength=34).float()  # [34] — token count per expert
        top1_var = counts.var()  # High variance = diverse routing
        diversity_loss = -top1_var

        # Load balance loss: prevent expert starvation
        eps_balance = 1e-8
        probs_mean = routing_weights.mean(dim=0) + eps_balance
        probs_mean = probs_mean / probs_mean.sum()
        uniform = torch.ones_like(probs_mean) / probs_mean.shape[0]
        balance_loss = F.kl_div(probs_mean.log(), uniform, reduction='batchmean')

        # Path diversity: encourage tokens to use all 3 router paths
        path_counts = torch.bincount(path_indices, minlength=3).float()
        path_diversity = -path_counts.var()

        loss_total = diversity_loss + 0.1 * balance_loss + 0.05 * path_diversity

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_scaled = loss_total / ACCUM
        loss_scaled.backward()

        micro_step += 1

        if micro_step % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                sps = step / max(time.time() - t0, 1e-6)
                var_val = per_token_var.mean().item()
                balance_val = balance_loss.item()
                vram = f' | VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB' if device == 'cuda' else ''
                print(f'  step {step:>4}/{STEPS} | var {var_val:.4f} | bal {balance_val:.4f} | {sps:.2f} st/s{vram}')
                sys.stdout.flush()

            if step % SAVE_EVERY == 0:
                torch.save({
                    'state_dict': model.state_dict(),
                    'step': step,
                    'config': cfg,
                }, 'checkpoints/router_trained.pt')
                print(f'  [SAVED] router_trained.pt (step {step})')
                if device == 'cuda':
                    torch.cuda.empty_cache()

            if step >= STEPS:
                break

    # Save final router checkpoint
    torch.save({
        'state_dict': model.state_dict(),
        'step': step,
        'config': cfg,
    }, 'checkpoints/router_trained.pt')
    print(f'\nDONE — {step} steps')
    print('Router training complete. Proceed to Stage B (train_finetune_v2.py).')


if __name__ == '__main__':
    main()
