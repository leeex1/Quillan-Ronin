#!/usr/bin/env python3
"""Lightweight CPU-optimized training (100 steps) to validate fixes without stressing hardware"""
import os, sys, time, torch, torch.nn.functional as F, gc
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

class TokDataset(Dataset):
    def __init__(self, paths, seq=64):  # Reduced seq length
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
        self.stride = seq
        self.num_chunks = (len(self.t) - seq - 1) // self.stride
        print(f"  Loaded concatenated size: {len(self.t)} tokens. Total chunks: {self.num_chunks}")

    def __len__(self): return self.num_chunks
    def __getitem__(self, i):
        start = i * self.stride
        c = self.t[start : start + self.seq + 1]
        return c[:-1], c[1:]

def main():
    STEPS = 100  # Conservative for validation
    print(f"LIGHTWEIGHT TRAINING ({STEPS} steps, CPU-optimized)")

    # Reduced resource settings
    SEQ, BS, LR = 64, 1, 2e-5  # Smaller seq, lower LR
    ACCUM_STEPS = 1  # No accumulation to reduce memory

    print(f"  Seq: {SEQ}, Batch: {BS}, LR: {LR}, Accum: {ACCUM_STEPS}")

    device = 'cpu'  # Force CPU
    print(f"  Device: {device}")

    print("\n[1] Loading model...")
    model = QuillanRoninSovereign(QuillanArchConfig(device=device))

    # Load checkpoint
    if os.path.exists("checkpoints/quillan_transplanted_v8.pt"):
        ckpt = torch.load("checkpoints/quillan_transplanted_v8.pt", map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Loaded transplanted_v8.pt (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        del ckpt
    else:
        print("  ERROR: No checkpoint found")
        sys.exit(1)

    gc.collect()
    model.eval()

    # Scale routers
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'moe.router.fast_router.weight' in name or 'moe.router.balanced_router.weight' in name or 'moe.router.diffusion_router.weight' in name:
                p.data.mul_(15.0)

    # Minimal trainable set - only critical components
    trainable_keys = ['router', 'decomposition', 'quillan_finalizer', 'txt_dec', 'expert_swarms']
    for name, p in model.named_parameters():
        p.requires_grad = any(t in name for t in trainable_keys)

    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    print(f"  Trainable: {n/1e6:.1f}M parameters (minimal set)")

    model.to(device)

    print("\n[2] Loading data...")
    data_dir = r'C:\Users\Admin\Quillan-Ronin\training_data'
    files = [os.path.join(data_dir, f) for f in ['full_train.pt'] if os.path.exists(os.path.join(data_dir, f))]
    if not files:
        print("  ERROR: No training data found")
        sys.exit(1)

    dataset = TokDataset(files, seq=SEQ)
    loader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)

    print(f"\n[3] Training {STEPS} steps...")
    step = 0
    t0 = time.time()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < STEPS:
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)

            optimizer.zero_grad()
            out = model(inp)
            logits = out['logits']
            rl = out.get('routing_loss', torch.tensor(0.0, device=device))
            loss_ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss = loss_ce + rl

            # NaN check
            if torch.isnan(loss).item():
                print(f"  FATAL: NaN at step {step}")
                sys.exit(1)

            loss.backward()

            # Gradient clipping
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)

            optimizer.step()
            scheduler.step()

            step += 1

            # Log every 10 steps
            if step % 10 == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {step:>3}/{STEPS} | ce_loss {loss_ce.item():.4f} | routing {rl.item():.4f} | grad_norm {gn:.2f} | lr {lr_now:.2e} | {step/elapsed:.2f} steps/s")

                # Memory check
                if step % 50 == 0:
                    mem_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    print(f"    Memory: {mem_mb:.0f}MB GPU | CPU: {sys.getsizeof(model)/1024/1024:.0f}MB")
                    gc.collect()

            if step >= STEPS:
                break

    print(f"\nDONE - {STEPS} steps, final loss {loss_ce.item():.4f}")
    print(f"Time: {time.time() - t0:.1f}s")

    # Save if loss improved
    if loss_ce.item() < 10.0:  # Below random baseline
        torch.save(model.state_dict(), 'checkpoints/quillan_lightweight.pt')
        print("Saved to checkpoints/quillan_lightweight.pt")
        print(">> FIXES VALIDATED - Ready for full training <<")
    else:
        print(">> Loss still high - may need more investigation <<")

if __name__ == '__main__':
    main()
