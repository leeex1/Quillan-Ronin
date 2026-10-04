#!/usr/bin/env python3
"""Quillan-Ronin Phase 6: Fine-tune (2000 steps default)"""
import os, sys, time, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from transplant_weights import QuillanConfig

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
        self.stride = seq
        self.num_chunks = (len(self.t) - seq - 1) // self.stride
        print(f"  Loaded {len(paths)} datasets, concatenated size: {len(self.t)} tokens. Total chunks: {self.num_chunks}")

    def __len__(self): return self.num_chunks
    def __getitem__(self, i):
        start = i * self.stride
        c = self.t[start : start + self.seq + 1]
        return c[:-1], c[1:]

def main():
    STEPS = 2000
    if len(sys.argv) > 1:
        try:
            STEPS = int(sys.argv[1])
        except ValueError:
            pass
    print(f"QUILLAN-RONIN PHASE 6: FINE-TUNE ({STEPS} steps)")

    SEQ, BS, LR = 128, 1, 3e-5
    VOCAB = 50257

    print("\n[1] Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        try:
            torch.ones(1, device='cuda')
        except Exception as e:
            print(f"  WARNING: GPU is incompatible with current PyTorch CUDA build ({e}). Falling back to CPU.")
            device = 'cpu'
    print(f"  Using device: {device}")
    model = QuillanRoninSovereign(QuillanArchConfig(device=device))
    # Resume from finetuned checkpoint if available, else fall back to transplanted
    if os.path.exists("checkpoints/quillan_finetuned.pt"):
        ckpt = torch.load("checkpoints/quillan_finetuned.pt", map_location='cpu', weights_only=True)
        sd = ckpt if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt else ckpt['model_state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Resuming from quillan_finetuned.pt (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        del sd
        del ckpt
    elif os.path.exists("checkpoints/quillan_transplanted_v8.pt"):
        ckpt = torch.load("checkpoints/quillan_transplanted_v8.pt", map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  Starting from transplanted_v8.pt (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        del ckpt
    else:
        print("  WARNING: No checkpoint found, starting from scratch")
        
    import gc
    gc.collect()
    model.eval()

    # Scale up the router weights in-place to make the logits larger than Gumbel noise.
    # We only apply this 15x scaling if we are starting from the raw transplanted checkpoint.
    # If resuming from a fine-tuned checkpoint, the weights are already scaled and saved.
    if not os.path.exists("checkpoints/quillan_finetuned.pt"):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'moe.router.fast_router.weight' in name or 'moe.router.balanced_router.weight' in name or 'moe.router.diffusion_router.weight' in name:
                    p.data.mul_(15.0)
                    print(f"  Scaled up router weights: {name} by 15x")

    # VRAM budget: 4GB GTX 1050
    # Train: txt_dec (output) + eggroll/LoRA adapters (all layers) + router + finalizer
    # This lets every BitLinear layer adapt via tiny rank-16 adapters (~2M extra params)
    trainable_sets = ['txt_dec', 'pre_final_norm', 'lora_A', 'lora_B',
                      'moe.router', 'quillan_finalizer', 'decomposition',
                      'w1_lora', 'wgate_lora', 'w2_lora']
    for name, p in model.named_parameters():
        p.requires_grad = any(t in name for t in trainable_sets)

    model.to(device)

    # Convert frozen params to FP16 on-device to save VRAM
    for name, p in model.named_parameters():
        if not p.requires_grad and p.device.type == 'cuda':
            p.data = p.data.half()

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  {n/1e6:.1f}M trainable (txt_dec + eggroll + router + decomp + finalizer)")
    print(f"  {frozen/1e6:.1f}M frozen (FP16)")
    if device == 'cuda':
        print(f"  Est VRAM: trainable {n*2/1024**3:.1f}GB + frozen {frozen*2/1024**3:.1f}GB + opt ~{n*4/1024**3:.1f}GB")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print("\n[2] Loading data...")
    data_dir = r'C:\Users\Admin\Quillan-Ronin\training_data'
    # Use all available tokenized training data for maximum learning signal
    candidate_files = ['full_train.pt', 'code_train.pt', 'GPT_5.5_Distilled.pt',
                       'instruct_train.pt', 'quillan_science_absolute.pt',
                       'quillan_science_additional.pt']
    files = [os.path.join(data_dir, f) for f in candidate_files if os.path.exists(os.path.join(data_dir, f))]
    for f in files: print(f"  + {os.path.basename(f)}")

    dataset = TokDataset(files, seq=SEQ)
    loader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    nan_count = 0  # Track consecutive NaN batches

    ACCUM_STEPS = 4  # Gradient accumulation for effective batch size of 4
    import gc
    gc.collect()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB used / {torch.cuda.memory_reserved()/1024**2:.0f}MB reserved")

    # NaN/Inf diagnostic: check model weights before training
    print("  Checking weights for NaN/Inf...")
    nan_count = sum(torch.isnan(p).any().item() for p in model.parameters())
    inf_count = sum(torch.isinf(p).any().item() for p in model.parameters())
    if nan_count > 0 or inf_count > 0:
        print(f"  FATAL: {nan_count} NaN tensors, {inf_count} Inf tensors found in model!")
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f"    NaN in: {name} [{p.shape}]")
        sys.exit(1)
    print(f"  Weights clean (0 NaN, 0 Inf)")

    # Validate token IDs in training data
    print("  Validating training data token IDs...")
    sample_tensor = torch.load(files[0], weights_only=True, map_location='cpu')
    max_id = sample_tensor.max().item()
    min_id = sample_tensor.min().item()
    print(f"  Token IDs: min={min_id}, max={max_id}, vocab_size={VOCAB}")
    if max_id >= VOCAB:
        print(f"  WARNING: Token IDs exceed vocab size ({max_id} >= {vocab})!")
        print(f"  Clamping will be needed")
    del sample_tensor
    gc.collect()

    print(f"\n[3] Training {STEPS} steps (accum={ACCUM_STEPS})...")
    if device == 'cuda':
        print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB used / {torch.cuda.memory_reserved()/1024**2:.0f}MB reserved")
        if torch.cuda.memory_allocated() > 3.8 * 1024**3:
            print(f"  WARNING: VRAM > 3.8GB, may OOM!")
    step = 0
    micro_step = 0
    t0 = time.time()
    if device == 'cuda':
        torch.cuda.empty_cache()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < STEPS:
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            
            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda'), dtype=torch.float16):
                out = model(inp)
                logits = out['logits']

                # Retrieve MoE auxiliary load balancing loss
                routing_loss = out.get('routing_loss', torch.tensor(0.0, device=device))
                
                loss_ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
                loss_total = loss_ce + routing_loss
                
                # NaN safety: skip bad batches
                if torch.isnan(loss_total) or torch.isinf(loss_total):
                    nan_count += 1
                    if nan_count <= 5:
                        print(f"  WARNING: NaN loss (ce={loss_ce.item() if not torch.isnan(loss_ce) else 'nan'}, routing={routing_loss.item() if isinstance(routing_loss, torch.Tensor) else 0}) at step {micro_step} ({nan_count}/5)")
                    if nan_count >= 5:
                        print(f"  FATAL: 5 consecutive NaN batches. Saving diagnostic...")
                        torch.save(model.state_dict(), 'checkpoints/quillan_finetuned.pt')
                        sys.exit(1)
                    optimizer.zero_grad(set_to_none=True)
                    continue
                nan_count = 0  # Reset on clean batch
                
                loss_scaled = loss_total / ACCUM_STEPS  # Scale for accumulation
                
            scaler.scale(loss_scaled).backward()
            micro_step += 1
            
            # Initialize EMAs on first step, otherwise update
            if 'loss_ce_ema' not in locals():
                loss_ce_ema = loss_ce.item()
                loss_total_ema = loss_total.item()
            else:
                loss_ce_ema = 0.95 * loss_ce_ema + 0.05 * loss_ce.item()
                loss_total_ema = 0.95 * loss_total_ema + 0.05 * loss_total.item()
            
            if micro_step % ACCUM_STEPS == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                step += 1
                
                # Log every step for first 50, then every 10
                if step <= 50 or step % 10 == 0:
                    elapsed = time.time() - t0
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  step {step:>4}/{STEPS} | total_loss {loss_total_ema:.4f} | ce_loss {loss_ce_ema:.4f} | grad_norm {grad_norm:.2f} | lr {lr_now:.2e} | {step/elapsed:.2f} steps/s")
                
                if step % 500 == 0:
                    torch.save(model.state_dict(), 'checkpoints/quillan_finetuned.pt')
                    print(f"  [saved] step {step}")
                
                if step >= STEPS:
                    break

    torch.save(model.state_dict(), 'checkpoints/quillan_finetuned.pt')
    print(f"\nDONE - {STEPS} steps, final loss {loss_ce_ema:.4f}")
    print("Saved to checkpoints/quillan_finetuned.pt")

if __name__ == '__main__':
    main()
