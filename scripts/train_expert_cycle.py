#!/usr/bin/env python3
"""
Quillan-Ronin Expert-Cycling Training
Trains 1 expert at a time in full FP32 — lets each expert actually learn
without exceeding 4GB VRAM. Cycles through all 34 experts.
"""
import os, sys, time, torch, torch.nn.functional as F, gc
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

STEPS_PER_EXPERT = 200
NUM_EXPERTS = 34
SEQ, BS, LR = 128, 1, 1e-5
ACCUM_STEPS = 4

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
        print(f"  Loaded concatenated size: {len(self.t)} tokens. Total chunks: {self.num_chunks}")
    def __len__(self): return self.num_chunks
    def __getitem__(self, i):
        start = i * self.stride
        c = self.t[start : start + self.seq + 1]
        return c[:-1], c[1:]

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"QUILLAN EXPERT CYCLING — {NUM_EXPERTS} experts, {STEPS_PER_EXPERT} steps each")

    model = QuillanRoninSovereign(QuillanArchConfig(device=device))
    ckpt = torch.load("checkpoints/quillan_finetuned.pt", map_location='cpu', weights_only=True)
    sd = ckpt if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt else ckpt['model_state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint (missing={len(missing)}, unexpected={len(unexpected)})")
    del ckpt, sd; gc.collect()

    # Always-trainable: txt_dec + router + LoRA adapters (tiny overhead)
    always_train = ['txt_dec', 'pre_final_norm', 'lora_A', 'lora_B',
                    'moe.router', 'quillan_finalizer', 'decomposition']

    # Expert cycling: train ONE expert's full weights at a time
    for cycle_start in range(0, NUM_EXPERTS):
        current_expert = cycle_start
        print(f"\n{'='*60}")
        print(f"CYCLE: Training expert C{current_expert} ({current_expert}/{NUM_EXPERTS-1})")
        print(f"{'='*60}")

        # Set requires_grad
        for name, p in model.named_parameters():
            p.requires_grad = any(t in name for t in always_train)
            # Also train this expert's raw FFN weights
            if f'moe.w1' in name or f'moe.wgate' in name or f'moe.w2' in name:
                # Only make the slice for THIS expert trainable
                # We store full [34, ...] tensor, but only 1 expert's slice needs grad
                # PyTorch doesn't support per-slice requires_grad on a single Parameter.
                # So we train ALL w1/wgate/w2 but with frozen non-active experts via zero-grad trick.
                p.requires_grad = True  # We'll zero-out non-target expert grads in optimizer hook

        model.to(device)
        for name, p in model.named_parameters():
            if not p.requires_grad and p.device.type == 'cuda':
                p.data = p.data.half()

        n = count_trainable(model)
        print(f"  Trainable: {n/1e6:.1f}M")

        # Prepare data
        data_dir = r'C:\Users\Admin\Quillan-Ronin\training_data'
        files = [os.path.join(data_dir, f) for f in
                 ['full_train.pt', 'code_train.pt', 'GPT_5.5_Distilled.pt',
                  'instruct_train.pt', 'quillan_science_absolute.pt',
                  'quillan_science_additional.pt']
                 if os.path.exists(os.path.join(data_dir, f))]
        dataset = TokDataset(files, seq=SEQ)
        loader = DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=0)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

        # Hook to zero gradients for non-active experts
        def zero_non_target_grad():
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'moe.w1' in name or 'moe.wgate' in name or 'moe.w2' in name:
                        if p.grad is not None:
                            # Zero out grads for experts that aren't the current target
                            for e_idx in range(NUM_EXPERTS):
                                if e_idx != current_expert:
                                    p.grad[e_idx].zero_()

        print(f"\nTraining expert C{current_expert} for {STEPS_PER_EXPERT} steps...")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step = 0; micro_step = 0
        t0 = time.time()

        while step < STEPS_PER_EXPERT:
            for inp, tgt in loader:
                inp, tgt = inp.to(device), tgt.to(device)

                with torch.amp.autocast(device_type=device, enabled=(device == 'cuda'), dtype=torch.float16):
                    out = model(inp)
                    logits = out['logits']
                    routing_loss = out.get('routing_loss', torch.tensor(0.0, device=device))
                    loss_ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
                    loss_total = loss_ce + routing_loss

                    if torch.isnan(loss_total) or torch.isinf(loss_total):
                        print(f"  WARNING: NaN at micro_step {micro_step}, skipping")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss_scaled = loss_total / ACCUM_STEPS

                scaler.scale(loss_scaled).backward()
                micro_step += 1

                if micro_step % ACCUM_STEPS == 0:
                    # Zero out gradients for non-active experts
                    zero_non_target_grad()

                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % 20 == 0 or step <= 10:
                        elapsed = time.time() - t0
                        print(f"  Expert C{current_expert:>2} step {step:>3}/{STEPS_PER_EXPERT} | ce_loss {loss_ce.item():.4f} | grad_norm {grad_norm:.2f} | {step/elapsed:.2f} steps/s")

                    if step >= STEPS_PER_EXPERT:
                        break

    # Final save
    torch.save(model.state_dict(), 'checkpoints/quillan_expert_trained.pt')
    print(f"\nDONE - All {NUM_EXPERTS} experts trained")
    print("Saved to checkpoints/quillan_expert_trained.pt")

if __name__ == '__main__':
    main()
