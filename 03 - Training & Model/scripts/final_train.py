#!/usr/bin/env python3
"""
Quillan-Ronin Final Training (GTX 1050 Optimized)
=================================================
Papers leveraged:
- Paper 9 (BitDistill): 1.58-bit ternary STE training via _weight_quant
- Paper 20 (zFLoRA): EGGROLL fused LoRA adapters (disabled during training)
- Paper 18 (FP16): fp16 for all weights, trains on GPU with AMP
- Paper 3 (TRM): Small dataset, recursive refinement via 5-wave diffusion
- Paper 8 (MCMC): Base model already contains latent capabilities
"""
import sys, torch, os, time
os.environ['PYTHONUNBUFFERED'] = '1'
os.chdir(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin\_dev')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig, BitLinear
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Device detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pascal = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7
if pascal:
    print('Pascal mode (GTX 1050): fp16 compute')

# NOTE: Do NOT disable EGGROLL — LoRA adapters need to be in the forward
# computation graph to receive gradients. With eggroll=False, lora_A/lora_B
# are excluded from the forward pass and never train.

print('Building model...', flush=True)
cfg = QuillanArchConfig(
    device='cpu', pascal_mode=pascal, text_only=True,
    low_mem=True, low_gpu=(device != 'cuda'), top_k=2
)
model = QuillanRoninSovereign(cfg)
print('Model built. Loading checkpoint...', flush=True)

ckpt = torch.load('checkpoints/router_trained.pt', map_location='cpu', weights_only=False)
print('Checkpoint loaded. Applying state dict...', flush=True)
model.load_state_dict(ckpt['state_dict'], strict=False)

print('Freezing/unfreezing params...', flush=True)
for n, p in model.named_parameters(): p.requires_grad_(False)
for n, p in model.named_parameters():
    if any(k in n for k in ['quillan_finalizer','quillan_gate','txt_dec','pre_final_norm']):
        p.requires_grad_(True)
    if any(f'moe.{k}_lora' in n for k in ['w1','w2','wgate']):
        p.requires_grad_(True)

n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f'Trainable: {n_t:,} / {n_all:,} ({100*n_t/n_all:.1f}%)')

# Move to device in fp16 (fits in 4GB VRAM)
model = model.to(device)
if device == 'cuda':
    model = model.half()
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')
    torch.cuda.synchronize()
    print(f'VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB')

model.train()

# QuillanFusedOptimizer: Muon (matrix weights) + AdamW (scalars)
# Lower LR for stability with diverse data
from quillan_fused_optimizer import QuillanFusedOptimizer
opt = QuillanFusedOptimizer(
    model,
    lr_muon=0.01, lr_adamw=1e-4,
    momentum=0.95, weight_decay=0.01,
    use_sophia=True, ns_steps=3,
    warmup=100, total_steps=1500,
)

trainable = [p for p in model.parameters() if p.requires_grad]

# Load ALL training datasets separately (no concat — avoids cross-boundary garbage)
print('Loading training data...', flush=True)
data_files = ['full_train.pt', 'code_train.pt', 'instruct_train.pt', 
              'quillan_science_absolute.pt', 'quillan_science_additional.pt',
              'quillan_corpus_CLEAN_V7.pt', 'quillan_12mb_training_dataset.pt',
              'GPT_5.5_Distilled.pt']
datasets = []
for f in data_files:
    d = torch.load(f'training_data/{f}', map_location='cpu', weights_only=True)
    print(f'  {f}: {len(d):,} tokens', flush=True)
    if device == 'cuda':
        d = d.half().cuda()
    datasets.append(d)
total_tokens = sum(len(d) for d in datasets)
print(f'Total: {total_tokens:,} tokens ({total_tokens/1e6:.1f}M) across {len(datasets)} datasets', flush=True)

t0 = time.time()
first = None
total = 1500
save_interval = 250

for step in range(total):
    # Pick a random dataset, then a random chunk within it (preserves coherence)
    ds = datasets[torch.randint(0, len(datasets), (1,)).item()]
    ctx = 128
    idx = torch.randint(0, len(ds)-ctx-1, (1,)).item()
    inp = ds[idx:idx+ctx].long().unsqueeze(0)
    tgt = ds[idx+1:idx+ctx+1].long().unsqueeze(0)

    opt.zero_grad()
    with autocast(enabled=(device=='cuda')):
        out = model(inp)
        loss = F.cross_entropy(out['logits'].reshape(-1, out['logits'].size(-1)), tgt.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 5.0)
    opt.step()

    if step == 0: first = loss.item()
    if step % 50 == 0 or step == total - 1:
        sps = (step+1) / (time.time()-t0)
        info = f'Step {step:>4}/{total}: CE={loss.item():.4f} ({time.time()-t0:.0f}s, {sps:.2f} st/s'
        if device == 'cuda':
            info += f', VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB'
        print(info + ')', flush=True)
    
    # Periodic save (keeps on GPU, torch handles CPU copy internally)
    if step > 0 and step % save_interval == 0:
        torch.save({'state_dict': model.state_dict(), 'step': step}, f'checkpoints/model_final_step{step}.pt')
        print(f'  Saved checkpoint at step {step}', flush=True)

print(f'\nDone! CE: {first:.4f} -> {loss.item():.4f}')

# Save final
model.cpu()
torch.save({'state_dict': model.state_dict(), 'step': total}, 'checkpoints/model_final.pt')
gb = os.path.getsize('checkpoints/model_final.pt') / 1e9
print(f'Saved model_final.pt ({gb:.2f} GB)')
