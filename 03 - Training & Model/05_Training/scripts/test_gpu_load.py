#!/usr/bin/env python3
"""GPU load test with fp16 only (no fp32 trainable params)"""
import sys, torch, os, gc
os.chdir(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig, BitLinear
from torch.cuda.amp import autocast

BitLinear.set_global_eggroll(False)
cfg = QuillanArchConfig(device='cpu', pascal_mode=True, text_only=True, low_mem=True, top_k=2)
model = QuillanRoninSovereign(cfg)

ckpt = torch.load('checkpoints/router_trained.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)

for n, p in model.named_parameters(): p.requires_grad_(False)
for n, p in model.named_parameters():
    if any(k in n for k in ['quillan_finalizer','quillan_gate','txt_dec','pre_final_norm']):
        p.requires_grad_(True)
    if any(f'moe.{k}_lora' in n for k in ['w1','w2','wgate']):
        p.requires_grad_(True)

# Everything in fp16 on CUDA (including trainable params)
model = model.cuda().half()

n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f'Trainable: {n_t:,} / {n_all:,} total')
print(f'GPU: {torch.cuda.get_device_name(0)}')
torch.cuda.synchronize()
print(f'VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')

# Forward
model.eval()
inp = torch.randint(0, 100, (1, 64)).cuda()
with torch.no_grad():
    out = model(inp)

# Training step
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
with autocast():
    out2 = model(inp)
    loss = torch.nn.functional.cross_entropy(
        out2['logits'].reshape(-1, out2['logits'].size(-1)), inp.reshape(-1)
    )
opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
opt.step()
torch.cuda.synchronize()
print(f'Train step VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB')
print(f'Loss: {loss.item():.4f}')
print('Test complete!')
