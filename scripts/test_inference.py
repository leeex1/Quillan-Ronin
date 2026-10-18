#!/usr/bin/env python3
import sys, torch, os, time
os.chdir(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin\_dev')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
print('Loading...', flush=True)
t0 = time.time()
cfg = QuillanArchConfig(device='cpu', pascal_mode=True, text_only=True, top_k=4)
model = QuillanRoninSovereign(cfg)
ckpt = torch.load('checkpoints/router_trained.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.cuda().half().eval()
print(f'Loaded in {time.time()-t0:.1f}s', flush=True)
inp = torch.randint(0, 100, (1, 1)).cuda()
t1 = time.time()
with torch.no_grad():
    out = model(inp)
print(f'Forward (1 token): {time.time()-t1:.1f}s', flush=True)
probs = torch.softmax(out['logits'][0, 0], dim=-1)
entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
print(f'Entropy: {entropy:.4f}', flush=True)
print(f'Top-5: {probs.topk(5).indices.tolist()}', flush=True)
print(f'VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB', flush=True)
