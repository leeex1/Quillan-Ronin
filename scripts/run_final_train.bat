@echo off
cd /d C:\Users\Admin\Quillan-Ronin
C:\Users\Admin\Quillan-Ronin\.venv-cuda\Scripts\python.exe -c "
import sys, torch, os, time
os.chdir(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig, BitLinear
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on {device} - {torch.cuda.get_device_properties(0).name if device==\"cuda\" else \"CPU\"}')
BitLinear.set_global_eggroll(False)

cfg = QuillanArchConfig(device=device, pascal_mode=True, text_only=True, low_mem=True, top_k=2)
model = QuillanRoninSovereign(cfg)
ckpt = torch.load('checkpoints/router_trained.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.half(); model.to(device)

for n, p in model.named_parameters(): p.requires_grad_(False)
for n, p in model.named_parameters():
    if any(f'moe.{k}_lora' in n for k in ['w1','w2','wgate']): p.requires_grad_(True)
    if any(k in n for k in ['quillan_finalizer','quillan_gate','txt_dec','pre_final_norm']): p.requires_grad_(True)

model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable: {n_t:,} / {sum(p.numel() for p in model.parameters()):,} total')
print(f'Estimated time: ~{500*21/60:.0f} minutes')

SEEDS = torch.load('training_data/GPT_5.5_Distilled.pt', map_location='cpu', weights_only=True)
t0 = time.time()
first = None

for step in range(500):
    idx = torch.randint(0, len(SEEDS)-65, (1,)).item()
    inp = SEEDS[idx:idx+64].to(device).long().unsqueeze(0)
    tgt = SEEDS[idx+1:idx+65].to(device).long().unsqueeze(0)
    opt.zero_grad()
    out = model(inp)
    loss = F.cross_entropy(out['logits'].reshape(-1, out['logits'].size(-1)), tgt.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
    opt.step()
    if step == 0: first = loss.item()
    if step % 100 == 0:
        print(f'Step {step:>3}: CE={loss.item():.4f} ({time.time()-t0:.0f}s)')

print(f'\nDone! CE: {first:.4f} -> {loss.item():.4f} ({(first-loss.item())/first*100:.1f}% drop)')
torch.save({'state_dict': model.state_dict(), 'step': 500}, 'checkpoints/model_final.pt')
print(f'Saved model_final.pt ({os.path.getsize(\"checkpoints/model_final.pt\")/1e9:.2f} GB)')
" > training_logs\final_train.log 2>&1
echo Training complete!
