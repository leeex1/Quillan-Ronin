#!/usr/bin/env python3
"""Minimal training — ternaries everything, fits 4GB VRAM, packed checkpoints."""
import os, sys, time, gc, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin\_dev')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

def pack_ternary(w):
    w_int = (w.round().clamp(-1, 1) + 1).long()
    f = w_int.reshape(-1)
    p = (4 - f.numel() % 4) % 4
    if p:
        f = torch.cat([f, f.new_zeros(p)])
    return (f[0::4]|(f[1::4]<<2)|(f[2::4]<<4)|(f[3::4]<<6)).to(torch.uint8)

def unpack_ternary(packed, shape):
    flat=packed.to(torch.uint8); v=torch.stack([(flat>>0)&3,(flat>>2)&3,(flat>>4)&3,(flat>>6)&3],-1).reshape(-1)
    r=torch.where(v==0,-1,torch.where(v==2,1,0))
    return r[:torch.Size(shape).numel()].reshape(shape).half()


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
    def __getitem__(self,i):
        start = i * self.stride
        c = self.t[start : start + self.seq + 1]
        return c[:-1], c[1:]

import json
def save_packed(model, path, step, loss):
    state = model.state_dict()
    packed = {}
    for k, v in state.items():
        if v.dim() >= 2 and 'norm' not in k and 'bias' not in k:
            try:
                s = v.abs().mean().clamp(min=0.01)
                w_q = torch.round(torch.clamp(v.float()/s, -1, 1))
                packed[k] = pack_ternary(w_q)
            except: packed[k] = v.half().contiguous()
        else: packed[k] = v.half().contiguous()
    torch.save(dict(state_dict=packed, step=step, loss=loss, packed=True, version='v5.3.1'), path)
    print(f'  Saved {Path(path).name} ({os.path.getsize(path)/1e9:.2f}GB packed)')

def main():
    STEPS = int(sys.argv[1]) if len(sys.argv)>1 else 200
    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    print(f'QUILLAN TRAIN ({STEPS} steps) — {device}')
    sys.stdout.flush()
    torch.backends.cudnn.benchmark = True

    model = QuillanRoninSovereign(QuillanArchConfig(device=device))
    if device == 'cuda':
        model = model.half()
    ckpt_path = 'checkpoints/quillan_finetuned.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/router_trained.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/quillan_fixed.pt'
        
    print(f"  Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    if ckpt.get('packed'):
        print('  Unpacking ternary checkpoint...')
        for k,v in sd.items():
            if v.dtype==torch.uint8: sd[k] = unpack_ternary(v, model.state_dict()[k].shape)
    elif 'model_state_dict' in ckpt: sd = ckpt['model_state_dict']
    
    # Dynamically filter out weights whose shapes don't match the newly configured ranks
    valid_sd = {}
    model_state = model.state_dict()
    for k, v in sd.items():
        if k in model_state and v.shape == model_state[k].shape:
            valid_sd[k] = v
        else:
            print(f"  Ignoring shape mismatch for {k}: checkpoint {v.shape} vs model {model_state.get(k, torch.tensor([])).shape}")
    sd = valid_sd
    
    model.load_state_dict(sd, strict=False)
    model.txt_dec.weight = model.ingestion.txt_emb.weight

    for p in model.parameters(): p.requires_grad_(False)
    trainable_keys = ['moe.router.', 'quillan_finalizer', 'quillan_gate', 'pre_final_norm', 'lora', 'txt_dec', 'expert_swarms']
    for n,p in model.named_parameters():
        if any(k in n for k in trainable_keys): p.requires_grad_(True)
    model.to(device)

    # Load only the curated, high-quality datasets for fast CPU convergence
    data_dir = r'C:\Users\Admin\Quillan-Ronin\training_data'
    paths = [
        os.path.join(data_dir, 'quillan_12mb_training_dataset.pt'),
        os.path.join(data_dir, 'train.pt')
    ]
    ds = TokDataset(paths, seq=128)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    print(f'  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M | Data: {len(ds)} chunks')
    print(f'  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB')

    class F16Adafactor(torch.optim.Optimizer):
        def __init__(self, p, lr=1e-3, wd=0.01):
            super().__init__(p, dict(lr=lr, wd=wd))
        def step(self, c=None):
            for g in self.param_groups:
                for p in g['params']:
                    if p.grad is None: continue
                    s=self.state[p]
                    if not s: s['sq']=torch.zeros_like(p.data,dtype=p.dtype)
                    s['sq'].mul_(0.999).addcmul_(p.grad,p.grad,value=0.001)
                    p.data.addcdiv_(p.grad.to(p.dtype), s['sq'].sqrt().add_(1e-3), value=-g['lr'])
                    if g['wd']>0: p.data.add_(p.data, alpha=-g['wd']*g['lr'])

    base_params = []
    lora_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'lora_' in n or 'expert_swarms' in n:
            lora_params.append(p)
        else:
            base_params.append(p)
    opt = F16Adafactor([
        {'params': base_params, 'lr': 1e-5},
        {'params': lora_params, 'lr': 1e-4}
    ], lr=1e-5)
    model.train(); step=micro_step=0; t0=time.time(); ema=None
    
    accum_steps = 2
    it = iter(loader)

    while step < STEPS:
        try: inp,tgt = next(it)
        except StopIteration: it=iter(loader); inp,tgt=next(it)
        inp=inp.to(device); tgt=tgt.to(device)
        out=model(inp)
        loss_ce = F.cross_entropy(out['logits'].float().reshape(-1,50257), tgt.reshape(-1))
        routing_loss = out.get('routing_loss', torch.tensor(0.0, device=device))
        ccrl_loss = out.get('ccrl_loss', torch.tensor(0.0, device=device))
        loss = loss_ce + routing_loss + ccrl_loss
        if torch.isnan(loss): opt.zero_grad(); continue
        (loss/accum_steps).backward(); micro_step+=1
        ema = loss.item() if ema is None else 0.95*ema+0.05*loss.item()
        if micro_step%accum_steps==0:
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if torch.isnan(gn): opt.zero_grad(); continue
            opt.step(); opt.zero_grad(); step+=1
            if step%1==0:
                v = f' | VRAM:{torch.cuda.memory_allocated()/1024**2:.0f}MB' if device=='cuda' else ''
                print(f'  step {step:>4}/{STEPS} | loss {ema:.4f} (ce: {loss_ce.item():.4f}, rtg: {routing_loss.item():.4f}, ccrl: {ccrl_loss.item():.4f}) | gn {gn:.2f} | {step/max(time.time()-t0,1e-6):.2f} st/s{v}')
            if step > 0 and step % 200 == 0:
                torch.save(model.state_dict(), f'checkpoints/quillan_step_{step}.pt')
            if step>=STEPS: break

    print(f'\nDONE — {step} steps, final loss {ema:.4f}')
    torch.save(model.state_dict(), 'checkpoints/quillan_finetuned.pt')
    print(f'Peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB' if device=='cuda' else '')

if __name__ == '__main__':
    main()
