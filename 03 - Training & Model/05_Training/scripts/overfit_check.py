import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math

# Import model definition from _dev
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('_dev'))
from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

# Custom tiny dataset
class TinyDataset(Dataset):
    def __init__(self, path, num_samples=20, seq=128):
        t = torch.load(path, weights_only=True, map_location='cpu').to(torch.long)
        self.t = t[:num_samples * (seq // 2) + seq + 1]
        self.seq = seq
        self.stride = seq // 2
        self.num_chunks = num_samples
        print(f"  Tiny Dataset initialized with {len(self.t)} tokens, {self.num_chunks} chunks.")
    def __len__(self):
        return self.num_chunks
    def __getitem__(self, i):
        start = i * self.stride
        return self.t[start:start+self.seq], self.t[start+1:start+self.seq+1]

def main():
    print("--- Running Overfitting Sanity Check ---")
    cfg = QuillanArchConfig()
    model = QuillanRoninSovereign(cfg)
    
    # Load weights
    ckpt_path = 'checkpoints/router_trained.pt'
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt)
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    model.load_state_dict(sd, strict=False)
    model.txt_dec.weight = model.ingestion.txt_emb.weight
    
    # Freeze 99% of model, unfreeze 1% (same as corrected train.py)
    for p in model.parameters(): p.requires_grad_(False)
    trainable_keys = ['moe.router.', 'quillan_finalizer', 'quillan_gate', 'pre_final_norm', 'lora', 'txt_dec']
    for n,p in model.named_parameters():
        if any(k in n for k in trainable_keys):
            p.requires_grad_(True)
            
    # Load 20 chunks from GPT_5.5_Distilled.pt
    ds = TinyDataset(r'C:\Users\Admin\Quillan-Ronin\training_data\GPT_5.5_Distilled.pt', num_samples=20, seq=128)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    
    # F16Adafactor optimizer
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
                        
    opt = F16Adafactor([p for p in model.parameters() if p.requires_grad], lr=5e-5) # Stable learning rate for overfitting
    
    model.train()
    step = 0
    t0 = time.time()
    ema = None
    
    STEPS = 150
    it = iter(loader)
    
    while step < STEPS:
        try: inp, tgt = next(it)
        except StopIteration:
            it = iter(loader)
            inp, tgt = next(it)
        
        out = model(inp)
        loss = F.cross_entropy(out['logits'].float().reshape(-1, 50257), tgt.reshape(-1))
        
        if torch.isnan(loss):
            opt.zero_grad()
            continue
            
        loss.backward()
        
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 50.0)
        if torch.isnan(gn):
            opt.zero_grad()
            continue
            
        opt.step()
        opt.zero_grad()
        step += 1
        
        ema = loss.item() if ema is None else 0.95*ema+0.05*loss.item()
        print(f"  step {step:>3}/{STEPS} | loss {loss.item():.4f} (ema: {ema:.4f}) | gn {gn:.2f} | {step/(time.time()-t0):.2f} st/s")
        
    print(f"\nFinished overfitting check. Final loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()
