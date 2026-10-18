import torch, sys
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from transplant_weights import QuillanConfig, QuillanRoninSovereign
import torch.nn.functional as F

device = torch.device('cuda')
ckpt = torch.load('checkpoint_phase5.pt', weights_only=False, map_location='cpu')
cfg = ckpt['config']
model = QuillanRoninSovereign(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
del ckpt

model = model.to(device)
model.eval()

x = torch.randint(0, 50257, (1, 16), device=device)

with torch.no_grad():
    # Embedding
    emb = model.txt_emb(x)
    print(f'1. Embedding: [{emb.min():.4f}, {emb.max():.4f}]')

    # Decomposition
    decomp = model.decomposition(emb)
    print(f'2. Decomposition: [{decomp.min():.4f}, {decomp.max():.4f}]')

    # Router
    flat_x = decomp.view(-1, cfg.hidden_dim)
    logits = model.router(flat_x)
    probs = F.softmax(logits, dim=-1)
    topk_p, topk_idx = torch.topk(probs, k=cfg.top_k, dim=-1)
    print(f'3. Router probs: [{probs.min():.4f}, {probs.max():.4f}]')
    print(f'   Top-4 probs sum: {topk_p.sum(dim=-1).mean():.4f}')

    # Expert computation
    flat_out = torch.zeros_like(flat_x)
    for k in range(cfg.top_k):
        idx = topk_idx[:, k]
        weight = topk_p[:, k].unsqueeze(-1)
        for e in range(cfg.num_experts):
            mask = (idx == e)
            if mask.any():
                expert_out = model.experts[e](flat_x[mask])
                print(f'4. Expert {e} output: [{expert_out.min():.4f}, {expert_out.max():.4f}] mask_sum={mask.sum()}')
                flat_out[mask] += expert_out * weight[mask]
                if e >= 2:
                    break
        if k >= 0:
            break

    x = flat_out.view(1, 16, cfg.hidden_dim)
    print(f'5. After experts: [{x.min():.4f}, {x.max():.4f}]')

    # Diffusion
    for layer in model.diffusion:
        x = layer(x)
    print(f'6. After diffusion: [{x.min():.4f}, {x.max():.4f}]')

    # Finalizer
    x = model.quillan_finalizer(x)
    print(f'7. After finalizer: [{x.min():.4f}, {x.max():.4f}]')

    # Decode
    logits = model.txt_dec(x)
    print(f'8. Final logits: [{logits.min():.4f}, {logits.max():.4f}]')
