import torch, sys
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from transplant_weights import QuillanConfig, QuillanRoninSovereign

ckpt = torch.load('checkpoint_phase5.pt', weights_only=False, map_location='cpu')
cfg = ckpt['config']
model = QuillanRoninSovereign(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)

# Check txt_dec vs txt_emb
print(f'txt_emb.weight: shape={model.txt_emb.weight.shape} range=[{model.txt_emb.weight.min():.4f}, {model.txt_emb.weight.max():.4f}]')
print(f'txt_dec.weight: shape={model.txt_dec.weight.shape} range=[{model.txt_dec.weight.min():.4f}, {model.txt_dec.weight.max():.4f}]')
print(f'Are they same data_ptr? {model.txt_emb.weight.data_ptr() == model.txt_dec.weight.data_ptr()}')

# Check router
print(f'router.weight: shape={model.router.weight.shape} range=[{model.router.weight.min():.4f}, {model.router.weight.max():.4f}]')

# Check decomposition
for name, p in model.named_parameters():
    if 'decomposition' in name:
        print(f'{name}: shape={p.shape} range=[{p.min():.4f}, {p.max():.4f}]')
        break

# Check quillan_finalizer
print(f'quillan_finalizer.weight: shape={model.quillan_finalizer.weight.shape} range=[{model.quillan_finalizer.weight.min():.4f}, {model.quillan_finalizer.weight.max():.4f}]')
