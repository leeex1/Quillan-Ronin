import torch, sys
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from transplant_weights import QuillanConfig, QuillanRoninSovereign

ckpt = torch.load('checkpoint_phase5.pt', weights_only=False, map_location='cpu')
cfg = ckpt['config']
model = QuillanRoninSovereign(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)

dl = model.diffusion[0]
print(f'norm1.weight: [{dl.norm1.weight.min():.4f}, {dl.norm1.weight.max():.4f}]')
print(f'q_proj.weight: [{dl.q_proj.weight.min():.4f}, {dl.q_proj.weight.max():.4f}]')
print(f'k_proj.weight: [{dl.k_proj.weight.min():.4f}, {dl.k_proj.weight.max():.4f}]')
print(f'v_proj.weight: [{dl.v_proj.weight.min():.4f}, {dl.v_proj.weight.max():.4f}]')
print(f'o_proj.weight: [{dl.o_proj.weight.min():.4f}, {dl.o_proj.weight.max():.4f}]')

# Check ffn
for name, p in dl.ffn.named_parameters():
    print(f'ffn.{name}: [{p.min():.4f}, {p.max():.4f}]')

# Check ALL diffusion layers
print('\nAll diffusion layers:')
for i, layer in enumerate(model.diffusion):
    print(f'  Layer {i}: q_proj range=[{layer.q_proj.weight.min():.4f}, {layer.q_proj.weight.max():.4f}]')
