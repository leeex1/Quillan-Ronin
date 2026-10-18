import torch
import sys
from transplant_weights import QuillanConfig

def check():
    ckpt = torch.load('quillan_transplanted.pt', map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    print(f"txt_emb.weight: {sd['txt_emb.weight'].shape}")
    print(f"router.weight: {sd['router.weight'].shape}")
    print(f"experts.0.w1.weight: {sd['experts.0.w1.weight'].shape}")
    if 'diffusion.0.q_proj.weight' in sd:
        print(f"diffusion.0.q_proj.weight: {sd['diffusion.0.q_proj.weight'].shape}")

if __name__ == "__main__":
    check()
