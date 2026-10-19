import sys, torch
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ckpt = torch.load(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_sft_v3_best.pt", map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)

# Inspect txt_dec weights
for k in sorted(sd.keys()):
    if 'txt_dec' in k:
        v = sd[k]
        print(f"  {k}: shape={v.shape}  mean={v.mean().item():.6f}  std={v.std().item():.6f}  max={v.abs().max().item():.6f}")

# Check the LoRA scaling: 16.0 / sqrt(lora_B.shape[0])
lora_B = sd.get('txt_dec.lora_B')
if lora_B is not None:
    scaling = 16.0 / (lora_B.shape[0] ** 0.5)
    print(f"\n  LoRA scaling = 16.0 / sqrt({lora_B.shape[0]}) = {scaling:.4f}")
    
    lora_A = sd.get('txt_dec.lora_A')
    if lora_A is not None:
        # Simulate LoRA contribution magnitude
        # out += (x @ lora_A) @ lora_B * scaling
        # For unit-norm x, the LoRA output magnitude ≈ ||lora_A|| * ||lora_B|| * scaling
        print(f"  ||lora_A|| (Frobenius) = {lora_A.norm().item():.4f}")
        print(f"  ||lora_B|| (Frobenius) = {lora_B.norm().item():.4f}")
        product = lora_A.norm().item() * lora_B.norm().item() * scaling
        print(f"  Estimated LoRA output magnitude ≈ {product:.4f}")

# Also check the main weight
w = sd.get('txt_dec.weight')
if w is not None:
    print(f"\n  ||weight|| (Frobenius) = {w.norm().item():.4f}")
    print(f"  weight row norms: mean={w.norm(dim=1).mean().item():.4f}, max={w.norm(dim=1).max().item():.4f}")
