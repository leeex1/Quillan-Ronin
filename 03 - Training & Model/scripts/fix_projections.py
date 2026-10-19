#!/usr/bin/env python3
"""Phase 1: Fix expert projections by directly copying source weights.

The 3 source models were sliced into Quillan's 34 experts.
The problem: dimension mismatches were handled with random SVD projections
that corrupted the signal. Fix: pad/trim source weights directly to target dims.
"""
import os, sys, torch, gc
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from safetensors import safe_open
from pathlib import Path

BASE = Path(r'C:\Users\Admin\Quillan-Ronin')
MODEL_DIR = BASE / 'Quillan-v4.2-model'
CKPT_PATH = BASE / 'checkpoints' / 'quillan_finetuned.pt'
OUT_PATH = BASE / 'checkpoints' / 'quillan_fixed.pt'
H, F = 2048, 4096

def dequantize_bitnet(w_uint8, w_scale):
    """BitNet stores uint8 with per-tensor scale. 
    Formula: w_fp = (uint8 - 128.0) * scale"""
    return (w_uint8.float() - 128.0) * w_scale.float()

def pad_to_target(w, target_in, target_out):
    """Pad or trim a weight to target dimensions.
    w: [src_in, src_out] in Quillan [in, out] convention
    Returns: [target_in, target_out]
    """
    w = w.float()
    in_dim, out_dim = w.shape
    
    # Pad or trim in dimension
    if in_dim < target_in:
        pad_in = target_in - in_dim
        zeros = torch.zeros(pad_in, out_dim)
        w = torch.cat([w, zeros], dim=0)
    elif in_dim > target_in:
        w = w[:target_in, :]
    
    # Pad or trim out dimension
    if out_dim < target_out:
        pad_out = target_out - out_dim
        zeros = torch.zeros(target_in, pad_out)
        w = torch.cat([w, zeros], dim=1)
    elif out_dim > target_out:
        w = w[:, :target_out]
    
    return w

# ─── Load source models ─────────────────────────────────────────────────
print('=== PHASE 1: FIX EXPERT WEIGHTS ===')
print('\n[1] Loading source models...')

# Llama (C0-C7): direct copies, no changes needed
# Qwen (C8-C21): 14 experts, hidden=1024, ffn=3584
# BitNet (C22-C33): 12 experts, hidden=2560, ffn=1728

# Qwen FFN weights
print('  Qwen FFN weights...')
q_w1, q_wg, q_w2 = [], [], []
with safe_open(str(MODEL_DIR / 'qwen model.safetensors'), framework='pt') as f:
    for i in range(14):
        q_w1.append(f.get_tensor(f'model.language_model.layers.{i}.mlp.gate_proj.weight').float())
        q_wg.append(f.get_tensor(f'model.language_model.layers.{i}.mlp.up_proj.weight').float())
        q_w2.append(f.get_tensor(f'model.language_model.layers.{i}.mlp.down_proj.weight').float())
q_w1 = torch.stack(q_w1)  # [14, 3584, 1024] — [out, in]
q_wg = torch.stack(q_wg)
q_w2 = torch.stack(q_w2)  # [14, 1024, 3584] — [out, in]
print(f'  Qwen w1: {list(q_w1.shape)}, w2: {list(q_w2.shape)}')

# BitNet FFN weights (dequantize from uint8)
print('  BitNet FFN weights...')
b_w1, b_wg, b_w2 = [], [], []
with safe_open(str(MODEL_DIR / 'bitnet model.safetensors'), framework='pt') as f:
    for i in range(12):
        w1_u = f.get_tensor(f'model.layers.{i}.mlp.gate_proj.weight')
        w1_s = f.get_tensor(f'model.layers.{i}.mlp.gate_proj.weight_scale')
        b_w1.append(dequantize_bitnet(w1_u, w1_s))
        
        wg_u = f.get_tensor(f'model.layers.{i}.mlp.up_proj.weight')
        wg_s = f.get_tensor(f'model.layers.{i}.mlp.up_proj.weight_scale')
        b_wg.append(dequantize_bitnet(wg_u, wg_s))
        
        w2_u = f.get_tensor(f'model.layers.{i}.mlp.down_proj.weight')
        w2_s = f.get_tensor(f'model.layers.{i}.mlp.down_proj.weight_scale')
        b_w2.append(dequantize_bitnet(w2_u, w2_s))
b_w1 = torch.stack(b_w1)  # [12, 1728, 2560] — [out, in]
b_wg = torch.stack(b_wg)
b_w2 = torch.stack(b_w2)  # [12, 2560, 1728] — [out, in]
print(f'  BitNet w1: {list(b_w1.shape)}, w2: {list(b_w2.shape)}')

# ─── Convert to Quillan format [experts, in_dim, out_dim] ─────────────
print('\n[2] Converting to Quillan format...')

# Qwen: [14, 3584, 1024] [out, in] → transpose → [14, 1024, 3584] [in, out] → pad → [14, 2048, 4096]
q_w1 = torch.stack([pad_to_target(w, H, F) for w in q_w1.transpose(1, 2)])  # [14, H, F]
q_wg = torch.stack([pad_to_target(w, H, F) for w in q_wg.transpose(1, 2)])
q_w2 = torch.stack([pad_to_target(w, F, H) for w in q_w2.transpose(1, 2)])  # down: [in=F, out=H] → pad to [4096, 2048]

# BitNet: [12, 1728, 2560] [out, in] → transpose → [12, 2560, 1728] [in, out] → trim/pad → [12, 2048, 4096]
# In this case, in_dim=2560 > target=2048, so we TRIM the first 2048 dims
b_w1 = torch.stack([pad_to_target(w, H, F) for w in b_w1.transpose(1, 2)])
b_wg = torch.stack([pad_to_target(w, H, F) for w in b_wg.transpose(1, 2)])
b_w2 = torch.stack([pad_to_target(w, F, H) for w in b_w2.transpose(1, 2)])  # down: [in=F, out=H]

print(f'  Qwen converted: w1={list(q_w1.shape)}')
print(f'  BitNet converted: w1={list(b_w1.shape)}')

# ─── Load checkpoint and replace experts ───────────────────────────────
print('\n[3] Loading checkpoint...')
ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
sd = ckpt.get('state_dict', ckpt)
if 'model_state_dict' in ckpt: sd = ckpt['model_state_dict']

print('  Replacing C8-C21 (Qwen)...')
sd['moe.w1'][8:22] = q_w1.to(sd['moe.w1'].dtype)
sd['moe.wgate'][8:22] = q_wg.to(sd['moe.wgate'].dtype)
sd['moe.w2'][8:22] = q_w2.to(sd['moe.w2'].dtype)

# BitNet weights have ~1000x larger magnitude than Llama. Rescale to match.
llama_norm = sd['moe.w1'][:8].norm()
bitnet_norm = b_w1.norm()
scale_factor = llama_norm / bitnet_norm
print(f'  Rescaling BitNet by {scale_factor:.4f} ({llama_norm:.1f} / {bitnet_norm:.1f})')
b_w1 = b_w1 * scale_factor
b_wg = b_wg * scale_factor
b_w2 = b_w2 * scale_factor

print('  Replacing C22-C33 (BitNet)...')
sd['moe.w1'][22:34] = b_w1.to(sd['moe.w1'].dtype)
sd['moe.wgate'][22:34] = b_wg.to(sd['moe.wgate'].dtype)
sd['moe.w2'][22:34] = b_w2.to(sd['moe.w2'].dtype)

del q_w1, q_wg, q_w2, b_w1, b_wg, b_w2; gc.collect()

# ─── Verify ─────────────────────────────────────────────────────────────
s1 = str(tuple(sd['moe.w1'].shape))
s2 = str(tuple(sd['moe.w2'].shape))
sg = str(tuple(sd['moe.wgate'].shape))
dtype = sd['moe.w1'].dtype
print(f'  w1: {s1} {dtype}')
print(f'  w2: {s2} {dtype}')
print(f'  wgate: {sg} {dtype}')

# Quick sanity: check norm of Llama vs Qwen vs BitNet experts
for name, start, end in [('Llama C0', 0, 8), ('Qwen C8', 8, 22), ('BitNet C22', 22, 34)]:
    n = sd['moe.w1'][start:end].norm().item()
    print(f'  {name}: w1_norm={n:.2f}')

# ─── Save ───────────────────────────────────────────────────────────────
print(f'\n[4] Saving to {OUT_PATH.name}...')
torch.save(dict(state_dict=sd, step=0, loss=0.0, version='v5.3.1-fixed'), OUT_PATH)
gb = os.path.getsize(OUT_PATH) / 1e9
print(f'  Saved: {gb:.2f} GB')
print('\nPHASE 1 COMPLETE')
