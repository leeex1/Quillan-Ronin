import sys, torch
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ckpt = torch.load(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_sft_v3_best.pt", map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)

# Check MoE shapes
for k in sorted(sd.keys()):
    if 'moe.w1' in k or 'moe.w2' in k or 'moe.wgate' in k or 'expert_swarms' in k:
        print(f"  {k}: {sd[k].shape}")
        break

# Just check w1
w1 = sd.get('moe.w1')
if w1 is not None:
    print(f"\n  moe.w1: {w1.shape}  (num_experts={w1.shape[0]}, hidden={w1.shape[1]}, ffn={w1.shape[2]})")
    
w2 = sd.get('moe.w2')
if w2 is not None:
    print(f"  moe.w2: {w2.shape}  (num_experts={w2.shape[0]}, ffn={w2.shape[1]}, hidden={w2.shape[2]})")

swarm_A = sd.get('moe.expert_swarms.0.A')
if swarm_A is not None:
    print(f"  swarm[0].A: {swarm_A.shape}  (ffn_dim={swarm_A.shape[0]}, rank={swarm_A.shape[1]})")
