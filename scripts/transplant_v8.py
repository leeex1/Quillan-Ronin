#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
from safetensors import safe_open

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Ensure absolute paths
BASE_DIR = r"c:\Users\Admin\Quillan-Ronin"
SOURCE_DIR = os.path.join(BASE_DIR, "Quillan-v4.2-model")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "quillan_transplanted.pt")
OUT_PATH = os.path.join(BASE_DIR, "checkpoints", "quillan_transplanted_v8.pt")

sys.path.insert(0, BASE_DIR)
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

@dataclass
class QuillanConfig:
    hidden_dim: int = 2048
    ffn_dim: int = 4096
    vocab_size: int = 50257
    num_experts: int = 34
    top_k: int = 4
    swarm_rank: int = 8
    swarm_instances: int = 8
    diffusion_layers: int = 6
    diffusion_heads: int = 8
    num_vectors: int = 9

class ProjectionMatrices(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen_hidden = nn.Linear(1024, 2048, bias=False)
        self.qwen_ffn_gate = nn.Linear(3584, 4096, bias=False)
        self.qwen_ffn_up = nn.Linear(3584, 4096, bias=False)
        self.qwen_ffn_down = nn.Linear(3584, 4096, bias=False)
        self.qwen_o_proj = nn.Linear(2048, 1024, bias=False)
        self.bitnet_hidden = nn.Linear(2560, 2048, bias=False)
        self.bitnet_ffn_gate = nn.Linear(1728, 4096, bias=False)
        self.bitnet_ffn_down = nn.Linear(4096, 6912, bias=False)
        self.bitnet_down_output = nn.Linear(640, 2048, bias=False)

def load_source_tensor(path: str, key: str) -> torch.Tensor:
    with safe_open(path, framework='pt') as f:
        return f.get_tensor(key)

def unpack_bitnet_weight(packed_weight, scale, out_features, in_features):
    w = torch.zeros((out_features, in_features), dtype=torch.float32)
    w[0::4] = ((packed_weight >> 6) & 0x3).float() - 1.0
    w[1::4] = ((packed_weight >> 4) & 0x3).float() - 1.0
    w[2::4] = ((packed_weight >> 2) & 0x3).float() - 1.0
    w[3::4] = ((packed_weight >> 0) & 0x3).float() - 1.0
    return w * scale

def main():
    print("=" * 60)
    print("RE-TRANSPLANTING SwiGLU EXPERT WEIGHTS FOR QUILLAN v8")
    print("=" * 60)
    
    # 1. Load the original transplant file to get projections
    print(f"Loading {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    
    proj = ProjectionMatrices()
    proj.load_state_dict(ckpt['proj_state_dict'])
    print("Loaded ProjectionMatrices successfully.")
    
    # 2. Instantiate v8 Model
    cfg = QuillanArchConfig()
    model = QuillanRoninSovereign(cfg)
    
    # Initialize the target weights
    new_sd = model.state_dict()
    old_sd = ckpt['model_state_dict']
    
    # 3. Map non-expert weights from old state dict
    print("\n--- Mapping non-expert weights ---")
    
    # Basic mapping mappings
    basic_mappings = {
        'txt_emb.weight': 'ingestion.txt_emb.weight',
        'mod_emb.weight': 'ingestion.mod_emb.weight',
        'router.weight': 'moe.router.weight',
        'router.bias': 'moe.router.bias',
        'quillan_finalizer.weight': 'quillan_finalizer.weight',
        'quillan_finalizer.bias': 'quillan_finalizer.bias',
        'txt_dec.weight': 'txt_dec.weight',
    }
    
    for old_k, new_k in basic_mappings.items():
        if old_k in old_sd and new_k in new_sd:
            new_sd[new_k].copy_(old_sd[old_k])
            print(f"  Mapped {old_k} -> {new_k}")
            
    # Map decomposition vectors
    for k in old_sd.keys():
        if k.startswith('decomposition.'):
            if k in new_sd:
                new_sd[k].copy_(old_sd[k])
                print(f"  Mapped {k}")
            
    # Map diffusion layer 0 to diffusion_core
    diffusion_mappings = {
        'diffusion.0.q_proj.weight': 'diffusion_core.q_proj.weight',
        'diffusion.0.k_proj.weight': 'diffusion_core.k_proj.weight',
        'diffusion.0.v_proj.weight': 'diffusion_core.v_proj.weight',
        'diffusion.0.o_proj.weight': 'diffusion_core.o_proj.weight',
        'diffusion.0.norm1.weight': 'diffusion_core.norm1.weight',
        'diffusion.0.norm1.bias': 'diffusion_core.norm1.bias',
        'diffusion.0.norm2.weight': 'diffusion_core.norm2.weight',
        'diffusion.0.norm2.bias': 'diffusion_core.norm2.bias',
        'diffusion.0.ffn.0.weight': 'diffusion_core.ffn.0.weight',
        'diffusion.0.ffn.2.weight': 'diffusion_core.ffn.2.weight',
    }
    for old_k, new_k in diffusion_mappings.items():
        if old_k in old_sd and new_k in new_sd:
            new_sd[new_k].copy_(old_sd[old_k])
            print(f"  Mapped {old_k} -> {new_k}")

    # Normalize diffusion core weights to prevent exponential blowup in sequential loops
    for k in ['diffusion_core.q_proj.weight', 'diffusion_core.k_proj.weight', 
              'diffusion_core.v_proj.weight', 'diffusion_core.o_proj.weight']:
        if k in new_sd:
            w = new_sd[k]
            new_sd[k].copy_(w * (0.02 / w.std()))
            print(f"  Normalized scale of {k} to std 0.02")
            
    for k in ['diffusion_core.norm1.weight', 'diffusion_core.norm2.weight']:
        if k in new_sd:
            w = new_sd[k]
            # Center LayerNorm weights around 1.0 with std 0.1
            w_normalized = (w - w.mean()) * (0.1 / w.std()) + 1.0
            new_sd[k].copy_(w_normalized)
            print(f"  Normalized scale of {k} to mean 1.0, std 0.1")
            
    # Map expert swarms
    for e in range(34):
        old_prefix = f"experts.{e}.swarm"
        new_prefix = f"moe.expert_swarms.{e}"
        for name in ['A', 'B', 'clone_diversity', 'clone_coupling', 'population_mean', 'population_std']:
            old_k = f"{old_prefix}.{name}"
            new_k = f"{new_prefix}.{name}"
            if old_k in old_sd and new_k in new_sd:
                old_val = old_sd[old_k]
                new_val = new_sd[new_k]
                if name == 'A':
                    new_val[:, :old_val.shape[1]].copy_(old_val)
                elif name == 'B':
                    new_val[:old_val.shape[0], :].copy_(old_val)
                else:
                    new_val.copy_(old_val)
    print("  Mapped all expert swarms.")

    # 4. Correctly transplant SwiGLU weights for Llama experts C0-C7
    print("\n--- Transplanting Llama Experts C0-C7 (SwiGLU) ---")
    llama_path = os.path.join(SOURCE_DIR, "llama model.safetensors")
    for e in range(8):
        prefix = f"model.layers.{e}"
        # Shape in Llama is [8192, 2048]
        gate = load_source_tensor(llama_path, f"{prefix}.mlp.gate_proj.weight")[:4096, :]
        up   = load_source_tensor(llama_path, f"{prefix}.mlp.up_proj.weight")[:4096, :]
        down = load_source_tensor(llama_path, f"{prefix}.mlp.down_proj.weight")[:, :4096]
        
        # Scale to target standard deviations (0.03 for w1/wgate, 0.02 for w2)
        gate_scaled = gate.T * (0.03 / gate.std())
        up_scaled = up.T * (0.03 / up.std())
        down_scaled = down.T * (0.02 / down.std())
        
        new_sd['moe.w1'][e].copy_(gate_scaled)
        new_sd['moe.wgate'][e].copy_(up_scaled)
        new_sd['moe.w2'][e].copy_(down_scaled)
        print(f"  Llama Layer {e} -> Expert {e} (C{e}) done (Scaled to target stds).")

    # 5. Correctly transplant SwiGLU weights for Qwen experts C8-C21
    print("\n--- Transplanting Qwen Experts C8-C21 (SwiGLU) ---")
    qwen_path = os.path.join(SOURCE_DIR, "qwen model.safetensors")
    for e in range(14):
        layer_idx = e
        expert_idx = 8 + e
        lm_prefix = f"model.language_model.layers.{layer_idx}"
        
        # Qwen weights [3584, 1024]
        gate = load_source_tensor(qwen_path, f"{lm_prefix}.mlp.gate_proj.weight").float()
        up   = load_source_tensor(qwen_path, f"{lm_prefix}.mlp.up_proj.weight").float()
        down = load_source_tensor(qwen_path, f"{lm_prefix}.mlp.down_proj.weight").float()
        
        # Project hidden: 1024 -> 2048
        gate_proj = gate @ proj.qwen_hidden.weight.T.float() # [3584, 2048]
        up_proj   = up @ proj.qwen_hidden.weight.T.float()   # [3584, 2048]
        
        # Project FFN: 3584 -> 4096
        # proj.qwen_ffn_gate is Linear(3584, 4096)
        gate_final = proj.qwen_ffn_gate(gate_proj.T) # [2048, 4096]
        up_final   = proj.qwen_ffn_up(up_proj.T)     # [2048, 4096]
        
        # Project down: [1024, 3584] -> [2048, 4096]
        down_step1 = proj.qwen_hidden.weight @ down # [2048, 3584]
        down_step2 = down_step1 @ proj.qwen_ffn_down.weight.T # [2048, 4096]

        # Scale to target standard deviations (0.03 for w1/wgate, 0.02 for w2)
        gate_final_scaled = gate_final * (0.03 / gate_final.std())
        up_final_scaled = up_final * (0.03 / up_final.std())
        down_final_scaled = down_step2.T * (0.02 / down_step2.std())
        
        # Copy to moe weights
        new_sd['moe.w1'][expert_idx].copy_(gate_final_scaled)
        new_sd['moe.wgate'][expert_idx].copy_(up_final_scaled)
        new_sd['moe.w2'][expert_idx].copy_(down_final_scaled)
        print(f"  Qwen Layer {layer_idx} -> Expert {expert_idx} (C{expert_idx}) done (Scaled to target stds).")

    # 6. Correctly transplant SwiGLU weights for BitNet experts C22-C33
    print("\n--- Transplanting BitNet Experts C22-C33 (SwiGLU) ---")
    bitnet_path = os.path.join(SOURCE_DIR, "bitnet model.safetensors")
    for e in range(12):
        layer_idx = e
        expert_idx = 22 + e
        prefix = f"model.layers.{layer_idx}"
        
        # BitNet weights are uint8 packed (unpack first)
        gate_packed = load_source_tensor(bitnet_path, f"{prefix}.mlp.gate_proj.weight")
        up_packed   = load_source_tensor(bitnet_path, f"{prefix}.mlp.up_proj.weight")
        down_packed = load_source_tensor(bitnet_path, f"{prefix}.mlp.down_proj.weight")
        
        gate_scale = load_source_tensor(bitnet_path, f"{prefix}.mlp.gate_proj.weight_scale").float().item()
        up_scale   = load_source_tensor(bitnet_path, f"{prefix}.mlp.up_proj.weight_scale").float().item()
        down_scale = load_source_tensor(bitnet_path, f"{prefix}.mlp.down_proj.weight_scale").float().item()
        
        gate = unpack_bitnet_weight(gate_packed, gate_scale, 6912, 2560)
        up   = unpack_bitnet_weight(up_packed, up_scale, 6912, 2560)
        down = unpack_bitnet_weight(down_packed, down_scale, 2560, 6912)
        
        # Slice FFN dimension from 6912 to 4096 (directly slices weights)
        gate_sliced = gate[:4096, :]
        up_sliced   = up[:4096, :]
        down_sliced = down[:, :4096]
        
        # Project hidden dimension using proj.bitnet_hidden
        proj_weight = proj.bitnet_hidden.weight.float()
        gate_final = gate_sliced @ proj_weight.T
        up_final   = up_sliced @ proj_weight.T
        down_final = (proj_weight @ down_sliced).T
        
        # Scale to target standard deviations (0.03 for w1/wgate, 0.02 for w2)
        gate_final_scaled = gate_final * (0.03 / gate_final.std())
        up_final_scaled = up_final * (0.03 / up_final.std())
        down_final_scaled = down_final * (0.02 / down_final.std())
        
        new_sd['moe.w1'][expert_idx].copy_(gate_final_scaled.T)
        new_sd['moe.wgate'][expert_idx].copy_(up_final_scaled.T)
        new_sd['moe.w2'][expert_idx].copy_(down_final_scaled)
        print(f"  BitNet Layer {layer_idx} -> Expert {expert_idx} (C{expert_idx}) done (Unpacked, Projected, Sliced, Scaled).")

    # Save checkpoint
    print(f"\nSaving correct checkpoint to {OUT_PATH}...")
    torch.save({'model_state_dict': new_sd}, OUT_PATH)
    print("Re-transplant completed successfully!")

if __name__ == "__main__":
    main()
