import os
import sys
import torch
import torch.nn as nn
from safetensors import safe_open
from dataclasses import dataclass

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

BASE_DIR = r"c:\Users\Admin\Quillan-Ronin"
SOURCE_DIR = os.path.join(BASE_DIR, "Quillan-v4.2-model")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "quillan_transplanted.pt")
OUT_PATH = os.path.join(BASE_DIR, "checkpoints", "quillan_transplanted_v8.pt")

sys.path.insert(0, BASE_DIR)
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

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
    print("CLEAN WEIGHT TRANSPLANT FOR QUILLAN v8 (NO RANDOM PROJECTIONS)")
    print("=" * 60)
    
    print(f"Loading base checkpoint {CKPT_PATH} for structures...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    old_sd = ckpt['model_state_dict']
    
    cfg = QuillanArchConfig()
    model = QuillanRoninSovereign(cfg)
    new_sd = model.state_dict()
    
    # 1. Map non-expert weights (Embeddings, Finalizer, Decomposition)
    print("\n--- Mapping non-expert weights ---")
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
                
    # 2. Map diffusion layer from Llama 3.2 Layer 0 attention (unscaled, exact)
    print("\n--- Mapping Diffusion Attention from Llama 3.2 Layer 0 ---")
    llama_path = os.path.join(SOURCE_DIR, "llama model.safetensors")
    
    diffusion_llama = {
        'model.layers.0.input_layernorm.weight': 'diffusion_core.couil_attn.norm.weight',
        'model.layers.0.post_attention_layernorm.weight': 'diffusion_core.norm2.weight',
        'model.layers.0.self_attn.q_proj.weight': 'diffusion_core.couil_attn.q_proj.weight',
        'model.layers.0.self_attn.o_proj.weight': 'diffusion_core.couil_attn.o_proj.weight',
    }
    for old_k, new_k in diffusion_llama.items():
        w = load_source_tensor(llama_path, old_k)
        new_sd[new_k].copy_(w)
        print(f"  Direct copied Llama {old_k} -> {new_k}")
        
    # LayerNorm bias parameters are set to zero (RMSNorm has no bias)
    new_sd['diffusion_core.couil_attn.norm.bias'].zero_()
    new_sd['diffusion_core.norm2.bias'].zero_()
        
    # Map k_proj and v_proj (from [512, 2048] tiled 4x to [2048, 2048])
    for proj_name in ['k_proj', 'v_proj']:
        w = load_source_tensor(llama_path, f"model.layers.0.self_attn.{proj_name}.weight")
        # Tile 4x along output dimension
        w_tiled = w.repeat(4, 1)
        new_sd[f"diffusion_core.couil_attn.{proj_name}.weight"].copy_(w_tiled)
        print(f"  Tiled Llama model.layers.0.self_attn.{proj_name}.weight 4x -> diffusion_core.couil_attn.{proj_name}.weight")
        
    # Map diffusion FFN from Llama Layer 0 FFN sliced from 8192 to 4096
    # In diffusion layer, ffn is nn.Sequential(Linear(2048, 8192), GELU(), Linear(8192, 2048))
    new_sd['diffusion_core.ffn.0.weight'].copy_(load_source_tensor(llama_path, "model.layers.0.mlp.gate_proj.weight"))
    new_sd['diffusion_core.ffn.2.weight'].copy_(load_source_tensor(llama_path, "model.layers.0.mlp.down_proj.weight"))
    print("  Direct copied Llama Layer 0 mlp to diffusion_core.ffn")

    # 3. Map expert swarms
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
                elif name == 'clone_diversity':
                    new_val[:old_val.shape[0]].copy_(old_val)
                else:
                    new_val.copy_(old_val)
    print("  Mapped all expert swarms.")

    # 4. Llama Experts C0-C7 (SwiGLU, sliced FFN 8192->4096)
    print("\n--- Mapping Llama Experts C0-C7 (Sliced FFN, No Scaling) ---")
    for e in range(8):
        prefix = f"model.layers.{e}"
        gate = load_source_tensor(llama_path, f"{prefix}.mlp.gate_proj.weight")[:4096, :]
        up   = load_source_tensor(llama_path, f"{prefix}.mlp.up_proj.weight")[:4096, :]
        down = load_source_tensor(llama_path, f"{prefix}.mlp.down_proj.weight")[:, :4096]
        
        # Llama weights are [out_dim, in_dim], target moe expects [in_dim, out_dim] for w1/wgate
        new_sd['moe.w1'][e].copy_(gate.T)
        new_sd['moe.wgate'][e].copy_(up.T)
        new_sd['moe.w2'][e].copy_(down.T)
        print(f"  Expert C{e} (Llama Layer {e}) mapped.")

    # 5. Qwen Experts C8-C21 (SwiGLU, Zero-padded hidden 1024->2048 and FFN 3584->4096)
    print("\n--- Mapping Qwen Experts C8-C21 (Zero-padded, No Scaling) ---")
    qwen_path = os.path.join(SOURCE_DIR, "qwen model.safetensors")
    for e in range(14):
        layer_idx = e
        expert_idx = 8 + e
        lm_prefix = f"model.language_model.layers.{layer_idx}"
        
        # Qwen shapes: gate [3584, 1024], up [3584, 1024], down [1024, 3584]
        gate = load_source_tensor(qwen_path, f"{lm_prefix}.mlp.gate_proj.weight").float()
        up   = load_source_tensor(qwen_path, f"{lm_prefix}.mlp.up_proj.weight").float()
        down = load_source_tensor(qwen_path, f"{lm_prefix}.mlp.down_proj.weight").float()
        
        # Target w1 shape is [2048, 4096]. We place Qwen gate [1024, 3584] in the top-left corner
        w1 = torch.zeros((2048, 4096))
        w1[:1024, :3584] = gate.T
        
        wgate = torch.zeros((2048, 4096))
        wgate[:1024, :3584] = up.T
        
        w2 = torch.zeros((4096, 2048))
        w2[:3584, :1024] = down.T
        
        new_sd['moe.w1'][expert_idx].copy_(w1)
        new_sd['moe.wgate'][expert_idx].copy_(wgate)
        new_sd['moe.w2'][expert_idx].copy_(w2)
        print(f"  Expert C{expert_idx} (Qwen Layer {layer_idx}) zero-padded mapped.")

    # 6. BitNet Experts C22-C33 (SwiGLU, Sliced hidden 2560->2048 and FFN 6912->4096)
    print("\n--- Mapping BitNet Experts C22-C33 (Sliced, No Scaling) ---")
    bitnet_path = os.path.join(SOURCE_DIR, "bitnet model.safetensors")
    for e in range(12):
        layer_idx = e
        expert_idx = 22 + e
        prefix = f"model.layers.{layer_idx}"
        
        gate_packed = load_source_tensor(bitnet_path, f"{prefix}.mlp.gate_proj.weight")
        up_packed   = load_source_tensor(bitnet_path, f"{prefix}.mlp.up_proj.weight")
        down_packed = load_source_tensor(bitnet_path, f"{prefix}.mlp.down_proj.weight")
        
        gate_scale = load_source_tensor(bitnet_path, f"{prefix}.mlp.gate_proj.weight_scale").float().item()
        up_scale   = load_source_tensor(bitnet_path, f"{prefix}.mlp.up_proj.weight_scale").float().item()
        down_scale = load_source_tensor(bitnet_path, f"{prefix}.mlp.down_proj.weight_scale").float().item()
        
        # Unpack to float32
        gate = unpack_bitnet_weight(gate_packed, gate_scale, 6912, 2560) # [6912, 2560]
        up   = unpack_bitnet_weight(up_packed, up_scale, 6912, 2560)     # [6912, 2560]
        down = unpack_bitnet_weight(down_packed, down_scale, 2560, 6912) # [2560, 6912]
        
        # Slice hidden to 2048, FFN to 4096
        w1 = gate[:4096, :2048].T     # [2048, 4096]
        wgate = up[:4096, :2048].T   # [2048, 4096]
        w2 = down[:2048, :4096].T     # [4096, 2048]
        
        new_sd['moe.w1'][expert_idx].copy_(w1)
        new_sd['moe.wgate'][expert_idx].copy_(wgate)
        new_sd['moe.w2'][expert_idx].copy_(w2)
        print(f"  Expert C{expert_idx} (BitNet Layer {layer_idx}) sliced mapped.")

    print(f"\nSaving clean transplanted checkpoint to {OUT_PATH}...")
    torch.save({'model_state_dict': new_sd}, OUT_PATH)
    print("Clean transplant completed successfully!")

if __name__ == "__main__":
    main()
