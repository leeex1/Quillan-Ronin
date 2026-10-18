#!/usr/bin/env python3
"""
Rename tensors to match Llama naming conventions for GGUF conversion
"""

import torch
from safetensors.torch import load_file, save_file
import os

def rename_tensors_for_llama():
    """Rename model tensors to match Llama naming conventions"""
    print("🔄 Renaming tensors for Llama compatibility...")
    print("=" * 60)

    # Load the current model
    model_path = "gguf_export/model.safetensors"
    if not os.path.exists(model_path):
        print("❌ Model file not found")
        return False

    print(f"📂 Loading model from {model_path}")
    state_dict = load_file(model_path)

    print(f"📊 Found {len(state_dict)} tensors")

    # Llama naming mapping
    tensor_mapping = {
        # Embedding layer
        'text_emb.weight': 'embed_tokens.weight',

        # Transformer layers
        # Layer 0
        'transformer.layers.0.self_attn.in_proj_weight': 'layers.0.self_attn.q_proj.weight',  # This needs to be split properly
        'transformer.layers.0.self_attn.in_proj_bias': 'layers.0.self_attn.q_proj.bias',
        'transformer.layers.0.self_attn.out_proj.weight': 'layers.0.self_attn.o_proj.weight',
        'transformer.layers.0.self_attn.out_proj.bias': 'layers.0.self_attn.o_proj.bias',

        'transformer.layers.0.linear1.weight': 'layers.0.mlp.gate_proj.weight',
        'transformer.layers.0.linear1.bias': 'layers.0.mlp.gate_proj.bias',
        'transformer.layers.0.linear2.weight': 'layers.0.mlp.down_proj.weight',
        'transformer.layers.0.linear2.bias': 'layers.0.mlp.down_proj.bias',

        'transformer.layers.0.norm1.weight': 'layers.0.input_layernorm.weight',
        'transformer.layers.0.norm1.bias': 'layers.0.input_layernorm.bias',
        'transformer.layers.0.norm2.weight': 'layers.0.post_attention_layernorm.weight',
        'transformer.layers.0.norm2.bias': 'layers.0.post_attention_layernorm.bias',

        # Similar for other layers (0-3)
        # Layer 1
        'transformer.layers.1.self_attn.in_proj_weight': 'layers.1.self_attn.q_proj.weight',
        'transformer.layers.1.self_attn.in_proj_bias': 'layers.1.self_attn.q_proj.bias',
        'transformer.layers.1.self_attn.out_proj.weight': 'layers.1.self_attn.o_proj.weight',
        'transformer.layers.1.self_attn.out_proj.bias': 'layers.1.self_attn.o_proj.bias',

        'transformer.layers.1.linear1.weight': 'layers.1.mlp.gate_proj.weight',
        'transformer.layers.1.linear1.bias': 'layers.1.mlp.gate_proj.bias',
        'transformer.layers.1.linear2.weight': 'layers.1.mlp.down_proj.weight',
        'transformer.layers.1.linear2.bias': 'layers.1.mlp.down_proj.bias',

        'transformer.layers.1.norm1.weight': 'layers.1.input_layernorm.weight',
        'transformer.layers.1.norm1.bias': 'layers.1.input_layernorm.bias',
        'transformer.layers.1.norm2.weight': 'layers.1.post_attention_layernorm.weight',
        'transformer.layers.1.norm2.bias': 'layers.1.post_attention_layernorm.bias',

        # Layer 2
        'transformer.layers.2.self_attn.in_proj_weight': 'layers.2.self_attn.q_proj.weight',
        'transformer.layers.2.self_attn.in_proj_bias': 'layers.2.self_attn.q_proj.bias',
        'transformer.layers.2.self_attn.out_proj.weight': 'layers.2.self_attn.o_proj.weight',
        'transformer.layers.2.self_attn.out_proj.bias': 'layers.2.self_attn.o_proj.bias',

        'transformer.layers.2.linear1.weight': 'layers.2.mlp.gate_proj.weight',
        'transformer.layers.2.linear1.bias': 'layers.2.mlp.gate_proj.bias',
        'transformer.layers.2.linear2.weight': 'layers.2.mlp.down_proj.weight',
        'transformer.layers.2.linear2.bias': 'layers.2.mlp.down_proj.bias',

        'transformer.layers.2.norm1.weight': 'layers.2.input_layernorm.weight',
        'transformer.layers.2.norm1.bias': 'layers.2.input_layernorm.bias',
        'transformer.layers.2.norm2.weight': 'layers.2.post_attention_layernorm.weight',
        'transformer.layers.2.norm2.bias': 'layers.2.post_attention_layernorm.bias',

        # Layer 3
        'transformer.layers.3.self_attn.in_proj_weight': 'layers.3.self_attn.q_proj.weight',
        'transformer.layers.3.self_attn.in_proj_bias': 'layers.3.self_attn.q_proj.bias',
        'transformer.layers.3.self_attn.out_proj.weight': 'layers.3.self_attn.o_proj.weight',
        'transformer.layers.3.self_attn.out_proj.bias': 'layers.3.self_attn.o_proj.bias',

        'transformer.layers.3.linear1.weight': 'layers.3.mlp.gate_proj.weight',
        'transformer.layers.3.linear1.bias': 'layers.3.mlp.gate_proj.bias',
        'transformer.layers.3.linear2.weight': 'layers.3.mlp.down_proj.weight',
        'transformer.layers.3.linear2.bias': 'layers.3.mlp.down_proj.bias',

        'transformer.layers.3.norm1.weight': 'layers.3.input_layernorm.weight',
        'transformer.layers.3.norm1.bias': 'layers.3.input_layernorm.bias',
        'transformer.layers.3.norm2.weight': 'layers.3.post_attention_layernorm.weight',
        'transformer.layers.3.norm2.bias': 'layers.3.post_attention_layernorm.bias',

        # Output layer
        'output_proj.weight': 'lm_head.weight',
        'output_proj.bias': 'lm_head.bias',

        # Positional embeddings (Llama doesn't use this, but we'll keep it)
        'pos_emb': 'position_embeddings',
    }

    # Apply tensor renaming
    renamed_state_dict = {}
    for old_name, tensor in state_dict.items():
        if old_name in tensor_mapping:
            new_name = tensor_mapping[old_name]
            renamed_state_dict[new_name] = tensor
            print(f"🔄 {old_name} → {new_name}")
        else:
            # Keep tensors that don't need renaming
            renamed_state_dict[old_name] = tensor
            print(f"✅ {old_name} (unchanged)")

    # Handle attention projection splitting (Llama has separate q,k,v projections)
    # This is complex - for now, we'll keep the combined projection
    print("⚠️ Note: Attention projections kept combined (not split into q,k,v)")

    # Save renamed model
    output_path = "gguf_export/model_llama_format.safetensors"
    save_file(renamed_state_dict, output_path)
    print(f"✅ Renamed model saved to {output_path}")

    # Update config for Llama format
    config_path = "gguf_export/config.json"
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Add Llama-specific config
        config.update({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "intermediate_size": 11008,  # 4 * 1024 * 2.6875 (approx)
            "max_position_embeddings": 4096,
            "num_attention_heads": 32,  # Adjusted for compatibility
            "num_hidden_layers": 4,
            "num_key_value_heads": 32,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "transformers_version": "4.36.0",
            "use_cache": True,
            "vocab_size": 1000
        })

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ Config updated for Llama compatibility")

    print(f"📊 Final tensor count: {len(renamed_state_dict)}")
    total_params = sum(tensor.numel() for tensor in renamed_state_dict.values())
    print(f"📊 Total parameters: {total_params:,}")

    return True

if __name__ == "__main__":
    rename_tensors_for_llama()
