#!/usr/bin/env python3
"""
Phase 1: Analysis & Mapping for Quillan-Ronin Transplant
Analyzes source model tensors and builds complete mapping table.
"""
import os
import json
import logging
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("quillan_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Phase1")

BASE = r"C:\Users\Admin\Quillan-Ronin\Quillan-v4.2-model"

# Target architecture
TARGET = {
    "hidden_dim": 2048,
    "ffn_dim": 4096,
    "vocab_size": 50257,
    "num_experts": 34,
    "top_k": 4,
    "swarm_rank": 8,
    "swarm_instances": 8,
    "diffusion_layers": 6,
    "diffusion_heads": 8,
    "num_vectors": 9,
}


def analyze_llama():
    """Analyze Llama 3.2 model tensors."""
    logger.info("=== ANALYZING LLAMA 3.2 ===")
    path = os.path.join(BASE, "llama model.safetensors")
    
    with safe_open(path, framework='pt') as f:
        keys = sorted(f.keys())
        
        # Group by layer
        layers = {}
        for k in keys:
            if 'layers.' in k:
                parts = k.split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i+1 < len(parts):
                        try:
                            layer_num = int(parts[i+1])
                            if layer_num not in layers:
                                layers[layer_num] = {}
                            tensor_key = '.'.join(parts[i+2:])
                            layers[layer_num][tensor_key] = list(f.get_tensor(k).shape)
                        except ValueError:
                            pass
        
        # Get embedding
        emb_shape = list(f.get_tensor("model.embed_tokens.weight").shape)
        
        logger.info(f"  Layers: {len(layers)} (0-{max(layers.keys())})")
        logger.info(f"  Embedding: {emb_shape}")
        logger.info(f"  Layer 0 keys: {list(layers[0].keys()) if layers else 'none'}")
        
        # Print layer 0 details
        if 0 in layers:
            for k, v in sorted(layers[0].items()):
                logger.info(f"    {k}: {v}")
        
        return {
            "layers": layers,
            "embedding": emb_shape,
            "num_layers": len(layers),
        }


def analyze_qwen():
    """Analyze Qwen model tensors."""
    logger.info("=== ANALYZING QWEN ===")
    path = os.path.join(BASE, "qwen model.safetensors")
    
    with safe_open(path, framework='pt') as f:
        keys = sorted(f.keys())
        
        # Group by layer
        layers = {}
        for k in keys:
            if 'layers.' in k:
                parts = k.split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i+1 < len(parts):
                        try:
                            layer_num = int(parts[i+1])
                            if layer_num not in layers:
                                layers[layer_num] = {}
                            tensor_key = '.'.join(parts[i+2:])
                            layers[layer_num][tensor_key] = list(f.get_tensor(k).shape)
                        except ValueError:
                            pass
        
        # Get embedding
        emb_shape = list(f.get_tensor("model.language_model.embed_tokens.weight").shape)
        
        logger.info(f"  Layers: {len(layers)} (0-{max(layers.keys()) if layers else 0})")
        logger.info(f"  Embedding: {emb_shape}")
        
        # Print layer 0 details
        if 0 in layers:
            for k, v in sorted(layers[0].items()):
                logger.info(f"    {k}: {v}")
        
        return {
            "layers": layers,
            "embedding": emb_shape,
            "num_layers": len(layers),
        }


def analyze_bitnet():
    """Analyze BitNet model tensors."""
    logger.info("=== ANALYZING BITNET ===")
    path = os.path.join(BASE, "bitnet model.safetensors")
    
    with safe_open(path, framework='pt') as f:
        keys = sorted(f.keys())
        
        # Group by layer
        layers = {}
        for k in keys:
            if 'layers.' in k:
                parts = k.split('.')
                for i, p in enumerate(parts):
                    if p == 'layers' and i+1 < len(parts):
                        try:
                            layer_num = int(parts[i+1])
                            if layer_num not in layers:
                                layers[layer_num] = {}
                            tensor_key = '.'.join(parts[i+2:])
                            layers[layer_num][tensor_key] = list(f.get_tensor(k).shape)
                        except ValueError:
                            pass
        
        # Get embedding
        emb_shape = list(f.get_tensor("model.embed_tokens.weight").shape)
        
        logger.info(f"  Layers: {len(layers)} (0-{max(layers.keys()) if layers else 0})")
        logger.info(f"  Embedding: {emb_shape}")
        
        # Print layer 0 details
        if 0 in layers:
            for k, v in sorted(layers[0].items()):
                logger.info(f"    {k}: {v}")
        
        return {
            "layers": layers,
            "embedding": emb_shape,
            "num_layers": len(layers),
        }


def build_mapping(llama, qwen, bitnet):
    """Build complete source→target mapping."""
    logger.info("=== BUILDING MAPPING TABLE ===")
    
    mapping = {
        "target": TARGET,
        "source_models": {
            "llama": {
                "layers_available": llama["num_layers"],
                "experts_assigned": 8,
                "embedding": llama["embedding"],
                "note": "Take layers 0-7 (best attention weights)"
            },
            "qwen": {
                "layers_available": qwen["num_layers"],
                "experts_assigned": 14,
                "embedding": qwen["embedding"],
                "note": "Take layers 0-13 (best linear attention)"
            },
            "bitnet": {
                "layers_available": bitnet["num_layers"],
                "experts_assigned": 12,
                "embedding": bitnet["embedding"],
                "note": "Take layers 0-11 (best ternary weights)"
            },
        },
        "projections_needed": {
            "qwen_hidden": "Linear(1024, 2048, bias=False)",
            "bitnet_hidden": "Linear(2560, 2048, bias=False)",
            "qwen_ffn": "Linear(qwen_ffn_dim, 4096, bias=False)",
            "bitnet_ffn": "Linear(6912, 4096, bias=False)",
        },
        "expert_mapping": [],
    }
    
    # Build expert mapping
    expert_id = 0
    
    # Llama experts (C0-C7)
    for i in range(8):
        if i in llama["layers"]:
            layer = llama["layers"][i]
            mapping["expert_mapping"].append({
                "expert_id": expert_id,
                "council_member": ["ASTRA","VIR","SOLACE","PRAXIS","ECHO","OMNIS","LOGOS","METASYNTH"][i],
                "source": "llama",
                "source_layer": i,
                "projection": "none",
                "attn_shapes": {
                    "q_proj": layer.get("self_attn.q_proj.weight"),
                    "k_proj": layer.get("self_attn.k_proj.weight"),
                    "v_proj": layer.get("self_attn.v_proj.weight"),
                    "o_proj": layer.get("self_attn.o_proj.weight"),
                },
                "ffn_shapes": {
                    "gate_proj": layer.get("mlp.gate_proj.weight"),
                    "up_proj": layer.get("mlp.up_proj.weight"),
                    "down_proj": layer.get("mlp.down_proj.weight"),
                },
                "target_ffn": f"[{TARGET['ffn_dim']}, {TARGET['hidden_dim']}]",
                "operation": "slice 8192->4096"
            })
            expert_id += 1
    
    # Qwen experts (C8-C21)
    for i in range(14):
        if i in qwen["layers"]:
            layer = qwen["layers"][i]
            mapping["expert_mapping"].append({
                "expert_id": expert_id,
                "council_member": ["AETHER","CODEWEAVER","HARMONIA","SOPHIAE","WARDEN","KAIDO","LUMINARIS","VOXUM","NULLION","SHEPHERD","VIGIL","ARTIFEX","ARCHON","AURELION"][i],
                "source": "qwen",
                "source_layer": i,
                "projection": "qwen_hidden(1024->2048) + qwen_ffn",
                "ffn_shapes": {
                    "note": "Qwen uses linear_attn + mlp hybrid"
                },
                "target_ffn": f"[{TARGET['ffn_dim']}, {TARGET['hidden_dim']}]",
                "operation": "project 1024->2048, then map FFN"
            })
            expert_id += 1
    
    # BitNet experts (C22-C33)
    for i in range(12):
        if i in bitnet["layers"]:
            layer = bitnet["layers"][i]
            mapping["expert_mapping"].append({
                "expert_id": expert_id,
                "council_member": ["CADENCE","SCHEMA","PROMETHEUS","TECHNE","CHRONICLE","CALCULUS","NAVIGATOR","TESSERACT","NEXUS","AEON","TYPIST","PREDATOR"][i],
                "source": "bitnet",
                "source_layer": i,
                "projection": "bitnet_hidden(2560->2048) + bitnet_ffn(6912->4096)",
                "ffn_shapes": {
                    "gate_proj": layer.get("mlp.gate_proj.weight"),
                    "up_proj": layer.get("mlp.up_proj.weight"),
                    "down_proj": layer.get("mlp.down_proj.weight"),
                },
                "target_ffn": f"[{TARGET['ffn_dim']}, {TARGET['hidden_dim']}]",
                "operation": "project 2560->2048, slice 6912->4096"
            })
            expert_id += 1
    
    logger.info(f"  Total experts mapped: {len(mapping['expert_mapping'])}")
    logger.info(f"  Llama: 8 experts (layers 0-7)")
    logger.info(f"  Qwen: 14 experts (layers 0-13)")
    logger.info(f"  BitNet: 12 experts (layers 0-11)")
    
    return mapping


def main():
    logger.info("=" * 60)
    logger.info("QUILLAN-RONIN PHASE 1: ANALYSIS & MAPPING")
    logger.info("=" * 60)
    
    # Analyze each source model
    llama = analyze_llama()
    qwen = analyze_qwen()
    bitnet = analyze_bitnet()
    
    # Build mapping
    mapping = build_mapping(llama, qwen, bitnet)
    
    # Save mapping
    output_path = "transplant_map.json"
    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2, default=str)
    
    logger.info(f"Mapping saved to {output_path}")
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
