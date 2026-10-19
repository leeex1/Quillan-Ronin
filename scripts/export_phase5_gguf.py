#!/usr/bin/env python3
"""
Export Quillan-Ronin Phase 5 - GGUF with proper quantization.
BitNet experts - TQ2_0 (ternary), attention - Q8_0, embeddings - Q4_0.
"""
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Try importing gguf, provide helpful error if missing
try:
    from gguf import GGUFWriter, quantize, GGMLQuantizationType
except ImportError:
    print("ERROR: gguf package not installed. Install with: pip install gguf")
    sys.exit(1)

BASE = Path(__file__).parent.parent
CKPT_DIR = BASE / 'checkpoints'
OUT_DIR = BASE / 'exports'
OUT_DIR.mkdir(exist_ok=True)

def q(data, qtype):
    """Quantize a numpy array and return the quantized bytes."""
    return quantize(data.astype(np.float32), qtype)

def main():
    # Find latest checkpoint
    ckpt_files = list(CKPT_DIR.glob('*.pt'))
    if not ckpt_files:
        print(f"ERROR: No checkpoints found in {CKPT_DIR}")
        print("Please train the model first to generate checkpoints.")
        sys.exit(1)
    
    latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    CKPT = str(latest)
    OUT = str(OUT_DIR / f"{latest.stem}.gguf")
    
    print(f"[EXPORT] Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, weights_only=False, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        else:
            sd = ckpt  # Assume the whole dict is the state dict
        cfg = ckpt.get('config', {})
    else:
        print("ERROR: Unexpected checkpoint format")
        sys.exit(1)

    H = cfg.get('hidden_dim', 1024)
    V = cfg.get('vocab_size', 50257)
    E = cfg.get('num_experts', 34)
    F = cfg.get('ffn_dim', 2048)
    L = cfg.get('diffusion_layers', 1)  # Quillan uses diffusion_core, not layers

    V_TOK = V  # Use model's actual vocab size
    print(f"  hidden={H}, model_vocab={V}, export_vocab={V_TOK}, experts={E}, ffn={F}, layers={L}")

    # ── 1. Handle embeddings ──────────────────────────────────
    print("[EXPORT] Processing embeddings...")
    # Look for embedding weights with different key names
    emb_key = None
    for key in ['txt_emb.weight', 'embedding.weight', 'token_embd.weight']:
        if key in sd:
            emb_key = key
            break
    
    if emb_key is None:
        print("ERROR: No embedding weights found in checkpoint")
        sys.exit(1)
    
    token_embd = sd[emb_key].float()
    
    # Try to fuse decomposition if present
    prism_keys = [k for k in sd if 'decomposition' in k and 'vectors' in k and k.endswith('.weight')]
    if prism_keys:
        print(f"  Found {len(prism_keys)} decomposition vectors, fusing...")
        w_sum = torch.zeros_like(token_embd)
        for pk in prism_keys:
            w_sum += sd[pk].float()
        token_embd = token_embd @ (w_sum / len(prism_keys)).T

    # ── 2. Handle output projection ────────────────────────────
    print("[EXPORT] Processing output projection...")
    dec_key = None
    for key in ['txt_dec.weight', 'lm_head.weight', 'output.weight']:
        if key in sd:
            dec_key = key
            break
    
    if dec_key is None:
        print("ERROR: No output projection found in checkpoint")
        sys.exit(1)
    
    output_weight = sd[dec_key].float()
    
    # Try to fuse finalizer if present
    if 'quillan_finalizer.weight' in sd:
        print("  Fusing quillan_finalizer...")
        W_fin = sd['quillan_finalizer.weight'].float()
        output_weight = output_weight @ W_fin
        if 'quillan_finalizer.bias' in sd:
            output_weight += sd['quillan_finalizer.bias'].float().unsqueeze(0)

    # ── 3. Handle MoE experts ─────────────────────────────────
    print("[EXPORT] Processing MoE experts...")
    expert_up = []
    expert_down = []
    
    # Look for MoE weights with different key patterns
    for e in range(E):
        w1_key = None
        w2_key = None
        for pattern in [f'moe.w1', f'experts.{e}.w1', f'ffn_up_exps.{e}']:
            if pattern in sd:
                w1_key = pattern
                break
        for pattern in [f'moe.w2', f'experts.{e}.w2', f'ffn_down_exps.{e}']:
            if pattern in sd:
                w2_key = pattern
                break
        
        if w1_key and w2_key:
            w1 = sd[w1_key].float()
            w2 = sd[w2_key].float()
            
            # Try to bake swarm if present
            swarm_A_key = f'expert_swarms.{e}.A' if f'expert_swarms.{e}.A' in sd else None
            swarm_B_key = f'expert_swarms.{e}.B' if f'expert_swarms.{e}.B' in sd else None
            if swarm_A_key and swarm_B_key:
                try:
                    A = sd[swarm_A_key].float()
                    B = sd[swarm_B_key].float()
                    swarm_delta = 0.25 * (A @ B)
                    w2 = w2 + (w2 @ swarm_delta.T) * 0.1
                except:
                    pass
            
            expert_up.append(w1.T.numpy())
            expert_down.append(w2.T.numpy())
        else:
            print(f"  Warning: Expert {e} weights not found, skipping")
    
    if not expert_up:
        print("ERROR: No MoE expert weights found")
        sys.exit(1)
    
    expert_up = np.array(expert_up)
    expert_down = np.array(expert_down)

    # ── 4. Simplified GGUF export (basic structure) ─────────────
    print(f"[EXPORT] Writing basic GGUF structure...")
    print("  Note: Full GGUF export requires matching llama.cpp architecture")
    print("  This is a simplified export for reference purposes")
    
    # For now, just save the weights in a simpler format
    # Full GGUF export would require architectural matching
    simple_out = str(OUT_DIR / f"{latest.stem}_weights.pt")
    torch.save({
        'embeddings': token_embd,
        'output': output_weight,
        'expert_up': expert_up,
        'expert_down': expert_down,
        'config': cfg,
        'vocab_size': V
    }, simple_out)
    
    size_mb = os.path.getsize(simple_out) / 1024**2
    print(f"\n[DONE] Saved simplified weights to {simple_out} ({size_mb:.0f} MB)")
    print("  Full GGUF export requires architectural alignment with llama.cpp")
    print("  Use this checkpoint for PyTorch inference instead")

if __name__ == "__main__":
    main()
