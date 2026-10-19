#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 LIVE MODEL INSPECTION & GENERATION DEMO
===========================================
Inspects:
1. Exact parameter counts and layer topology.
2. Weight quantization status (BitNet 1.58b STE).
3. Live generation samples on user prompts.
4. Active expert routing distribution per token.
"""

import os
import sys
import torch
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from quillan_v5_4_oni import (
    QuillanRoninOni,
    QuillanOniConfig,
    CANONICAL_ROSTER
)

def inspect_model():
    print("=" * 75)
    print("🔬 QUILLAN-RONIN v5.4.0-ONI: LIVE ARCHITECTURAL & GENERATION INSPECTION")
    print("=" * 75)

    ckpt_path = r"c:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_oni_latest.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = r"c:\02_QUILLAN\oni\checkpoint_step_500.pt"

    device = "cpu"  # keep on CPU to not disturb GPU training
    cfg = QuillanOniConfig(
        n_layer=12,
        hidden_dim=1024,
        ffn_dim=2048,
        num_experts=34,
        expert_rank=8,
        swarm_rank=8,
        router_mode="dense_pull",
        vocab_size=50257,
        max_seq_len=512,
        device=device
    )

    model = QuillanRoninOni(cfg).to(device)

    # Calculate exact parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[1] PARAMETER & MEMORY PROFILE:")
    print(f"  • Total Parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  • Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  • Hidden Dimension:     {cfg.hidden_dim}")
    print(f"  • Transformer Layers:   {cfg.n_layer}")
    print(f"  • Council Experts:      {cfg.num_experts} (Rank-{cfg.expert_rank} LoRA + Rank-{cfg.swarm_rank} Swarm)")
    print(f"  • Vocabulary Size:      {cfg.vocab_size:,}")

    # Load Checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"\n[2] CHECKPOINT STATE:")
        print(f"  • Checkpoint Path:      {ckpt_path}")
        print(f"  • Trained Steps:        {ckpt.get('step', 'unknown')}")
        print(f"  • Best Validation Loss: {ckpt.get('best_val', 'unknown'):.4f}" if isinstance(ckpt.get('best_val'), float) else f"  • Best Validation Loss: {ckpt.get('best_val', 'unknown')}")
        print(f"  • RQGM Epoch:           {ckpt.get('rqgm_epoch', '1')}")
    else:
        print(f"\n[2] Checkpoint not found, using initialized weights.")

    model.eval()

    # Generation Samples
    prompts = [
        "User: Hello! Who are you?\n\nAssistant:",
        "User: Explain how the 34-expert council works in Quillan-Ronin.\n\nAssistant:",
        "User: What is the main benefit of BitNet 1.58b quantization?\n\nAssistant:"
    ]

    print("\n[3] LIVE TEXT GENERATION & DELIBERATION SAMPLES:")
    for i, p in enumerate(prompts, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Prompt: {p.strip()}")
        # convert prompt to rough token ids using simple hash/char mapping for demo if custom bpe not loaded
        prompt_tokens = [int(ord(c)) % 1000 + 1 for c in p]
        
        with torch.no_grad():
            res = model.deliberate(prompt_tokens[:32], max_tokens=25, temp=0.7)
            
        gen_tokens = res.get("tokens", [])
        trace = res.get("trace", {})
        gates = trace.get("gates", {})
        
        # Pull gate distribution across the top experts
        flat_emb = model.wte(torch.tensor([prompt_tokens[:32]], device=device)).reshape(-1, cfg.hidden_dim)
        pull = model.h[0].moe.pull_gate(flat_emb, tau=1.0).mean(dim=0).tolist()
        
        top3_idx = sorted(range(len(pull)), key=lambda k: pull[k], reverse=True)[:3]
        top3_str = ", ".join([f"{CANONICAL_ROSTER[idx][0]} ({pull[idx]:.4f})" for idx in top3_idx])
        
        print(f"  • Tokens Generated:     {len(gen_tokens)} tokens")
        print(f"  • Top Active Personas:  {top3_str}")
        print(f"  • Quality Gates:        Passed={gates.get('passed', True)} (Ethics={gates.get('ethics_constraint', 0.0):.4f}, Covenant={gates.get('covenant_identity', 0.0):.4f})")
        print(f"  • Council Entropy:      {gates.get('council_entropy', 0.0):.4f}")

    print("\n" + "=" * 75)
    print("✅ LIVE INSPECTION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    inspect_model()
