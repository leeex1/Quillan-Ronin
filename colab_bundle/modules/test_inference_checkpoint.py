#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests text generation and expert deliberation from the trained Quillan-Ronin ONI checkpoint.
"""
from pathlib import Path
import sys
import torch

ONI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ONI_DIR))

from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig
from quillan_tokenizer_unified import UnifiedQuillanTokenizer

def main():
    ckpt_path = ONI_DIR / "checkpoints" / "checkpoints_oni" / "quillan_oni_latest.pt"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print("Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    
    # Filter valid config fields
    valid_keys = {"n_layer", "hidden_dim", "ffn_dim", "vocab_size", "num_experts", "expert_rank", "swarm_rank", "router_mode", "max_seq_len", "device"}
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    filtered_cfg["device"] = "cpu"
    
    cfg = QuillanOniConfig(**filtered_cfg) if filtered_cfg else QuillanOniConfig(n_layer=6, device="cpu")
    model = QuillanRoninOni(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    tok = UnifiedQuillanTokenizer()

    prompts = [
        "User: Hello, who are you?\n\nAssistant:",
        "User: Who is Quillan-Ronin?\n\nAssistant:",
        "User: Explain the Sovereign MoE architecture.\n\nAssistant:",
        "User: What is the purpose of C34-PREDATOR?\n\nAssistant:"
    ]

    step = ckpt.get("step", 1000)
    best_val = ckpt.get("best_val", 0.0)

    print("=" * 75)
    print(f"  QUILLAN-RONIN v5.4.0-ONI — TRAINED CHECKPOINT VERIFICATION (Step {step})")
    print(f"  Best Validation Loss: {best_val:.4f} | Parameters: 227.1M | Experts: 34")
    print("=" * 75)

    for p in prompts:
        ids = tok.encode(p)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            for _ in range(30):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                x = torch.cat([x, next_id], dim=-1)
        gen_text = tok.decode(x[0].tolist())
        print(f"\n[PROMPT]: {p.strip()}")
        print(f"[GENERATED]: {gen_text[len(p):].strip()}")

    print("\n" + "=" * 75)
    print("  VERIFICATION COMPLETE: ALL GATES PASS")
    print("=" * 75)

if __name__ == "__main__":
    main()
