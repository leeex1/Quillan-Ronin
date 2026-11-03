#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Diagnostic Probe for Weight-Space Programmed Models
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

def probe_model(ckpt_path: str, model_tag: str):
    print(f"\n=======================================================")
    print(f"PROBING WSP MODEL: {model_tag}")
    print(f"=======================================================")
    tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
    tok = Tokenizer.from_file(tok_path)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_cfg = ckpt["config"]
    import inspect
    valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_keys}
    cfg = QuillanOniConfig(**filtered_cfg)
    
    model = QuillanRoninOni(cfg).cpu().eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    test_prompts = [
        "Quillan is a sovereign cognitive agent. Its mission is to",
        "Question: If all humans are mortal and Socrates is human, is Socrates mortal? Provide the formal syllogistic proof.\nAnswer:\n",
        "In the bushido code, honor and rectitude are"
    ]
    
    for p in test_prompts:
        enc = tok.encode(p).ids
        gen = list(enc)
        print(f"\n--- Prompt: {repr(p)} ---")
        with torch.no_grad():
            for _ in range(25):
                x = torch.tensor([gen[-64:]], dtype=torch.long)
                out = model(x, deliberation=False)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[0, -1, :].clone().float()
                
                # Sign-preserving repetition penalty
                for prev in set(gen[-16:]):
                    if next_logits[prev] > 0:
                        next_logits[prev] /= 1.25
                    else:
                        next_logits[prev] *= 1.25
                        
                v, idxs = torch.topk(next_logits / 0.7, 40)
                probs = F.softmax(v, dim=-1)
                idx = idxs[torch.multinomial(probs, 1).item()].item()
                gen.append(idx)
                if idx in [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")]:
                    break
        out_text = tok.decode(gen)
        print(f"Generated:\n{out_text}")

if __name__ == "__main__":
    p_mini = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_tripartite.pt"
    if os.path.exists(p_mini):
        probe_model(p_mini, "Mini 0.6B WSP")

