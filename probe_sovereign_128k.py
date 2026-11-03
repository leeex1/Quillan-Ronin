#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical Generation Probe for Quillan Sovereign 1B (Quad-Donor WSP 128k)
"""

import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

def run_probe():
    ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_wsp128k.pt"
    llama_dir = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots"
    llama_snap = os.path.join(llama_dir, os.listdir(llama_dir)[0])
    tok_path = os.path.join(llama_snap, "tokenizer.json")
    
    print(f"Loading tokenizer: {tok_path}")
    tok = Tokenizer.from_file(tok_path)
    print(f"Vocab size: {tok.get_vocab_size()}")
    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    
    # Filter config keys
    import inspect
    valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    cfg = QuillanOniConfig(**filtered_cfg)
    
    model = QuillanRoninOni(cfg).cpu().eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("Model loaded into CPU eval mode successfully!")
    
    test_prompts = [
        "Quillan is a sovereign cognitive agent. Its mission is to",
        "In the bushido code, honor and rectitude are",
        "The fundamental law of gravity states that"
    ]
    
    for prompt in test_prompts:
        enc = tok.encode(prompt).ids
        gen = list(enc)
        print(f"\n=======================================================")
        print(f"PROMPT: {repr(prompt)}")
        print(f"=======================================================")
        
        with torch.no_grad():
            for step in range(25):
                inp = torch.tensor([gen[-64:]], dtype=torch.long)
                out = model(inp, deliberation=False)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[0, -1, :].clone().float()
                
                # Sign-preserving repetition penalty
                for prev in set(gen[-16:]):
                    if next_logits[prev] > 0:
                        next_logits[prev] /= 1.2
                    else:
                        next_logits[prev] *= 1.2
                        
                top_v, top_i = torch.topk(next_logits / 0.7, 40)
                probs = F.softmax(top_v, dim=-1)
                next_id = top_i[torch.multinomial(probs, 1).item()].item()
                gen.append(next_id)
                if next_id in [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")]:
                    break
                    
        decoded = tok.decode(gen)
        print(f"OUTPUT:\n{decoded}")

if __name__ == "__main__":
    run_probe()
