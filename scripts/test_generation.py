#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F

ROOT = r"C:\Users\Admin\Quillan-Ronin"
sys.path.insert(0, os.path.join(ROOT, "_dev"))
sys.path.insert(0, ROOT)

from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from quillan_bpe_tokenizer import QuillanBPETokenizer

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, device='cuda'):
    print(f"\n--- Generating text from prompt: '{prompt}' ---")
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            out = model(x)
            logits = out['logits']
            
            # Scale by temperature and sample
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
            
            # Print decoded token on the fly
            decoded_token = tokenizer.decode([next_token.item()])
            try:
                print(decoded_token, end="", flush=True)
            except UnicodeEncodeError:
                print("?", end="", flush=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    print("\n-------------------------------------------")

def main():
    device = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = QuillanBPETokenizer()
    tokenizer.load(os.path.join(ROOT, "quillan_bpe_tokenizer.pkl"))
    print(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}")
    
    # Load model
    cfg = QuillanArchConfig(device=device)
    model = QuillanRoninSovereign(cfg)
    
    ckpt_path = "checkpoints/quillan_finetuned.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/router_trained.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/quillan_fixed.pt"
        
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print("Model loaded successfully.")
    
    prompts = [
        "The future of artificial intelligence in software development is",
        "Explain the concept of quantum computing in simple terms:",
        "def quicksort(arr):"
    ]
    
    for prompt in prompts:
        generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, device=device)

if __name__ == "__main__":
    main()
