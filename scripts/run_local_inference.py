#!/usr/bin/env python3
"""
Local inference script for Quillan model trained on Colab.
Loads checkpoints from checkpoints/ directory.
"""
import sys, os, json
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn.functional as F

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / '_dev'))

from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from quillan_bpe_tokenizer import QuillanBPETokenizer

# ─── TOKENIZER ────────────────────────────────────────────────────────────────
def load_tokenizer():
    """Load BPE tokenizer using QuillanBPETokenizer."""
    tok_path = BASE / 'training_data' / 'tokenizer.json'
    if not tok_path.exists():
        tok_path = BASE / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
    if not tok_path.exists():
        tok_path = BASE / '_dev' / 'Quillan-v4.2-model' / 'tokenizer.json'
    
    tokenizer = QuillanBPETokenizer()
    if tok_path.exists():
        tokenizer.load(str(tok_path))
        print(f'Loaded tokenizer: {tokenizer.vocab_size} vocab from {tok_path}')
    else:
        print('WARNING: No tokenizer.json found, using fallback')
        tokenizer = None
    
    return tokenizer

def tokenize(text, tokenizer, max_len=256):
    """Tokenize text using BPE tokenizer."""
    if tokenizer is None:
        return [ord(ch) % 50257 for ch in text[:max_len]]
    tokens = tokenizer.encode(text)
    return tokens[:max_len]

def detokenize(tokens, tokenizer):
    """Convert token IDs back to text."""
    if tokenizer is None:
        return ''.join(chr(t % 128) for t in tokens)
    return tokenizer.decode(tokens)

# ─── MODEL LOADING ────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path: str, device='cuda'):
    """Load trained checkpoint from Colab training."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    cfg = QuillanArchConfig(
        text_only=True,
        hidden_dim=config.get('hidden_dim', 2048),
        ffn_dim=config.get('ffn_dim', 5504),
        vocab_size=config.get('vocab_size', 128000),
        num_experts=config.get('num_experts', 34),
        top_k=config.get('top_k', 4),
        e_ice_limit_ms=100,
        device=device
    )
    
    model = QuillanRoninSovereign(cfg).to(device)
    state = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Training step: {ckpt.get('step', 'unknown')}")
    print(f"Training loss: {ckpt.get('loss', 'unknown')}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model

def generate_text(model, prompt, tokenizer, max_tokens=100, temperature=0.8, device='cuda'):
    """Generate text from prompt using trained model."""
    tokens = tokenize(prompt, tokenizer, max_len=256)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(x)
            logits = out['logits'] if isinstance(out, dict) else out
            next_token = torch.multinomial(F.softmax(logits[0, -1] / temperature, dim=-1), 1)
            generated.append(next_token.item())
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
    
    return detokenize(tokens + generated, tokenizer)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = load_tokenizer()
    
    # Find latest checkpoint
    ckpt_dir = BASE / 'checkpoints'
    ckpt_files = list(ckpt_dir.glob('*.pt'))
    
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        print("Please ensure training has produced checkpoints.")
        sys.exit(1)
    
    # Load latest checkpoint
    latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    model = load_checkpoint(str(latest), device)
    
    # Run inference
    prompt = "The future of AI is"
    print(f"\nPrompt: {prompt}")
    output = generate_text(model, prompt, tokenizer, max_tokens=50, temperature=0.8, device=device)
    print(f"Generated: {output}")
