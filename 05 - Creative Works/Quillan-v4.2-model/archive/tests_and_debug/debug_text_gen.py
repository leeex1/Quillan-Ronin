#!/usr/bin/env python3
"""
Simple Text Generation Debug Test
"""

import torch
import torch.nn.functional as F
from train_full_multimodal import QuillanRoninV5_3, Config, SimpleTokenizer
from data_loader import QuillanDataset

def quick_test():
    """Quick text generation test"""
    print("🔍 Quick Text Generation Debug")
    print("=" * 40)
    
    # Load model
    cfg = Config()
    model = QuillanRoninV5_3(cfg)
    
    checkpoint = torch.load("best_multimodal_quillan.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cpu')
    model = model.to(device)
    cfg.device = device
    
    # Setup tokenizer
    dataset = QuillanDataset()
    tokenizer = SimpleTokenizer(vocab_size=1000)
    all_texts = [s['text'] for s in dataset.samples]
    tokenizer.train(all_texts)
    
    print(f"Vocabulary size: {len(tokenizer.char_to_idx)}")
    print(f"Sample tokens: {list(tokenizer.char_to_idx.keys())[:10]}")
    
    # Test single generation
    prompt = "AI"
    prompt_tokens = tokenizer.encode(prompt, max_length=10)
    print(f"Prompt tokens: {prompt_tokens}")
    
    # Create inputs
    input_text = torch.tensor([prompt_tokens], device=device)
    dummy_image = torch.randn(1, 3, 256, 256, device=device)
    dummy_audio = torch.randn(1, 1, 2048, device=device)
    dummy_video = torch.randn(1, 3, 8, 32, 32, device=device)
    
    with torch.no_grad():
        outputs = model(input_text, dummy_image, dummy_audio, dummy_video)
        
        # Analyze text logits
        text_logits = outputs['text']  # Shape: [1, seq_len, vocab_size]
        print(f"Text logits shape: {text_logits.shape}")
        
        # Get last token predictions
        last_logits = text_logits[0, -1, :]  # Remove batch, get last token
        probabilities = F.softmax(last_logits, dim=-1)
        
        # Show top predictions
        top_probs, top_indices = torch.topk(probabilities, 10)
        
        print(f"\nTop 10 predictions for next token:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            char = tokenizer.idx_to_char.get(idx.item(), f"<unk_{idx.item()}>")
            print(f"  {i+1}. '{char}' (token {idx.item()}) - prob: {prob.item():.4f}")
        
        # Sample next token
        next_token = torch.multinomial(probabilities, 1).item()
        next_char = tokenizer.idx_to_char.get(next_token, f"<unk_{next_token}>")
        
        print(f"\nSampled next token: '{next_char}' (token {next_token})")
        
        # Decode full sequence
        full_tokens = prompt_tokens + [next_token]
        decoded = ""
        for token in full_tokens:
            if token in tokenizer.idx_to_char:
                decoded += tokenizer.idx_to_char[token]
        
        print(f"Decoded sequence: '{decoded}'")

if __name__ == "__main__":
    quick_test()
