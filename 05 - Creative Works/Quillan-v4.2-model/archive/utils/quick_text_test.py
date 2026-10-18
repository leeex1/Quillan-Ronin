#!/usr/bin/env python3
"""
Quick Text Generation Test with Bias Correction
"""

import torch
import torch.nn.functional as F
from train_full_multimodal import QuillanRoninV5_3, Config, SimpleTokenizer
from data_loader import QuillanDataset

def quick_text_test():
    """Quick text generation with pad token bias correction"""
    print("🔧 Quick Text Generation with Bias Correction")
    print("=" * 50)
    
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
    
    print(f"Vocabulary: {len(tokenizer.char_to_idx)} tokens")
    
    # Test generation with strong bias correction
    prompt = "AI will"
    prompt_tokens = tokenizer.encode(prompt, max_length=10)
    generated_tokens = prompt_tokens.copy()
    
    print(f"Prompt: '{prompt}' -> tokens: {prompt_tokens}")
    
    # Create inputs
    dummy_image = torch.randn(1, 3, 256, 256, device=device)
    dummy_audio = torch.randn(1, 1, 2048, device=device)
    dummy_video = torch.randn(1, 3, 8, 32, 32, device=device)
    
    print("\n📝 Generating text...")
    
    model.eval()
    with torch.no_grad():
        for i in range(20):  # Generate 20 tokens
            input_text = torch.tensor([generated_tokens], device=device)
            outputs = model(input_text, dummy_image, dummy_audio, dummy_video)
            
            # Get logits for next token
            text_logits = outputs['text'][0, -1, :]
            
            # STRONGLY bias against pad token (token 0)
            text_logits[0] = -1000  # Make pad token very unlikely
            text_logits[1] = -500   # Make unk token unlikely too
            
            # Apply temperature
            text_logits = text_logits / 0.8  # Temperature 0.8
            
            probabilities = F.softmax(text_logits, dim=-1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            print(f"Step {i+1}: Top 5 predictions:")
            for j, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                char = tokenizer.idx_to_char.get(idx.item(), f"<unk_{idx.item()}>")
                print(f"  {j+1}. '{char}' - prob: {prob.item():.4f}")
            
            # Sample from top 3 (more diverse)
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            next_token = torch.multinomial(top3_probs, 1).item()
            next_char = tokenizer.idx_to_char.get(next_token, f"<unk_{next_token}>")
            
            print(f"  Selected: '{next_char}' (token {next_token})")
            
            generated_tokens.append(next_token)
            
            # Stop if we get reasonable content
            if next_token not in [0, 1]:  # Not pad or unk
                break
            
            print()
    
    # Decode the result
    decoded_text = ""
    for token in generated_tokens:
        if token in tokenizer.idx_to_char:
            decoded_text += tokenizer.idx_to_char[token]
    
    print(f"\n📄 Final Result:")
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{decoded_text}'")
    
    # Test with different prompts
    print(f"\n🔄 Testing different prompts...")
    
    test_prompts = ["The", "In", "Machine", "Future"]
    
    for prompt in test_prompts:
        try:
            prompt_tokens = tokenizer.encode(prompt, max_length=10)
            generated_tokens = prompt_tokens.copy()
            
            for _ in range(10):
                input_text = torch.tensor([generated_tokens], device=device)
                outputs = model(input_text, dummy_image, dummy_audio, dummy_video)
                
                text_logits = outputs['text'][0, -1, :]
                text_logits[0] = -1000  # Anti-pad bias
                text_logits[1] = -500   # Anti-unk bias
                text_logits = text_logits / 0.8
                
                probabilities = F.softmax(text_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1).item()
                generated_tokens.append(next_token)
                
                if next_token not in [0, 1]:
                    break
            
            decoded = ""
            for token in generated_tokens:
                if token in tokenizer.idx_to_char:
                    decoded += tokenizer.idx_to_char[token]
            
            print(f"  '{prompt}' -> '{decoded}'")
            
        except Exception as e:
            print(f"  '{prompt}' -> Error: {e}")

if __name__ == "__main__":
    quick_text_test()
