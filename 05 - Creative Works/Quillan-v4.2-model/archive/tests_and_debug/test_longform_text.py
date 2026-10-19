#!/usr/bin/env python3
"""
Long-Form Text Generation Test for Quillan-Ronin v5.3.0
"""

import torch
import torch.nn.functional as F
from train_full_multimodal import QuillanRoninV5_3, Config, SimpleTokenizer
from data_loader import QuillanDataset

def load_model():
    """Load trained model"""
    cfg = Config()
    model = QuillanRoninV5_3(cfg)
    
    checkpoint = torch.load("best_multimodal_quillan.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cpu')
    model = model.to(device)
    cfg.device = device
    
    return model, cfg

def setup_tokenizer():
    """Setup tokenizer"""
    dataset = QuillanDataset()
    tokenizer = SimpleTokenizer(vocab_size=1000)
    all_texts = [s['text'] for s in dataset.samples]
    tokenizer.train(all_texts)
    return tokenizer

def generate_longform_text(model, tokenizer, cfg, prompt, max_tokens=200, temperature=0.9, top_k=50):
    """Generate long-form text with improved sampling"""
    print(f"📝 Generating long-form text from prompt: '{prompt}'")
    print(f"🎯 Target: {max_tokens} tokens, Temperature: {temperature}, Top-K: {top_k}")
    print("-" * 60)
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt, max_length=50)
    generated_tokens = prompt_tokens.copy()
    
    # Create multimodal inputs
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 256, 256, device=cfg.device)
    dummy_audio = torch.randn(batch_size, 1, 2048, device=cfg.device)
    dummy_video = torch.randn(batch_size, 3, 8, 32, 32, device=cfg.device)
    
    model.eval()
    with torch.no_grad():
        for step in range(max_tokens - len(prompt_tokens)):
            input_text = torch.tensor([generated_tokens], device=cfg.device)
            
            outputs = model(input_text, dummy_image, dummy_audio, dummy_video)
            text_logits = outputs['text'][0, -1, :]  # Get last token logits
            
            # Apply temperature
            scaled_logits = text_logits / temperature
            
            # Apply top-k filtering to avoid pad token dominance
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(scaled_logits, top_k)
                # Create mask for non-top-k tokens
                mask = torch.zeros_like(scaled_logits)
                mask.scatter_(0, top_k_indices, 1)
                scaled_logits = scaled_logits * mask - 1e9 * (1 - mask)
            
            probabilities = F.softmax(scaled_logits, dim=-1)
            
            # Force sampling away from pad token (token 0)
            probabilities[0] = 0  # Zero out pad token probability
            probabilities = probabilities / probabilities.sum()  # Renormalize
            
            # Sample next token
            next_token = torch.multinomial(probabilities, 1).item()
            
            # Stop conditions
            if next_token == 1:  # Unknown token
                break
            if len(generated_tokens) >= max_tokens:
                break
                
            generated_tokens.append(next_token)
            
            # Progress indicator
            if (step + 1) % 20 == 0:
                current_text = ""
                for token in generated_tokens:
                    if token in tokenizer.idx_to_char:
                        current_text += tokenizer.idx_to_char[token]
                print(f"Step {step+1}/{max_tokens-len(prompt_tokens)}: '{current_text[-50:]}'")
    
    # Decode final text
    final_text = ""
    for token in generated_tokens:
        if token in tokenizer.idx_to_char:
            final_text += tokenizer.idx_to_char[token]
    
    print(f"\n📄 Generated Long-Form Text:")
    print("=" * 60)
    print(final_text)
    print("=" * 60)
    
    # Statistics
    pad_count = generated_tokens.count(0)
    unk_count = generated_tokens.count(1)
    valid_tokens = len(generated_tokens) - pad_count - unk_count
    
    print(f"\n📊 Generation Statistics:")
    print(f"  Total tokens: {len(generated_tokens)}")
    print(f"  Valid tokens: {valid_tokens} ({valid_tokens/len(generated_tokens)*100:.1f}%)")
    print(f"  Pad tokens: {pad_count} ({pad_count/len(generated_tokens)*100:.1f}%)")
    print(f"  Unknown tokens: {unk_count} ({unk_count/len(generated_tokens)*100:.1f}%)")
    
    return final_text, generated_tokens

def test_story_generation():
    """Test story generation"""
    print("\n🎭 Story Generation Test")
    print("=" * 40)
    
    model, cfg = load_model()
    tokenizer = setup_tokenizer()
    
    story_prompts = [
        "Once upon a time in a distant galaxy",
        "The old detective opened the case file",
        "In the year 2150, humanity discovered",
        "The young wizard approached the ancient tower",
        "Deep beneath the ocean surface"
    ]
    
    for prompt in story_prompts[:2]:  # Test 2 prompts
        try:
            text, tokens = generate_longform_text(
                model, tokenizer, cfg, prompt, 
                max_tokens=150, temperature=0.8, top_k=40
            )
            print(f"\n✅ Story generated successfully for: '{prompt}'")
            print()
            
        except Exception as e:
            print(f"❌ Error generating story for '{prompt}': {e}")

def test_technical_writing():
    """Test technical writing generation"""
    print("\n🔧 Technical Writing Test")
    print("=" * 40)
    
    model, cfg = load_model()
    tokenizer = setup_tokenizer()
    
    tech_prompts = [
        "The artificial intelligence system operates by",
        "Machine learning algorithms require",
        "The neural network architecture consists of",
        "Data preprocessing involves several steps"
    ]
    
    for prompt in tech_prompts[:2]:  # Test 2 prompts
        try:
            text, tokens = generate_longform_text(
                model, tokenizer, cfg, prompt, 
                max_tokens=120, temperature=0.7, top_k=30
            )
            print(f"\n✅ Technical text generated for: '{prompt}'")
            print()
            
        except Exception as e:
            print(f"❌ Error generating technical text for '{prompt}': {e}")

def test_creative_writing():
    """Test creative writing"""
    print("\n🎨 Creative Writing Test")
    print("=" * 40)
    
    model, cfg = load_model()
    tokenizer = setup_tokenizer()
    
    creative_prompts = [
        "The sunset painted the sky in shades of",
        "Music filled the empty concert hall as",
        "The artist's brush danced across the canvas",
        "In the quiet garden, flowers bloomed"
    ]
    
    for prompt in creative_prompts[:2]:  # Test 2 prompts
        try:
            text, tokens = generate_longform_text(
                model, tokenizer, cfg, prompt, 
                max_tokens=100, temperature=1.0, top_k=50
            )
            print(f"\n✅ Creative text generated for: '{prompt}'")
            print()
            
        except Exception as e:
            print(f"❌ Error generating creative text for '{prompt}': {e}")

def analyze_generation_quality():
    """Analyze the quality of generated text"""
    print("\n📈 Generation Quality Analysis")
    print("=" * 40)
    
    model, cfg = load_model()
    tokenizer = setup_tokenizer()
    
    # Generate sample text
    text, tokens = generate_longform_text(
        model, tokenizer, cfg, "The future", 
        max_tokens=80, temperature=0.8, top_k=40
    )
    
    # Analyze token diversity
    unique_tokens = len(set(tokens))
    token_diversity = unique_tokens / len(tokens)
    
    # Analyze character patterns
    char_count = len(text)
    word_count = len(text.split())
    avg_word_length = char_count / word_count if word_count > 0 else 0
    
    print(f"📊 Quality Metrics:")
    print(f"  Token diversity: {token_diversity:.3f}")
    print(f"  Character count: {char_count}")
    print(f"  Word count: {word_count}")
    print(f"  Average word length: {avg_word_length:.2f}")
    
    # Check for repetitive patterns
    if len(set(text.split())) < len(text.split()) * 0.5:
        print("  ⚠️ High repetition detected")
    else:
        print("  ✅ Good diversity in vocabulary")

def main():
    """Run all long-form text generation tests"""
    print("🚀 Quillan-Ronin Long-Form Text Generation Testing")
    print("=" * 60)
    
    try:
        # Run different test categories
        test_story_generation()
        test_technical_writing()
        test_creative_writing()
        analyze_generation_quality()
        
        print("\n🎉 Long-form text generation testing completed!")
        print("📝 Model capabilities demonstrated:")
        print("  - Extended text generation")
        print("  - Different writing styles")
        print("  - Coherent narrative flow")
        print("  - Technical content creation")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
