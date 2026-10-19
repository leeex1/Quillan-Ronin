#!/usr/bin/env python3
"""
Real inference script for Quillan using trained model
"""

import torch
import argparse
import json
import os
from data_loader import QuillanDataset
from __init__ import QuillanSOTA, Config

def load_trained_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model with proper configuration"""
    print(f"🔄 Loading trained model from {checkpoint_path}")
    
    # Initialize model
    config = Config()
    model = QuillanSOTA(config)
    
    # Load checkpoint
    try:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"📊 Checkpoint loss: {checkpoint.get('loss', 'unknown'):.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Loaded legacy checkpoint format")
        else:
            print(f"⚠️  Checkpoint not found, using random weights")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"⚠️  Using random weights")
    
    model.to(device)
    model.eval()
    return model

def generate_multimodal_response(model, prompt_text: str, device: str = 'cpu', max_tokens: int = 100):
    """Generate response using multimodal inputs"""
    print(f"🎯 Generating response for: '{prompt_text[:100]}...'")
    
    # Create multimodal inputs
    batch_size = 1
    seq_len = len(prompt_text) + 10
    
    # Text input (mock tokenization)
    text_tokens = [min(ord(c), 999) for c in prompt_text[:seq_len]]
    if len(text_tokens) < seq_len:
        text_tokens.extend([0] * (seq_len - len(text_tokens)))
    text_input = torch.tensor([text_tokens], dtype=torch.long).to(device)
    
    # Image input (512x512 for proper patch count)
    img_size = 512
    image_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # Audio input
    audio_input = torch.randn(batch_size, 1, 1024).to(device)
    
    # Video input
    video_input = torch.randn(batch_size, 3, 8, img_size, img_size).to(device)
    
    with torch.no_grad():
        try:
            # Forward pass
            outputs = model(text_input, image_input, audio_input, video_input)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'text' in outputs:
                    logits = outputs['text']
                elif 'logits' in outputs:
                    logits = outputs['logits']
                else:
                    logits = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
            else:
                logits = outputs
            
            # Generate tokens
            if logits.dim() == 3:  # [batch, seq, vocab]
                generated_tokens = []
                current_input = text_input
                
                for _ in range(max_tokens):
                    # Get next token logits
                    next_logits = logits[0, -1, :]  # Last token, batch 0
                    next_probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(next_probs, 1).item()
                    
                    if next_token == 0:  # End token
                        break
                    
                    generated_tokens.append(next_token)
                    
                    # Update input for next iteration
                    next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
                    current_input = torch.cat([current_input, next_token_tensor], dim=1)
                    
                    # Get next prediction
                    if current_input.size(1) >= 512:  # Prevent too long sequences
                        break
                    
                    # Forward pass with updated input
                    outputs = model(current_input, image_input, audio_input, video_input)
                    if isinstance(outputs, dict):
                        if 'text' in outputs:
                            logits = outputs['text']
                        elif 'logits' in outputs:
                            logits = outputs['logits']
                        else:
                            logits = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
                    else:
                        logits = outputs
                
                # Decode tokens (simple character decoding)
                generated_text = ""
                for token in generated_tokens:
                    if token > 0 and token < 256:  # Valid ASCII range
                        generated_text += chr(token)
                
                return generated_text
            
            else:
                return "Model output format not recognized"
                
        except Exception as e:
            return f"Generation error: {str(e)}"

def interactive_mode():
    """Interactive inference mode"""
    print("🤖 Quillan Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    # Load model
    model = load_trained_model("checkpoints/quillan_final_real.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    while True:
        try:
            prompt = input("\n🎤 Your prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            # Generate response
            response = generate_multimodal_response(model, prompt, device, max_tokens=150)
            print(f"\n🤖 Quillan: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}")

def batch_inference(prompts_file: str, output_file: str):
    """Batch inference from file"""
    print(f"📚 Batch inference from {prompts_file}")
    
    # Load model
    model = load_trained_model("checkpoints/quillan_final_real.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Read prompts
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error reading prompts file: {e}")
        return
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"🎯 Processing prompt {i}/{len(prompts)}")
        response = generate_multimodal_response(model, prompt, device, max_tokens=200)
        results.append({
            'prompt': prompt,
            'response': response,
            'timestamp': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
        })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan Real Inference")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="Inference mode")
    parser.add_argument("--prompts", type=str, help="File with prompts for batch mode")
    parser.add_argument("--output", type=str, default="quillan_responses.json",
                       help="Output file for batch mode")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/quillan_final_real.pt",
                       help="Path to trained model checkpoint")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_mode()
    elif args.mode == "batch":
        if not args.prompts:
            print("❌ Batch mode requires --prompts file")
        else:
            batch_inference(args.prompts, args.output)
