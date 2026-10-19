#!/usr/bin/env python3
"""
Local inference script for trained Quillan model
Run your multimodal AI locally via Python
"""

import torch
import os
import sys

# Add the model directory to path
sys.path.insert(0, os.path.dirname(__file__))

from __init__ import QuillanSOTA, Config
from data_loader import QuillanDataset

def load_trained_model():
    """Load the trained Quillan model from checkpoint"""
    print("🧠 Loading trained Quillan model...")

    # Initialize model
    config = Config()
    model = QuillanSOTA(config)

    # Load checkpoint
    checkpoint_path = "checkpoints/quillan_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run training first to create checkpoints.")
        return None

    print(f"📁 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✅ Model loaded successfully!")
    print(f"🏆 Best loss achieved: {checkpoint.get('loss', 'Unknown'):.4f}")
    print(f"📅 Trained for {checkpoint.get('epoch', 'Unknown')} epochs")

    return model

def simple_text_inference(model, text_input, max_length=50):
    """Simple text-only inference for testing"""
    print(f"\n🔍 Processing: '{text_input}'")

    # Use the same tokenization as training
    tokens = [min(ord(c), 999) for c in text_input]
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]

    text_tensor = torch.tensor([tokens], dtype=torch.long)

    # Create dummy multimodal inputs for now (focus on text)
    batch_size = 1
    img = torch.randn(batch_size, 3, 256, 256)
    aud = torch.randn(batch_size, 1, 2048)
    vid = torch.randn(batch_size, 3, 8, 32, 32)

    print("🎯 Running inference...")

    with torch.no_grad():
        try:
            outputs = model(text_tensor, img, aud, vid)

            # Extract text output
            if 'text' in outputs:
                logits = outputs['text']

                # The model outputs next-token predictions
                # For a simple response, take the most likely tokens
                predicted_tokens = torch.argmax(logits, dim=-1)

                # Convert tokens back to characters (reverse of training tokenization)
                response_chars = []
                for token in predicted_tokens[0]:
                    token_val = token.item()
                    if token_val == 0:
                        break  # Stop at padding
                    elif 1 <= token_val <= 999:
                        # Convert back to character using chr, but clamp to printable range
                        char_code = min(max(token_val, 32), 126)
                        response_chars.append(chr(char_code))
                    # Skip tokens > 999 as they weren't used in training

                response_text = ''.join(response_chars).strip()

                if response_text:
                    print(f"📝 Response: {response_text}")
                    return response_text
                else:
                    print("📝 Response: [Generated empty response]")
                    return "[Empty response]"

        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

def interactive_mode(model):
    """Interactive chat mode"""
    print("\n" + "="*50)
    print("🎉 QUILLAN LOCAL INFERENCE - READY!")
    print("="*50)
    print("Your multimodal AI is running locally!")
    print("Type 'quit' to exit")
    print("-" * 30)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Run inference
            response = simple_text_inference(model, user_input)

            if response:
                print(f"Quillan: {response}")
            else:
                print("Quillan: [Processing failed]")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

def main():
    """Main function to launch local inference"""
    print("🚀 Starting Quillan Local Inference...")

    # Load model
    model = load_trained_model()
    if model is None:
        return

    # Start interactive mode
    interactive_mode(model)

if __name__ == "__main__":
    main()
