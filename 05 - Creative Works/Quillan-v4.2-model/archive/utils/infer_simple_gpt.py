#!/usr/bin/env python3
"""
Inference script for the trained simple GPT model
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    """Simple Bigram model from the notebook"""
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, decode_func):
        """Generate text"""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return decode_func(idx[0].tolist())

def load_trained_model():
    """Load the trained GPT model"""
    print("🧠 Loading trained GPT model...")

    # Load checkpoint
    checkpoint_path = "simple_gpt_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model file not found: {checkpoint_path}")
        print("Please run training first to create the model.")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Recreate tokenizer
    chars = checkpoint['chars']
    vocab_size = checkpoint['vocab_size']

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi.get(c, 0) for c in s]  # Default to 0 for unknown chars

    def decode(l):
        return ''.join([itos.get(i, '?') for i in l])  # Default to ? for unknown tokens

    # Initialize and load model
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✅ Model loaded successfully!")
    print(f"🏆 Vocab size: {vocab_size}")
    print(f"📏 Block size: {checkpoint['block_size']}")

    return model, decode

def generate_text(model, decode_func, prompt="", max_length=200):
    """Generate text from a prompt"""
    print(f"\n🔍 Generating from prompt: '{prompt}'")

    if prompt:
        # Encode prompt
        encoded = []
        for c in prompt:
            # Use same encoding as training
            encoded.append(min(ord(c), 190))  # Clamp to our vocab range
    else:
        encoded = [0]  # Start with padding token

    # Convert to tensor
    context = torch.tensor([encoded], dtype=torch.long)

    print("🎯 Generating text...")

    with torch.no_grad():
        generated_tokens = model.generate(context, max_length, decode_func)
        return generated_tokens

def interactive_mode(model, decode_func):
    """Interactive chat mode"""
    print("\n" + "="*60)
    print("🎉 TRAINED GPT MODEL - READY!")
    print("="*60)
    print("Your character-level GPT model is ready!")
    print("Type 'quit' to exit, or just press Enter for random generation")
    print("-" * 40)

    while True:
        try:
            prompt = input("\nPrompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            # Generate text
            generated_text = generate_text(model, decode_func, prompt, max_length=300)

            if generated_text:
                print(f"\nGenerated: {generated_text}")
            else:
                print("❌ Generation failed")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

def main():
    """Main function"""
    print("🚀 Starting GPT Inference...")

    # Load model
    model, decode_func = load_trained_model()
    if model is None:
        return

    # Start interactive mode
    interactive_mode(model, decode_func)

if __name__ == "__main__":
    import os
    main()
