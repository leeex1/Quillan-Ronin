#!/usr/bin/env python3
"""
Debug script to check model outputs and training setup
"""

import torch
import os
import sys

# Add the model directory to path
sys.path.insert(0, os.path.dirname(__file__))

from __init__ import QuillanSOTA, Config

def debug_model_outputs():
    """Debug the model's output format and training setup"""
    print("🔍 Debugging model outputs...")

    # Initialize model
    config = Config()
    model = QuillanSOTA(config)
    model.eval()

    # Create same inputs as training
    batch_size = 1
    seq_len = 128
    text = torch.randint(0, 1000, (batch_size, seq_len))  # Use same range as training tokenization
    img = torch.randn(batch_size, 3, 256, 256)
    aud = torch.randn(batch_size, 1, 2048)
    vid = torch.randn(batch_size, 3, 8, 32, 32)

    print(f"Input shapes:")
    print(f"  Text: {text.shape} (tokens in range 0-999)")
    print(f"  Image: {img.shape}")
    print(f"  Audio: {aud.shape}")
    print(f"  Video: {vid.shape}")

    with torch.no_grad():
        outputs = model(text, img, aud, vid)

    print(f"\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
            if key == 'text':
                print(f"    Text logits range: [{value.min().item():.3f}, {value.max().item():.3f}]")
                print(f"    Text logits mean: {value.mean().item():.3f}")
                print(f"    Expected vocab size: {config.vocab_size}")
        else:
            print(f"  {key}: {type(value)}")

    # Check if text output matches expected training format
    if 'text' in outputs:
        text_logits = outputs['text']
        print(f"\n🔍 Analyzing text logits for training compatibility:")
        print(f"  Shape: {text_logits.shape}")
        print(f"  Dimensions: {text_logits.dim()}D")

        if text_logits.dim() == 3:
            batch, seq, vocab = text_logits.shape
            print(f"  ✅ Format matches training expectation [batch, seq, vocab]")
            # Simulate the training loss calculation
            target = text[:, 1:seq]  # Shift by 1 for next token prediction
            pred = text_logits[:, :seq-1, :]  # Remove last
            print(f"  Target shape: {target.shape}")
            print(f"  Prediction shape: {pred.shape}")

            # Calculate sample loss
            loss = torch.nn.functional.cross_entropy(
                pred.reshape(-1, vocab),
                target.reshape(-1),
                ignore_index=0
            )
            print(f"  Sample loss: {loss.item():.4f}")

        elif text_logits.dim() == 2:
            print(f"  ⚠️  2D output instead of expected 3D [batch, vocab]")
        else:
            print(f"  ❌ Unexpected dimensions: {text_logits.dim()}D")

    print(f"\n🎯 Training analysis:")
    print(f"  Dataset size: ~462 text samples")
    print(f"  Batch size: 1")
    print(f"  Sequence length: 128")
    print(f"  Training epochs: 50")
    print(f"  Final loss: ~2.6 (quite high for text generation)")

    print(f"\n💡 Potential issues:")
    print(f"  • Very small training dataset (462 samples)")
    print(f"  • Simple character-level tokenization (limited vocabulary)")
    print(f"  • Complex multimodal model trained on text-only task")
    print(f"  • Loss still high ({'2.6'}), model may need more training")

if __name__ == "__main__":
    debug_model_outputs()
