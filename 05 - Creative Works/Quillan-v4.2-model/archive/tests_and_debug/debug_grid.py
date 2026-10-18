#!/usr/bin/env python3
"""
Debug script to test grid mismatches in Quillan model
"""

import torch
import os
import sys

# Add the model directory to path
sys.path.insert(0, os.path.dirname(__file__))

from __init__ import QuillanSOTA, Config

def test_grid_matches():
    """Test the model forward pass to identify grid mismatches"""
    print("🧪 Testing Quillan model grid matches...")

    # Configuration
    config = Config()
    model = QuillanSOTA(config)
    model.eval()

    # Create test inputs matching our data loader shapes
    batch_size = 1
    text = torch.randint(0, config.vocab_size, (batch_size, 128))  # 128 sequence length
    img = torch.randn(batch_size, 3, 256, 256)  # 256x256 images
    aud = torch.randn(batch_size, 1, 2048)  # 2048 audio length
    vid = torch.randn(batch_size, 3, 8, 32, 32)  # 8 frames, 32x32 video

    print(f"Input shapes:")
    print(f"  Text: {text.shape}")
    print(f"  Image: {img.shape}")
    print(f"  Audio: {aud.shape}")
    print(f"  Video: {vid.shape}")

    try:
        with torch.no_grad():
            outputs = model(text, img, aud, vid)

        print("✅ Forward pass successful!")
        print(f"Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

    except Exception as e:
        print(f"❌ Grid mismatch error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    test_grid_matches()
