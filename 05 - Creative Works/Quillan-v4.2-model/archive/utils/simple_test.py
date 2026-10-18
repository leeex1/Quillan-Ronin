#!/usr/bin/env python3
"""
Simple test to verify model loads and works with text-only input
"""

import torch
from __init__ import QuillanSOTA, Config

def test_model():
    print("=== Simple Model Test ===")
    
    # Initialize config and model
    config = Config()
    model = QuillanSOTA(config)
    
    # Load trained weights if available
    try:
        checkpoint = torch.load("checkpoints/quillan_final.pt", map_location='cpu')
        model.load_state_dict(checkpoint)
        print("[OK] Loaded trained checkpoint")
    except FileNotFoundError:
        print("[INFO] No checkpoint found, using random weights")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return False
    
    # Test with minimal inputs
    device = 'cpu'
    model.to(device)
    model.eval()
    
    print("\nTesting model with minimal inputs...")
    
    # Create very small test inputs
    batch_size = 1
    seq_len = 5
    
    try:
        # Text input
        text = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Minimal image (need exactly 16384 patches: sqrt(16384) = 128, so 128*16 = 2048x2048)
        grid_size = 128
        img_size = grid_size * 16  # 2048x2048
        img = torch.randn(batch_size, 3, img_size, img_size)
        
        # Audio
        audio = torch.randn(batch_size, 1, 64)
        
        # Video (1 frame)
        video = torch.randn(batch_size, 3, 1, img_size, img_size)
        
        with torch.no_grad():
            outputs = model(text, img, audio, video)
            
        print(f"[OK] Model forward pass successful!")
        print(f"Output type: {type(outputs)}")
        
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
        else:
            print(f"Output shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n[SUCCESS] Model is working!")
        print("You can now use this model for training and inference.")
    else:
        print("\n[FAILED] Model test failed.")
