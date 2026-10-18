#!/usr/bin/env python3
"""
Simple Diagnostic Test for Multimodal Inference Issues
"""

import torch
import time
import os
from train_full_multimodal import QuillanRoninV5_3, Config, SimpleTokenizer
from data_loader import QuillanDataset

def diagnostic_test():
    """Run basic diagnostic tests to identify issues"""
    print("🔍 Running Multimodal Inference Diagnostics")
    print("=" * 50)

    # Test 1: Configuration loading
    print("1️⃣ Testing Configuration Loading...")
    try:
        cfg = Config()
        print("   ✅ Config loaded successfully")
        print(f"   • Hidden dims: {cfg.hidden_dim}")
        print(f"   • Experts: {cfg.num_experts}")
        print(f"   • Sub-agents: {cfg.num_subagents}")
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        return False

    # Test 2: Model initialization
    print("\n2️⃣ Testing Model Initialization...")
    try:
        start_time = time.time()
        device = torch.device('cpu')  # Use CPU for diagnostics
        cfg.device = device

        model = QuillanRoninV5_3(cfg).to(device)
        init_time = time.time() - start_time
        print(f"   ✅ Model created in {init_time:.2f}s")
        print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Checkpoint loading
    print("\n3️⃣ Testing Checkpoint Loading...")
    checkpoint_path = "best_multimodal_quillan.pt"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("   ✅ Checkpoint loaded from model_state_dict")
            else:
                model.load_state_dict(checkpoint)
                print("   ✅ Checkpoint loaded directly")
        except Exception as e:
            print(f"   ❌ Checkpoint loading failed: {e}")
            print("   🔄 Continuing with untrained model")
    else:
        print("   ⚠️ No checkpoint found, using untrained model")

    # Test 4: Tokenizer and dataset
    print("\n4️⃣ Testing Tokenizer and Dataset...")
    try:
        dataset = QuillanDataset()
        tokenizer = SimpleTokenizer(vocab_size=1000)
        all_texts = [s['text'] for s in dataset.samples]
        tokenizer.train(all_texts)
        print("   ✅ Tokenizer and dataset ready")
        print(f"   • Dataset samples: {len(dataset.samples)}")
        print(f"   • Vocab size: {len(tokenizer.char_to_idx)}")
    except Exception as e:
        print(f"   ❌ Tokenizer/dataset failed: {e}")
        return False

    # Test 5: Basic inference
    print("\n5️⃣ Testing Basic Inference...")
    try:
        model.eval()

        # Create minimal test inputs
        text_tokens = torch.tensor([[1, 2, 3, 4]], device=device)  # Simple tokens
        image_input = torch.randn(1, cfg.image_channels, cfg.image_size, cfg.image_size, device=device)
        audio_input = torch.randn(1, 1, cfg.audio_samples // 64, device=device)
        video_input = torch.randn(1, cfg.video_channels, cfg.video_frames // 8, cfg.video_height // 8, cfg.video_width // 8, device=device)

        print("   📥 Test inputs created:")
        print(f"      Text: {text_tokens.shape}")
        print(f"      Image: {image_input.shape}")
        print(f"      Audio: {audio_input.shape}")
        print(f"      Video: {video_input.shape}")

        start_time = time.time()
        with torch.no_grad():
            outputs = model(text_tokens, image_input, audio_input, video_input)
        inference_time = time.time() - start_time

        print(f"   ✅ Inference completed! (in {inference_time:.3f}s)")
        print("   📤 Output validation:")
        print(f"      Text logits: {outputs['text'].shape}")
        print(f"      Image: {outputs['image'].shape}")
        print(f"      Audio: {outputs['audio'].shape}")
        print(f"      Video: {outputs['video'].shape}")

        # Check for NaNs
        has_nans = any(torch.isnan(outputs[key]).any() for key in outputs.keys() if isinstance(outputs[key], torch.Tensor))
        if has_nans:
            print("   ⚠️ Warning: NaN values detected in outputs")
        else:
            print("   ✅ No NaN values detected")

    except Exception as e:
        print(f"   ❌ Basic inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("🎉 DIAGNOSTIC TESTS PASSED!")
    print("📊 The multimodal pipeline is functional")
    print("🚀 Ready for comprehensive testing")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = diagnostic_test()
    if not success:
        print("\n❌ Diagnostics failed - check issues above")
        exit(1)
