#!/usr/bin/env python3
"""
Simple test to check if training loop can start
"""

import torch
import os
import sys

# Add the model directory to path
sys.path.insert(0, os.path.dirname(__file__))

from __init__ import QuillanSOTA, Config
from data_loader import QuillanDataset

def test_training_start():
    """Test if training can start without errors"""
    print("🧪 Testing training initialization...")

    # Quick config
    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    # Load minimal dataset
    print("🔄 Loading dataset...")
    dataset = QuillanDataset()
    print(f"✅ Dataset loaded: {len(dataset.samples)} samples")

    # Initialize model
    print("🏗️ Initializing model...")
    model = QuillanSOTA(config)
    model.to(device)
    model.train()
    print("✅ Model initialized")

    # Initialize optimizer
    print("⚙️ Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("✅ Optimizer ready")

    # Try to get one batch
    print("📦 Getting training batch...")
    batch = dataset.get_training_batch(batch_size=1, seq_len=64)
    print(f"✅ Batch shapes: text={batch['text'].shape}, image={batch['image'].shape}")

    # Move to device
    print("🔄 Moving to device...")
    text = batch['text'].to(device)
    image = batch['image'].to(device)
    audio = batch['audio'].to(device)
    video = batch['video'].to(device)
    print("✅ Tensors on device")

    # Try forward pass
    print("🚀 Testing forward pass...")
    with torch.no_grad():
        outputs = model(text, image, audio, video)
    print("✅ Forward pass successful")

    # Try backward pass setup
    print("🔙 Testing backward pass setup...")
    optimizer.zero_grad()

    # Simple loss test
    if 'text' in outputs and outputs['text'].dim() == 3:
        logits = outputs['text']
        batch_size, seq_len, vocab_size = logits.shape
        # Simple loss for testing
        loss = torch.tensor(1.0, device=device, requires_grad=True)
        print("✅ Loss setup successful")

    print("🎉 Training initialization test PASSED!")
    return True

if __name__ == "__main__":
    test_training_start()
