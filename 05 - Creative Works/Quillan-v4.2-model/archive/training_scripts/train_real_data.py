#!/usr/bin/env python3
"""
Real training script for Quillan using your actual datasets
"""

import torch
import os
from data_loader import QuillanDataset
from __init__ import QuillanSOTA, RLConfig, GRPOTrainer, Config
import json

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Enhanced checkpoint saving with metadata"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'hidden_dim': 1024,
            'num_experts': 8,
            'vocab_size': 50000
        }
    }
    torch.save(checkpoint, path)
    print(f"💾 Saved checkpoint to {path} (loss: {loss:.4f})")

def train_real_data():
    """Train on your actual datasets"""
    print("🚀 Starting Quillan training on real data...")
    
    # Configuration
    config = RLConfig(
        learning_rate=1e-4,  # Lower learning rate for real data
        batch_size=1,  # Very small batch size to prevent memory issues
        num_trajectories=4,
        max_trajectory_len=128,  # Shorter sequences to save memory
        clip_epsilon=0.2,
        num_epochs=300,  # Extended training: 300 epochs to improve performance
        warmup_steps=100
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Training on {device}")
    
    # Initialize model
    model_config = Config()
    model = QuillanSOTA(model_config)
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Load dataset
    dataset = QuillanDataset()
    stats = dataset.get_dataset_stats()
    
    print(f"\n📊 Dataset loaded:")
    print(f"  📝 Text samples: {len(dataset.samples)}")
    print(f"  🖼️  Image files: {len(dataset.image_files)}")
    print(f"  🎵 Audio files: {len(dataset.audio_files)}")
    print(f"  🎬 Video files: {len(dataset.video_files)}")
    print(f"  📈 Total multimodal samples: {len(dataset.samples) + len(dataset.image_files) + len(dataset.audio_files) + len(dataset.video_files)}")
    
    print(f"\n📚 Text sources:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count} samples")
    
    # Training loop
    print(f"\n🏃‍♂️ Starting training for {config.num_epochs} epochs...")
    print(f"📊 Training on {len(dataset.samples)} text samples + {len(dataset.image_files)} images + {len(dataset.audio_files)} audio + {len(dataset.video_files)} videos")
    
    best_loss = float('inf')
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 10  # Number of batches per epoch
        
        for batch_idx in range(num_batches):
            try:
                # Get training batch with real multimodal data
                batch = dataset.get_training_batch(
                    batch_size=config.batch_size,
                    seq_len=config.max_trajectory_len
                )
                
                # Move to device
                text = batch['text'].to(device)
                image = batch['image'].to(device)
                audio = batch['audio'].to(device)
                video = batch['video'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=torch.float16):
                    outputs = model(text, image, audio, video)
                    
                    # Handle different output formats
                    if isinstance(outputs, dict):
                        if 'text' in outputs:
                            logits = outputs['text']
                        elif 'logits' in outputs:
                            logits = outputs['logits']
                        else:
                            # Use first tensor value
                            logits = next(v for v in outputs.values() if isinstance(v, torch.Tensor))
                    else:
                        logits = outputs
                    
                    # Calculate loss (simplified for multimodal)
                    if logits.dim() == 3:  # [batch, seq, vocab]
                        # Text generation loss
                        batch_size, seq_len, vocab_size = logits.shape
                        # Use next token prediction
                        target = text[:, 1:seq_len]  # Shift by 1
                        pred = logits[:, :seq_len-1, :]  # Remove last
                        loss = torch.nn.functional.cross_entropy(
                            pred.reshape(-1, vocab_size),
                            target.reshape(-1),
                            ignore_index=0
                        )
                    else:
                        # Fallback loss
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Memory management - clear unused tensors
                del text, image, audio, video, outputs, logits, target, pred
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if batch_idx % 5 == 0:
                    print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}: Loss = {loss.item():.4f}")
                    
            except Exception as e:
                print(f"⚠️  Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()  # Show full traceback
                # Clear memory on error too
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        avg_loss = epoch_loss / num_batches
        print(f"📈 Epoch {epoch+1}/{config.num_epochs}: Average Loss = {avg_loss:.4f}")
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, "checkpoints/quillan_best.pt")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, f"checkpoints/quillan_epoch_{epoch+1}.pt")
    
    # Save final model
    save_checkpoint(model, optimizer, config.num_epochs-1, avg_loss, "checkpoints/quillan_final_real.pt")
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best loss achieved: {best_loss:.4f}")
    print(f"💾 Final model saved to checkpoints/quillan_final_real.pt")

if __name__ == "__main__":
    train_real_data()
