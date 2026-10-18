#!/usr/bin/env python3
"""
Resize Quillan-Ronin model embedding layers from 50,257 to 128,256 vocab.
Preserves training progress while fixing tokenizer mismatch.
"""
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / '_dev'))

def resize_checkpoint(ckpt_path, output_path, old_vocab=50257, new_vocab=128256):
    """Resize embedding layers in checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt
    
    print(f"Original vocab size: {old_vocab}")
    print(f"Target vocab size: {new_vocab}")
    
    # Resize embedding layers
    keys_to_resize = [
        'txt_dec.weight',
        'ingestion.txt_emb.weight',
        'txt_dec.lora_B'  # LoRA adapter if present
    ]
    
    resized_count = 0
    for key in keys_to_resize:
        if key in state:
            old_tensor = state[key]
            old_shape = old_tensor.shape
            
            # Determine which dimension to resize
            if key == 'txt_dec.lora_B':
                # LoRA B matrix: (rank, vocab_size)
                if old_shape[1] == old_vocab:
                    new_shape = (old_shape[0], new_vocab)
                    new_tensor = torch.randn(new_shape, dtype=old_tensor.dtype)
                    new_tensor[:, :old_vocab] = old_tensor
                    state[key] = new_tensor
                    print(f"Resized {key}: {old_shape} -> {new_shape}")
                    resized_count += 1
            else:
                # Embedding matrices: (vocab_size, hidden_dim)
                if old_shape[0] == old_vocab:
                    new_shape = (new_vocab, old_shape[1])
                    new_tensor = torch.randn(new_shape, dtype=old_tensor.dtype)
                    # Preserve existing weights
                    new_tensor[:old_vocab] = old_tensor
                    # Initialize new tokens with Xavier/Glorot
                    torch.nn.init.xavier_uniform_(new_tensor[old_vocab:])
                    state[key] = new_tensor
                    print(f"Resized {key}: {old_shape} -> {new_shape}")
                    resized_count += 1
    
    print(f"Resized {resized_count} embedding layers")
    
    # Save resized checkpoint
    print(f"Saving resized checkpoint: {output_path}")
    torch.save(ckpt, output_path)
    print("Done!")
    
    return resized_count

if __name__ == '__main__':
    # Paths
    ckpt_dir = ROOT / 'checkpoints_sft'
    input_ckpt = ckpt_dir / 'quillan_sft_latest.pt'
    output_ckpt = ckpt_dir / 'quillan_sft_resized.pt'
    
    # Backup original
    backup_ckpt = ckpt_dir / 'quillan_sft_latest_backup.pt'
    if not backup_ckpt.exists():
        print(f"Creating backup: {backup_ckpt}")
        import shutil
        shutil.copy(input_ckpt, backup_ckpt)
    
    # Resize
    resize_checkpoint(input_ckpt, output_ckpt)
    
    print(f"\nOriginal checkpoint backed up to: {backup_ckpt}")
    print(f"Resized checkpoint saved to: {output_ckpt}")
    print(f"\nTo use resized checkpoint, update training script to load: {output_ckpt}")
