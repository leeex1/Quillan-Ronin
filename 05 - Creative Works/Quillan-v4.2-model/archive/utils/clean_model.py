#!/usr/bin/env python3
"""
Remove incompatible tensors from the model for GGUF conversion
"""

import os
from safetensors.torch import load_file, save_file

def remove_incompatible_tensors():
    """Remove tensors that are not compatible with Llama GGUF format"""
    print("🔄 Removing incompatible tensors for GGUF conversion")
    print("=" * 60)

    # Load the current model
    model_path = "llama_gguf_export/model.safetensors"
    if not os.path.exists(model_path):
        print("❌ Model file not found")
        return False

    print(f"📂 Loading model from {model_path}")
    state_dict = load_file(model_path)

    print(f"📊 Found {len(state_dict)} tensors")

    # Tensors to remove (not compatible with Llama)
    tensors_to_remove = [
        'position_embeddings',  # Llama uses rotary embeddings, not absolute position embeddings
        # Add any other incompatible tensors here
    ]

    # Remove incompatible tensors
    cleaned_state_dict = {}
    removed_count = 0

    for name, tensor in state_dict.items():
        if any(remove_name in name for remove_name in tensors_to_remove):
            print(f"🗑️ Removing incompatible tensor: {name}")
            removed_count += 1
        else:
            cleaned_state_dict[name] = tensor

    print(f"✅ Removed {removed_count} incompatible tensors")
    print(f"📊 Remaining tensors: {len(cleaned_state_dict)}")

    # Save cleaned model
    cleaned_path = "llama_gguf_export/model_clean.safetensors"
    save_file(cleaned_state_dict, cleaned_path)
    print(f"✅ Cleaned model saved to {cleaned_path}")

    # Update model file path in config if needed
    # For now, we'll rename the cleaned file to replace the original
    if os.path.exists(model_path):
        os.remove(model_path)  # Remove the original file first
    os.rename(cleaned_path, model_path)
    print("✅ Replaced original model with cleaned version")

    total_params = sum(tensor.numel() for tensor in cleaned_state_dict.values())
    print(f"📊 Final parameters: {total_params:,}")

    return True

if __name__ == "__main__":
    remove_incompatible_tensors()
