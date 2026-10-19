#!/usr/bin/env python3
"""
Export Quillan-Ronin model to SafeTensors format for GGUF conversion
"""

import torch
from safetensors.torch import save_file
from train_full_multimodal import QuillanRoninV5_3, Config, SimpleTokenizer
from data_loader import QuillanDataset
import json
import os

def export_model_for_gguf():
    """Export the trained model in a format suitable for GGUF conversion"""
    print("🚀 Exporting Quillan-Ronin model for GGUF conversion")
    print("=" * 60)

    # Load model
    print("🔄 Loading trained model...")
    cfg = Config()
    model = QuillanRoninV5_3(cfg)

    checkpoint = torch.load("best_multimodal_quillan.pt", map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create model directory
    os.makedirs("gguf_export", exist_ok=True)

    # Export model weights to safetensors
    print("💾 Exporting model weights to SafeTensors format...")

    # Get state dict and convert to safetensors format
    state_dict = model.state_dict()

    # Convert tensors to float16 for better compatibility (GGUF typically uses f16/f32)
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            converted_state_dict[key] = tensor.half()  # Convert to float16
        else:
            converted_state_dict[key] = tensor

    # Save as safetensors
    save_file(converted_state_dict, "gguf_export/model.safetensors")
    print("✅ Model weights saved to gguf_export/model.safetensors")

    # Export tokenizer
    print("🏗️ Exporting tokenizer...")
    dataset = QuillanDataset()
    tokenizer = SimpleTokenizer(vocab_size=1000)
    all_texts = [s['text'] for s in dataset.samples]
    tokenizer.train(all_texts)

    # Create tokenizer config
    tokenizer_config = {
        "vocab_size": len(tokenizer.char_to_idx),
        "model_type": "custom_multimodal",
        "tokenizer_class": "SimpleTokenizer",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": None,
        "eos_token": None,
    }

    # Save tokenizer config
    with open("gguf_export/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Save vocabulary
    vocab = {
        "char_to_idx": tokenizer.char_to_idx,
        "idx_to_char": {int(k): v for k, v in tokenizer.idx_to_char.items()}
    }
    with open("gguf_export/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print("✅ Tokenizer exported to gguf_export/")

    # Create model config for GGUF conversion
    model_config = {
        "architectures": ["QuillanRoninV5_3"],
        "model_type": "custom_multimodal",
        "vocab_size": len(tokenizer.char_to_idx),
        "hidden_size": cfg.hidden_dim,
        "num_hidden_layers": 4,  # Based on diffusion layers
        "num_attention_heads": 8,  # Estimated
        "intermediate_size": cfg.hidden_dim * 4,
        "max_position_embeddings": 1024,
        "num_experts": cfg.num_experts,
        "expert_capacity": cfg.expert_capacity,
        "torch_dtype": "float16",
        "transformers_version": "4.36.0"
    }

    with open("gguf_export/config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print("✅ Model config saved to gguf_export/config.json")

    # Create conversion instructions
    instructions = """
# GGUF Conversion Instructions

## Files Created:
- model.safetensors: Model weights in SafeTensors format
- config.json: Model configuration
- tokenizer_config.json: Tokenizer configuration
- vocab.json: Vocabulary mapping

## Online Conversion Tools:
1. https://huggingface.co/spaces/ggml-org/gguf-my-repo
2. https://convert.aitoolkit.org/
3. https://www.koboldai.com/gguf

## Steps:
1. Upload the model.safetensors file
2. Upload the config.json file
3. Select output format: GGUF
4. For QK4M quantization, choose 4-bit quantization
5. Download the resulting .gguf file

## Alternative: Local Conversion (if llama.cpp works)
```bash
# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py gguf_export/ --outtype f16

# Quantize to Q4_K_M
llama.cpp/build/bin/Release/llama-quantize.exe model.gguf model-Q4_K_M.gguf Q4_K_M
```
"""

    with open("gguf_export/README.md", "w") as f:
        f.write(instructions)

    print("✅ Conversion instructions saved to gguf_export/README.md")

    # Create file list
    print("\n📁 Exported files:")
    for file in os.listdir("gguf_export"):
        file_path = os.path.join("gguf_export", file)
        size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  - {file} ({size:.2f} MB)")

    print("\n🎉 Export complete!")
    print("📤 Upload the 'gguf_export' folder to an online GGUF converter")
    print("🔗 Recommended: https://huggingface.co/spaces/ggml-org/gguf-my-repo")
    print("📊 For QK4M quantization, select 4-bit quantization option")

if __name__ == "__main__":
    export_model_for_gguf()
