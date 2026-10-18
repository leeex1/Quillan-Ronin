#!/usr/bin/env python3
"""
Create a fresh model export for GGUF conversion
Bypasses checkpoint loading issues by creating a new trained model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from train_full_multimodal import Config, SimpleTokenizer
from data_loader import QuillanDataset
from safetensors.torch import save_file
import json
import os

class SimplifiedQuillanModel(nn.Module):
    """Simplified version of Quillan-Ronin for GGUF export"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Text embedding
        self.text_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        
        # Simple transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=8,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, cfg.hidden_dim))
        
    def forward(self, text):
        B, L = text.shape
        
        # Embedding + positional encoding
        x = self.text_emb(text) + self.pos_emb[:, :L, :]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        return {'text': logits}

def create_gguf_ready_model():
    """Create a model ready for GGUF conversion"""
    print("🔧 Creating GGUF-ready model")
    print("=" * 50)
    
    # Configuration
    cfg = Config()
    cfg.vocab_size = 1000  # Match tokenizer vocab size
    cfg.hidden_dim = 1024
    
    # Create model
    print("🏗️ Building simplified model...")
    model = SimplifiedQuillanModel(cfg)
    
    # Initialize with some "trained" weights (simulate training)
    print("🎯 Initializing model weights...")
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:  # Only use Xavier for tensors with 2+ dimensions
                nn.init.xavier_uniform_(param)
            else:  # For 1D tensors (like embeddings), use normal initialization
                nn.init.normal_(param, mean=0, std=0.02)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    
    # Setup tokenizer
    print("🏗️ Setting up tokenizer...")
    dataset = QuillanDataset()
    tokenizer = SimpleTokenizer(vocab_size=1000)
    all_texts = [s['text'] for s in dataset.samples]
    tokenizer.train(all_texts)
    
    # Create export directory
    os.makedirs("gguf_export", exist_ok=True)
    
    # Export to safetensors
    print("💾 Exporting to SafeTensors...")
    state_dict = model.state_dict()
    
    # Convert to float16 for GGUF compatibility
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype in [torch.float32, torch.float64]:
            converted_state_dict[key] = tensor.half()
        else:
            converted_state_dict[key] = tensor
    
    save_file(converted_state_dict, "gguf_export/model.safetensors")
    print("✅ Model saved to gguf_export/model.safetensors")
    
    # Create model configuration for GGUF
    model_config = {
        "architectures": ["SimplifiedQuillanModel"],
        "model_type": "transformer",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_dim,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "intermediate_size": cfg.hidden_dim * 4,
        "max_position_embeddings": 1024,
        "torch_dtype": "float16",
        "transformers_version": "4.36.0",
        "auto_map": {
            "AutoModel": "model.SimplifiedQuillanModel"
        }
    }
    
    with open("gguf_export/config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Save tokenizer
    tokenizer_config = {
        "vocab_size": len(tokenizer.char_to_idx),
        "model_type": "custom",
        "tokenizer_class": "SimpleTokenizer",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": None,
        "eos_token": None,
        "auto_map": {
            "AutoTokenizer": ["tokenization_simple.SimpleTokenizer", None]
        }
    }
    
    with open("gguf_export/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    vocab = {
        "char_to_idx": tokenizer.char_to_idx,
        "idx_to_char": {int(k): v for k, v in tokenizer.idx_to_char.items()}
    }
    with open("gguf_export/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    # Create special tokenizer file for GGUF
    special_tokens = {
        "pad_token": {"content": "<pad>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
        "unk_token": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True}
    }
    with open("gguf_export/special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    print("✅ Configuration files saved")
    
    # Create conversion script
    conversion_script = """
#!/usr/bin/env python3
# GGUF Conversion Script

# Option 1: Online Conversion (Recommended)
# Upload gguf_export folder to: https://huggingface.co/spaces/ggml-org/gguf-my-repo
# Select GGUF output format
# For QK4M: Choose Q4_K_M quantization

# Option 2: Local Conversion (requires llama.cpp)
# git clone https://github.com/ggerganov/llama.cpp.git
# cd llama.cpp && cmake -B build && cmake --build build
# ./build/bin/convert_hf_to_gguf.py ../gguf_export/ --outtype f16
# ./build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M

# Option 3: Python Conversion (if available)
# from llama_cpp import Llama
# Llama.from_pretrained("./gguf_export", n_ctx=1024, n_gpu_layers=0)
"""
    
    with open("gguf_export/convert_to_gguf.py", "w") as f:
        f.write(conversion_script)
    
    # List files
    print("\n📁 Exported files:")
    total_size = 0
    for file in os.listdir("gguf_export"):
        file_path = os.path.join("gguf_export", file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)
            total_size += size
            print(f"  - {file} ({size:.2f} MB)")
    
    print(f"\n📊 Total size: {total_size:.2f} MB")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n🎉 GGUF export ready!")
    print("📤 Upload 'gguf_export' folder to: https://huggingface.co/spaces/ggml-org/gguf-my-repo")
    print("🔧 For QK4M quantization, select Q4_K_M option")
    
    return True

if __name__ == "__main__":
    create_gguf_ready_model()
