import torch
import argparse
from __init__ import QuillanSOTA, Config
import os

def load_model(checkpoint_path: str, device: str = 'cuda'):
    print(f"Loading model from {checkpoint_path}...")
    # Initialize model structure
    config = Config()
    model = QuillanSOTA(config)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()
    return model

def generate_text(model, prompt_text, max_new_tokens=50, temperature=0.7, device='cuda'):
    """Simple text generation for demonstration"""
    print(f"Generating text with prompt: '{prompt_text}'")
    
    # Mock inputs with correct dimensions based on model expectations
    # The model uses patch_size=16, so we need dimensions that work with this
    batch_size = 1
    seq_len = 10
    
    # Calculate correct image dimensions
    patch_size = 16  # From Config
    # Model expects 65536 patches: sqrt(65536) = 256, so we need 256*16 = 4096x4096 image
    grid_size = 256
    img_size = grid_size * patch_size  # 4096x4096
    
    text_input = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    img_input = torch.randn(batch_size, 3, img_size, img_size).to(device)  # 4096x4096 for image (256x256 patches = 65536)
    aud_input = torch.randn(batch_size, 1, 1024).to(device)     # Audio length
    vid_input = torch.randn(batch_size, 3, 16, img_size, img_size).to(device)  # Video: 16 frames, 4096x4096
    
    with torch.no_grad():
        # Forward pass through the model
        outputs = model(text_input, img_input, aud_input, vid_input)
        
        # The model returns a dictionary with different modality outputs
        if isinstance(outputs, dict):
            text_logits = outputs.get('text', outputs.get('logits', None))
        else:
            text_logits = outputs
        
        # Simple sampling
        if text_logits is not None and isinstance(text_logits, torch.Tensor):
            if text_logits.dim() == 3:
                # Take the last token's logits
                last_logits = text_logits[0, -1, :]
                
                # Apply temperature
                scaled_logits = last_logits / temperature
                
                # Sample
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                return [next_token.item()]
            else:
                return [text_logits.argmax().item()]
        else:
            # Fallback: return a mock token
            return [42]  # Mock token ID

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/quillan_final.pt", help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Input prompt") # In real usage, need tokenizer
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)
    
    # Generate text
    output_ids = generate_text(model, args.prompt, args.max_tokens, args.temperature, device)
    print(f"Generated token IDs: {output_ids}")
    print("Note: This is a demonstration with mock multimodal inputs.")
    print("For actual text generation, you'll need to implement proper tokenization and handle the multimodal outputs.")
