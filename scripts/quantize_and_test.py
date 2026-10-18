#!/usr/bin/env python3
"""
Quillan-Ronin Quantized Export & Inference Test
Packs model weights to 1.58-bit ternary and tests GPU inference.
"""
import os, sys, torch, gc, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig, _weight_quant
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device == 'cuda':
    cap = torch.cuda.get_device_properties(0)
    print(f'GPU: {cap.name}, VRAM: {cap.total_memory/1e9:.1f}GB')

def export_quantized(model, out_path='checkpoints/quillan_quantized.pt'):
    """Export model with all BitLinear weights pre-quantized to ternary.
    This replaces full fp16 weights with tiny ternary buffers."""
    
    quantized_state = {}
    
    for name, param in model.named_parameters():
        data = param.data
        
        # For BitLinear weights (any large linear layer), store quantized ternary
        if 'weight' in name and data.dim() >= 2 and data.numel() > 10000:
            # Quantize to ternary {-1, 0, +1}
            scale = data.abs().mean().clamp(min=0.01)
            w_ternary = torch.round(torch.clamp(data / scale, -1.0, 1.0))
            # Pack 4 values per byte (2 bits each)
            w_shifted = (w_ternary + 1).long().clamp(0, 2)  # {-1,0,1} -> {0,1,2}
            flat = w_shifted.reshape(-1)
            pad = (4 - flat.numel() % 4) % 4
            if pad:
                flat = torch.cat([flat, torch.zeros(pad, dtype=torch.long, device=data.device)])
            packed = flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
            
            quantized_state[name] = packed.to(torch.uint8).cpu()
            quantized_state[name + '_scale'] = torch.tensor([scale], dtype=torch.float16)
            quantized_state[name + '_shape'] = torch.tensor(data.shape, dtype=torch.int64)
        else:
            # Small params (biases, norms, embeddings) keep full precision
            quantized_state[name] = data.cpu().half()
    
    # Save
    torch.save(quantized_state, out_path)
    gb = os.path.getsize(out_path) / 1e9
    print(f'Exported quantized model: {out_path} ({gb:.2f} GB)')
    return out_path

def load_quantized(model_class, config, ckpt_path, map_location='cpu'):
    """Load quantized checkpoint back into model."""
    state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    
    model = model_class(config)
    model_state = {}
    
    for name, param in model.named_parameters():
        if name + '_scale' in state:
            # Unpack ternary
            packed = state[name]
            scale = state[name + '_scale'].item()
            shape = tuple(state[name + '_shape'].tolist())
            
            # Unpack
            flat = packed.to(torch.uint8)
            v0 = (flat >> 0) & 3
            v1 = (flat >> 2) & 3
            v2 = (flat >> 4) & 3
            v3 = (flat >> 6) & 3
            interleaved = torch.stack([v0, v1, v2, v3], dim=-1).reshape(-1)
            interleaved = interleaved[:shape[0] * shape[1]]
            w_ternary = interleaved.float() - 1.0  # {0,1,2} -> {-1,0,1}
            
            model_state[name] = (w_ternary * scale).reshape(shape).half()
        elif name in state:
            model_state[name] = state[name].to(param.dtype)
    
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f'Loaded quantized: Missing={len(missing)}, Unexpected={len(unexpected)}')
    return model

def test_inference(model, device='cuda', num_tokens=5):
    """Test forward pass speed."""
    model.to(device)
    model.eval()
    
    inp = torch.randint(0, 100, (1, num_tokens), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(inp)
    
    # Timed
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(inp)
    torch.cuda.synchronize()
    t = time.time() - t0
    
    logits = out['logits']
    print(f'Inference: {num_tokens} tokens in {t:.2f}s ({num_tokens/t:.1f} tok/s)')
    print(f'Logits: {logits.shape}, range [{logits.min().item():.2f}, {logits.max().item():.2f}]')
    print(f'VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB')
    return logits

if __name__ == '__main__':
    # Load model
    cfg = QuillanArchConfig(device=device, pascal_mode=(device=='cuda' and torch.cuda.get_device_capability()[0] < 7), text_only=True, low_mem=True)
    model = QuillanRoninSovereign(cfg)
    
    # Try to load latest trained checkpoint, fall back to router_trained
    ckpt_path = 'checkpoints/model_final.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/router_trained.pt'
    
    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    
    # Test inference before quantization
    print('\n--- FP16 Inference ---')
    model.half()
    test_inference(model, device)
    
    # Export quantized
    print('\n--- Quantizing ---')
    qpath = export_quantized(model)
    
    # Load quantized and test
    print('\n--- Quantized Inference ---')
    qmodel = load_quantized(QuillanRoninSovereign, cfg, qpath)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    test_inference(qmodel, device)
    
    # Compare sizes
    fp16_size = os.path.getsize(ckpt_path) / 1e9
    q_size = os.path.getsize(qpath) / 1e9
    print(f'\nFP16: {fp16_size:.2f} GB -> Quantized: {q_size:.2f} GB ({q_size/fp16_size*100:.0f}% of original)')
    print('Done!')
