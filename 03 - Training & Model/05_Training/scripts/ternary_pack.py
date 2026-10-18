"""Ternary weight packing for BitNet 1.58b.

Packs 4 ternary values (-1, 0, 1) per byte (2 bits each).
Storage: 0.5 bytes/value vs 2 bytes/value for fp16 = 4x compression.
"""
import torch

# Map: 2-bit code -> ternary value
UNPACK_MAP = {0: -1, 1: 0, 2: 1, 3: 0}  # 3 is unused, treat as 0

def pack_ternary(w: torch.Tensor) -> torch.Tensor:
    """Pack a fp16 tensor of ternary {-1,0,1} values into uint8 (4 values per byte).
    
    Args:
        w: Tensor of dtype fp16/bf16/fp32 with values in {-1, 0, 1}
    Returns:
        uint8 tensor with 4 values packed per element (same number of elements / 4)
    """
    assert w.dtype in (torch.float16, torch.bfloat16, torch.float32)
    # Round to nearest ternary
    w_ternary = torch.round(torch.clamp(w, -1.0, 1.0)).long()
    w_ternary = w_ternary + 1  # shift to {0, 1, 2}
    w_ternary = torch.clamp(w_ternary, 0, 2)  # safety
    flat = w_ternary.reshape(-1)
    # Pad to multiple of 4
    pad = (4 - flat.numel() % 4) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.long, device=w.device)])
    # Pack: value0 | value1<<2 | value2<<4 | value3<<6
    packed = flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    return packed.to(torch.uint8)


def unpack_ternary(packed: torch.Tensor, original_shape, device='cpu') -> torch.Tensor:
    """Unpack uint8 tensor back to fp16 ternary values.
    
    Args:
        packed: uint8 tensor of packed ternary values
        original_shape: shape to reshape into
    Returns:
        fp16 tensor with values in {-1, 0, 1}
    """
    flat = packed.to(torch.uint8)
    # Unpack: 4 values per byte
    v0 = (flat >> 0) & 3
    v1 = (flat >> 2) & 3
    v2 = (flat >> 4) & 3
    v3 = (flat >> 6) & 3
    # Interleave
    interleaved = torch.stack([v0, v1, v2, v3], dim=-1).reshape(-1)
    # Shift back to {-1, 0, 1}: 0->-1, 1->0, 2->1, 3->0
    result = torch.where(interleaved == 0, -1, torch.where(interleaved == 2, 1, 0))
    # Trim padding and reshape
    total = original_shape.numel() if hasattr(original_shape, 'numel') else torch.Size(original_shape).numel()
    result = result[:total].reshape(original_shape)
    return result.half()


def quantize_and_pack(w: torch.Tensor) -> tuple:
    """Quantize weight to ternary and pack it. Returns (packed, shape, scale)."""
    scale = w.abs().mean().clamp(min=0.01).item()
    w_scaled = w.float() / scale
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    packed = pack_ternary(w_q)
    return packed, w.shape, scale


def unpack_and_dequantize(packed, shape, scale, device='cpu'):
    """Unpack and rescale ternary weights. Returns fp16 tensor."""
    w = unpack_ternary(packed, shape, device)
    return (w * scale).half()


# ─── Model-level helpers ────────────────────────────────────────────────

def pack_model_state(state_dict: dict) -> dict:
    """Pack all eligible weight matrices in a state dict.
    Skips: biases, norms, embeddings, buffers, quant caches, LoRA adapters.
    """
    packed = {}
    for k, v in state_dict.items():
        # Skip tiny tensors, biases, norms, embeddings, buffers
        if v.dim() < 2 or v.numel() < 4096:
            packed[k] = v
            continue
        # Skip embeddings (large vocab, dense gradients needed)
        if 'emb' in k or 'embed' in k:
            packed[k] = v
            continue
        # Skip norms, biases, buffers
        if v.dim() == 1 or 'norm' in k or 'bias' in k or 'buffer' in k or '_quant_' in k:
            packed[k] = v
            continue
        # Skip LoRA (small, already fp32)
        if 'lora' in k:
            packed[k] = v
            continue
        # Pack everything else (linear weights)
        try:
            p, shape, scale = quantize_and_pack(v)
            packed[k] = p
            packed[k.replace('weight', 'weight_scale')] = torch.tensor([scale], dtype=torch.float16)
            # CRITICAL FIX: Save original shape for unpacking
            packed[k.replace('weight', 'weight_orig_shape')] = torch.tensor(list(shape), dtype=torch.int32)
        except Exception as e:
            packed[k] = v  # fallback
    return packed


def unpack_model_state(packed_state: dict, device='cpu') -> dict:
    """Unpack a packed state dict back to fp16 tensors."""
    state = {}
    for k, v in packed_state.items():
        if v.dtype == torch.uint8:
            shape_key = k.replace('weight', 'weight_orig_shape')
            scale_key = k.replace('weight', 'weight_scale')
            shape = packed_state.get(shape_key, None)
            scale = packed_state.get(scale_key, torch.tensor([1.0]))
            if shape is not None:
                state[k] = unpack_and_dequantize(v, tuple(shape.tolist()), scale.item(), device)
            else:
                state[k] = v
        else:
            state[k] = v
    return state
