#!/usr/bin/env python3
# FlashAttention-3 (2407.08608) wrapper for Quillan Couil hybrid heads
import torch
try:
    from flash_attn import flash_attn_func
    HAS_FA = True
except ImportError:
    HAS_FA = False

def quillan_flash_attn(q, k, v, is_causal=True):
    """IO-aware exact attention: uses FlashAttention-3 kernel if available, else SDPA fallback"""
    if HAS_FA and q.is_cuda:
        # FA3 async + low-precision path (2407.08608)
        return flash_attn_func(q, k, v, causal=is_causal)
    else:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
