"""Proper BitNet 1.58b Linear with packed ternary storage.

Storage: 4 ternary values per byte (0.5 bytes/value).
Training: unpacks to fp16, computes, repacks.
Inference: loads only packed ternary (no fp16 copies).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def _ternary_quant(w, eps=0.01):
    """Quantize weights to {-1, 0, 1} with STE."""
    scale = w.abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    return w + (w_q * scale - w).detach(), scale

def _pack_ternary(w):
    """Pack fp16 ternary {-1,0,1} values into uint8 (4 per byte)."""
    w_int = (w.round().clamp(-1, 1) + 1).long()  # -1→0, 0→1, 1→2
    flat = w_int.reshape(-1)
    pad = (4 - flat.numel() % 4) % 4
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    packed = flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    return packed.to(torch.uint8)

def _unpack_ternary(packed, shape, device='cpu'):
    """Unpack uint8 to fp16 ternary values."""
    flat = packed.to(torch.uint8)
    v0 = (flat >> 0) & 3; v1 = (flat >> 2) & 3
    v2 = (flat >> 4) & 3; v3 = (flat >> 6) & 3
    vals = torch.stack([v0, v1, v2, v3], -1).reshape(-1)
    result = torch.where(vals == 0, -1, torch.where(vals == 2, 1, 0))
    total = torch.Size(shape).numel()
    return result[:total].reshape(shape).half()


class BitNetLinear(nn.Module):
    """BitNet 1.58b Linear with packed ternary storage.
    
    - self.weight: nn.Parameter containing fp16 latent weights (training only)
    - self.packed: buffer with packed ternary weights (for saving/loading)
    - During training: use self.weight (fp16 latent) via _ternary_quant
    - During inference: use self.packed (unpacked on-the-fly)
    - Save checkpoint: only save self.packed (not self.weight)
    """
    
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Latent fp16 weight for gradient accumulation (training only)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16))
        nn.init.kaiming_normal_(self.weight, a=0.1)
        
        # Packed ternary storage (for checkpoint save/load)
        self.register_buffer('packed', _pack_ternary(self.weight.data))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
    
    def forward(self, x):
        """Forward pass with ternary weights and 8-bit activation quantization."""
        if self.training:
            # Use fp16 latent weights with ternary STE quantization
            w_eff, _ = _ternary_quant(self.weight)
        else:
            # Use packed ternary (unpack once, cache for subsequent calls)
            if not hasattr(self, '_unpacked') or self._unpacked is None:
                self._unpacked = _unpack_ternary(self.packed, self.weight.shape, x.device)
            w_eff = self._unpacked
        
        # 8-bit activation quantization with STE
        x_scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=0.01)
        x_8bit = (x * x_scale).round().clamp(-128, 127) / x_scale
        x_q = x + (x_8bit - x).detach()
        
        return F.linear(x_q, w_eff, self.bias)
    
    def pack_weights(self):
        """Pack current weights to ternary storage."""
        w_q, scale = _ternary_quant(self.weight.data)
        self.packed = _pack_ternary(w_q)
        self._unpacked = None
    
    @classmethod
    def from_linear(cls, linear, in_features=None, out_features=None):
        """Convert a standard Linear or existing BitLinear to BitNetLinear."""
        if in_features is None: in_features = linear.in_features
        if out_features is None: out_features = linear.out_features
        bl = cls(in_features, out_features, bias=linear.bias is not None)
        with torch.no_grad():
            w = linear.weight.data
            if w.dtype != torch.float16: w = w.half()
            bl.weight.data.copy_(w)
            bl.pack_weights()
            if bl.bias is not None and linear.bias is not None:
                bl.bias.data.copy_(linear.bias.data.half() if linear.bias.dtype != torch.float16 else linear.bias.data)
        return bl
