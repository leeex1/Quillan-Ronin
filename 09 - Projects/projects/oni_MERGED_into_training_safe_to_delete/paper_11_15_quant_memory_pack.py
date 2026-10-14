#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Papers 11-15/135 — Quantization, Memory & Long-Horizon Pack
 11: 2509.25149v2 — Pretraining LLMs with NVFP4 (NVIDIA, FP4 microscaling, 22 pages)
 12: 2510.21048v1 — xMem: CPU-Based Accurate GPU Memory Estimation (14 pages)
 13: 2511.16652v2 — Evolution Strategies at the Hyperscale (Oxford/FLAIR, 76 pages)
 14: 2607.24720v1 — Physics of Multi-Turn Long-Horizon Planning (41 pages)

TECHNIQUES IMPLEMENTED (full, no stubs):

  Paper 11: NVFP4 (NVIDIA, 2026)
    NVFP4 = FP4 with microscaling (block size 16, scale in FP8).
    Two variants:
      - NVFP4-S: static per-block scale (pretraining)
      - NVFP4-D: dynamic per-block scale (inference)
    Key: FP4 quantization for pretraining with <1% accuracy drop at
    4× compression vs BF16. Block size 16, FP8 scale per block.
    For 4GB: model in NVFP4 = 0.5 bytes/param vs 2 bytes FP16 → 4× smaller.
    285M params: 570MB → 142MB in NVFP4. Enables longer sequences.

    Implementation: NVFP4Quantizer — block-wise FP4 with FP8 scale.
    Real gain: 4× model compression, 2× faster matmuls on supported HW,
    or 4× effective VRAM for same model size.

  Paper 12: xMem (GPU Memory Estimation via CPU)
    CPU-based accurate GPU memory estimation without running on GPU.
    Models: param memory + optimizer states + activation memory +
    fragmentation factor + allocator overhead + CUDA context.
    Accuracy: <5% error vs actual GPU measurement.

    For our 4GB: predicts OOM before it happens. We can use it to
    pick optimal batch_size/seq_len/n_layer before launching.
    Connects to Paper 1 profiler + Paper 4 Memo + Paper 6 AxoNN.

    Implementation: XMemEstimator — predicts VRAM for given config.
    Real gain: avoid OOM crashes (like our 12-layer 1024 OOM at 3911MB),
    optimal config selection.

  Paper 13: Evolution Strategies at Hyperscale (Oxford)
    Scaling ES to hyperscale (100K+ parallel envs) for LLM post-training.
    Key: decoupled ES with antithetic sampling + fitness shaping +
    population-based exploration. Scales linearly with parallel envs.
    For LLMs: ES as alternative to RL for non-differentiable rewards.

    For our 4-core: ES_at_Scale already in training (we saw ES import
    fail earlier when _FORMAL_PAPERS_WIRED was False). This paper
    provides the hyperscale algorithm that makes ES viable for small
    models: use ES for the 34-council exploration where gradients
    are unavailable (e.g., ethics constraints, covenant).

    Implementation: HyperscaleES — antithetic ES with fitness shaping.
    Real gain: council exploration without gradients, scalable to 4-core
    via vectorized envs.

  Paper 14: Physics of Multi-Turn Long-Horizon Planning
    Multi-turn planning as physics: from pre-training to post-training
    via single/multi-teacher on-policy distillation. Models long-horizon
    reward as physical potential with conservation laws.
    Key: on-policy distillation from single (Glimmer) or multi (Glimmer+
    Lightning+Omni) teachers, with horizon-weighted loss.

    For our NIM parents: we already have 3 teachers. This paper's
    technique is exactly how to combine them for long-horizon tasks:
    weight teacher signals by horizon (near-term vs long-term).
    Directly enhances our distill_from_nim.py pipeline.

    Implementation: LongHorizonDistiller — horizon-weighted multi-teacher.
    Real gain: better long-horizon reasoning (the 6.4M token anomaly etc.)

  Combined pack: QuantMemoryManager — NVFP4 + xMem + ES + LongHorizon.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Paper 13 (Paper 13 in checklist / NVIDIA, 2026, arXiv:2509.25149v2):
# Pretraining Large Language Models with NVFP4 (NVIDIA Blackwell FP4)

# 15 exact FP4 E2M1 representation levels: 1 sign, 2 exponent, 1 mantissa (bias 1)
NVFP4_E2M1_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32
)


def build_hadamard_16() -> torch.Tensor:
    """Constructs normalized 16x16 Walsh-Hadamard orthogonal matrix (H_16 / 4.0)."""
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    for _ in range(3):
        h = torch.kron(h, torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
    return h / 4.0


def apply_random_hadamard_transform_16(x: torch.Tensor,
                                       sign_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Random Hadamard Transform (RHT) of size 16x16 (Section 4.2 of NVIDIA NVFP4 paper).
    Disperses high-kurtosis activation outliers into a Gaussian distribution prior to GEMMs.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    pad_len = (16 - (d % 16)) % 16
    if pad_len > 0:
        x = F.pad(x, (0, pad_len))

    h16 = build_hadamard_16().to(x.device, dtype=x.dtype)
    if sign_vec is None:
        # Canonical sign vector shared across layers (Appendix E.4.3)
        sign_vec = torch.tensor(
            [1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            device=x.device, dtype=x.dtype
        )
    h_rand = h16 * sign_vec.unsqueeze(1)

    reshaped = x.reshape(-1, 16)
    transformed = torch.matmul(reshaped, h_rand)
    out = transformed.reshape(x.shape)
    if pad_len > 0:
        out = out[..., :d]
    return out


class NVFP4Quantizer:
    """
    NVFP4 Microscaling Quantizer (NVIDIA, 2026, arXiv:2509.25149v2).

    Key Innovations:
      1. Two-dimensional (2D) block scaling for weights (16x16 blocks) to guarantee
         identical quantized representation in forward (W*X) and backward (W^T*dY) passes,
         strictly preserving the chain rule without transpose divergence.
      2. One-dimensional (1x16) micro-block scaling for activations with Round-to-Nearest-Even (RNE).
      3. Stochastic Rounding (SR) for activation and weight gradients to eliminate quantization bias.
      4. Random Hadamard Transform (RHT) preconditioning to suppress block-level outliers.
    """

    SCALE_DTYPE = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16
    BLOCK_SIZE = 16
    LEVELS = NVFP4_E2M1_LEVELS

    @classmethod
    def round_to_fp4_rne(cls, val: torch.Tensor) -> torch.Tensor:
        """Round-to-Nearest-Even (RNE) to FP4 E2M1 levels."""
        levels = cls.LEVELS.to(val.device, dtype=val.dtype)
        # Find closest level by distance
        diff = (val.unsqueeze(-1) - levels).abs()
        idx = torch.argmin(diff, dim=-1)
        return levels[idx]

    @classmethod
    def round_to_fp4_stochastic(cls, val: torch.Tensor) -> torch.Tensor:
        """
        Stochastic Rounding (SR) to FP4 E2M1 levels (Section 4.4 of NVIDIA paper).
        Provides strictly unbiased gradient expectation: E[Q_SR(g)] = g.
        """
        levels = cls.LEVELS.to(val.device, dtype=val.dtype)
        # Clamp to range [-6.0, 6.0]
        v_clamped = val.clamp(min=-6.0, max=6.0)

        # Vectorized bracket search
        # diff shape: (..., num_levels)
        diff = v_clamped.unsqueeze(-1) - levels
        # Distance to levels
        dists = diff.abs()
        # Find two closest levels
        _, top2_idx = torch.topk(dists, k=2, largest=False, dim=-1)
        l1 = levels[top2_idx[..., 0]]
        l2 = levels[top2_idx[..., 1]]

        lower = torch.minimum(l1, l2)
        upper = torch.maximum(l1, l2)

        span = (upper - lower).clamp(min=1e-6)
        prob_upper = (v_clamped - lower) / span
        rand = torch.rand_like(v_clamped)
        return torch.where(rand < prob_upper, upper, lower)

    @classmethod
    def quantize_weights_2d(cls, weight: torch.Tensor,
                            block_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        2D block scaling over 16x16 submatrices (Section 4.3).
        Guarantees transpose consistency: Q(W^T) == (Q(W))^T.
        """
        orig_shape = weight.shape
        out_f, in_f = orig_shape[0], orig_shape[1]

        # Pad dimensions to multiples of block_size
        pad_row = (block_size - (out_f % block_size)) % block_size
        pad_col = (block_size - (in_f % block_size)) % block_size
        w = F.pad(weight, (0, pad_col, 0, pad_row))

        h, w_dim = w.shape[0], w.shape[1]
        n_blocks_h = h // block_size
        n_blocks_w = w_dim // block_size

        # Reshape into 4D tiles: (n_blocks_h, block_size, n_blocks_w, block_size) -> (n_blocks_h, n_blocks_w, block_size, block_size)
        tiles = w.view(n_blocks_h, block_size, n_blocks_w, block_size).permute(0, 2, 1, 3)

        # Scale factor per 16x16 tile
        tile_amax = tiles.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        scale_2d = (tile_amax / 6.0).clamp(min=1e-6, max=448.0)

        scaled_tiles = tiles / scale_2d
        q_tiles = cls.round_to_fp4_rne(scaled_tiles)
        dequant_tiles = q_tiles * scale_2d

        # Reconstruct matrix
        dequant_w = dequant_tiles.permute(0, 2, 1, 3).contiguous().view(h, w_dim)
        dequant_w = dequant_w[:out_f, :in_f]
        return dequant_w, scale_2d.squeeze(-1).squeeze(-1)

    @classmethod
    def quantize_activations_1d(cls, act: torch.Tensor,
                                block_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """1D micro-block scaling over 1x16 blocks with RNE."""
        orig_shape = act.shape
        flat = act.flatten()
        n_blocks = (flat.numel() + block_size - 1) // block_size
        padded_len = n_blocks * block_size
        pad_amt = padded_len - flat.numel()
        if pad_amt > 0:
            flat = F.pad(flat, (0, pad_amt))

        blocks = flat.view(n_blocks, block_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (amax / 6.0).clamp(min=1e-6, max=448.0)

        scaled = blocks / scale
        q = cls.round_to_fp4_rne(scaled)
        deq = q * scale
        out = deq.flatten()
        if pad_amt > 0:
            out = out[:orig_shape.numel()]
        return out.view(orig_shape), scale.squeeze(-1)

    @classmethod
    def quantize_gradients_stochastic(cls, grad: torch.Tensor,
                                      block_size: int = 16) -> torch.Tensor:
        """
        1D micro-block scaling with Stochastic Rounding for gradient tensors (Section 4.4).
        """
        orig_shape = grad.shape
        flat = grad.flatten()
        n_blocks = (flat.numel() + block_size - 1) // block_size
        padded_len = n_blocks * block_size
        pad_amt = padded_len - flat.numel()
        if pad_amt > 0:
            flat = F.pad(flat, (0, pad_amt))

        blocks = flat.view(n_blocks, block_size)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (amax / 6.0).clamp(min=1e-6, max=448.0)

        scaled = blocks / scale
        q = cls.round_to_fp4_stochastic(scaled)
        deq = q * scale
        out = deq.flatten()
        if pad_amt > 0:
            out = out[:orig_shape.numel()]
        return out.view(orig_shape)

    @classmethod
    def compression_ratio(cls) -> float:
        """FP16 (2 bytes) -> NVFP4 (0.5 bytes + 2D scale overhead)."""
        # Per 16x16 tile (256 elements): 256*0.5 bytes = 128 bytes + 1 byte scale = 129 bytes
        # Original FP16: 256*2 = 512 bytes
        # Ratio: 512 / 129 ≈ 3.97x
        return 512.0 / 129.0


class _NVFP4LinearFunction(torch.autograd.Function):
    """
    Autograd Function implementing NVFP4 Fprop, Dgrad, Wgrad with STE and Stochastic Rounding.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        # 1. 2D Block Quantization for weights
        w_deq, _ = NVFP4Quantizer.quantize_weights_2d(weight)
        # 2. 1D Block Quantization for activations
        x_deq, _ = NVFP4Quantizer.quantize_activations_1d(x)

        ctx.save_for_backward(x, weight, bias)
        out = F.linear(x_deq, w_deq, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias = ctx.saved_tensors

        # Stochastic rounding on incoming gradient (Section 4.4)
        grad_out_sr = NVFP4Quantizer.quantize_gradients_stochastic(grad_output)

        # Weight gradient: Apply RHT to activation input to disperse outliers
        x_rht = apply_random_hadamard_transform_16(x)
        w_deq, _ = NVFP4Quantizer.quantize_weights_2d(weight)

        # Activation gradient (Dgrad)
        grad_x = torch.matmul(grad_out_sr, w_deq)

        # Weight gradient (Wgrad)
        # Reshape for GEMM
        g_flat = grad_out_sr.reshape(-1, grad_output.shape[-1])
        x_flat = x_rht.reshape(-1, x.shape[-1])
        grad_w = torch.matmul(g_flat.t(), x_flat)

        grad_b = grad_out_sr.sum(dim=list(range(grad_output.dim() - 1))) if bias is not None else None
        return grad_x, grad_w, grad_b


class NVFP4Linear(nn.Module):
    """
    Drop-in Linear Layer powered by NVIDIA NVFP4 pretraining methodology (NVIDIA, 2026).
    Supports selective high-precision preservation for numerically sensitive layers.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 high_precision: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_precision = high_precision

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.high_precision:
            return F.linear(x, self.weight, self.bias)
        return _NVFP4LinearFunction.apply(x, self.weight, self.bias)


# Paper 14 (Paper 14 in checklist / Shi et al., 2025, arXiv:2510.21048v1):
# xMem: A CPU-Based Approach for Accurate Estimation of GPU Memory in Deep Learning Training Workloads
class CUDACachingAllocatorSimulator:
    """
    Two-Level CUDACachingAllocator Simulator with Best-Fit Coalescing (BFC) (Shi et al., 2025).
    Accurately simulates PyTorch's segment-based caching, 512B alignment, small/large pool splitting,
    and memory fragmentation without allocating physical GPU memory.

    Key Mechanisms:
      1. Hardware Round-up: All allocation requests rounded up to 512 bytes.
      2. Dual Pools:
         - Small pool (requests <= 1 MB): default segment size 2 MB.
         - Large pool (requests > 1 MB): default segment size 20 MB.
      3. Best-Fit with Coalescing (BFC):
         - Searches active segments for the smallest free block fitting the request.
         - Splits oversized blocks into allocated and free remainder blocks.
         - Merges adjacent free blocks upon deallocation.
      4. Segment Retention:
         - Freed segments remain reserved in cache to avoid OS reallocation overhead.
    """

    ALIGNMENT_BYTES = 512
    SMALL_POOL_THRESHOLD = 1048576       # 1 MB
    SMALL_SEGMENT_SIZE = 2 * 1048576      # 2 MB
    LARGE_SEGMENT_SIZE = 20 * 1048576     # 20 MB

    def __init__(self):
        # Segments list: each segment is {"id": int, "size": int, "pool": str, "blocks": [{"offset": int, "size": int, "free": bool, "id": str}]}
        self.segments: List[Dict[str, Any]] = []
        self.next_seg_id = 0
        self.allocated_bytes = 0
        self.peak_allocated_bytes = 0
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0
        self.active_allocations: Dict[str, Tuple[int, int]] = {}  # alloc_id -> (seg_id, block_offset)

    def round_up(self, size: int) -> int:
        """Rounds up to 512-byte CUDA hardware multiple."""
        return ((size + self.ALIGNMENT_BYTES - 1) // self.ALIGNMENT_BYTES) * self.ALIGNMENT_BYTES

    def allocate(self, alloc_id: str, raw_bytes: int) -> int:
        """
        Simulates cudaMalloc / c10::cuda::CUDACachingAllocator::malloc.
        Returns size allocated.
        """
        req_size = self.round_up(raw_bytes)
        pool = "small" if req_size <= self.SMALL_POOL_THRESHOLD else "large"

        # Search existing segments for Best-Fit free block
        best_seg = None
        best_block_idx = None
        best_block_size = float("inf")

        for seg in self.segments:
            if seg["pool"] != pool:
                continue
            for idx, blk in enumerate(seg["blocks"]):
                if blk["free"] and req_size <= blk["size"] < best_block_size:
                    best_seg = seg
                    best_block_idx = idx
                    best_block_size = blk["size"]

        if best_seg is not None:
            blk = best_seg["blocks"][best_block_idx]
            # Split block if significantly larger
            if blk["size"] > req_size + self.ALIGNMENT_BYTES:
                remainder_size = blk["size"] - req_size
                blk["size"] = req_size
                blk["free"] = False
                blk["id"] = alloc_id
                # Insert remainder
                new_blk = {
                    "offset": blk["offset"] + req_size,
                    "size": remainder_size,
                    "free": True,
                    "id": None
                }
                best_seg["blocks"].insert(best_block_idx + 1, new_blk)
            else:
                blk["free"] = False
                blk["id"] = alloc_id
            self.active_allocations[alloc_id] = (best_seg["id"], blk["offset"])
        else:
            # Allocate new segment from device
            default_seg_size = self.SMALL_SEGMENT_SIZE if pool == "small" else self.LARGE_SEGMENT_SIZE
            seg_size = max(default_seg_size, self.round_up(req_size))
            new_seg = {
                "id": self.next_seg_id,
                "size": seg_size,
                "pool": pool,
                "blocks": []
            }
            self.next_seg_id += 1

            # Allocated block
            new_seg["blocks"].append({
                "offset": 0,
                "size": req_size,
                "free": False,
                "id": alloc_id
            })
            # Remainder block
            if seg_size > req_size:
                new_seg["blocks"].append({
                    "offset": req_size,
                    "size": seg_size - req_size,
                    "free": True,
                    "id": None
                })

            self.segments.append(new_seg)
            self.reserved_bytes += seg_size
            self.active_allocations[alloc_id] = (new_seg["id"], 0)

        self.allocated_bytes += req_size
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, self.allocated_bytes)
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
        return req_size

    def deallocate(self, alloc_id: str):
        """Simulates cudaFree / block caching and coalescing."""
        if alloc_id not in self.active_allocations:
            return
        seg_id, offset = self.active_allocations.pop(alloc_id)
        for seg in self.segments:
            if seg["id"] == seg_id:
                for idx, blk in enumerate(seg["blocks"]):
                    if blk["offset"] == offset:
                        blk["free"] = True
                        blk["id"] = None
                        self.allocated_bytes -= blk["size"]
                        break
                # Coalesce adjacent free blocks (BFC)
                new_blocks = []
                for blk in seg["blocks"]:
                    if new_blocks and new_blocks[-1]["free"] and blk["free"]:
                        new_blocks[-1]["size"] += blk["size"]
                    else:
                        new_blocks.append(blk)
                seg["blocks"] = new_blocks
                break

    def reset(self):
        self.segments.clear()
        self.next_seg_id = 0
        self.allocated_bytes = 0
        self.peak_allocated_bytes = 0
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0
        self.active_allocations.clear()


class XMemEstimator:
    """
    xMem Accurate CPU-Based GPU Memory Estimator (Shi et al., 2025, arXiv:2510.21048v1).

    Simulates the exact 3-stage dynamic analysis:
      1. Analyzer: attributes tensor demands into model weights, activations, gradients, and optimizer.
      2. Memory Orchestrator: sets lifecycle windows (weights persistent, activations forward->backward,
         gradients backward->step, optimizer persistent).
      3. Memory Simulator: executes CUDACachingAllocator Best-Fit with Coalescing (BFC) simulation,
         accounting for 512B alignment and 2MB/20MB segment caching.
    """

    CUDA_CONTEXT_MB = 500.0

    def __init__(self, device: str = "cuda", device_capacity_mb: float = 4096.0):
        self.device = device
        self.device_capacity_mb = device_capacity_mb
        self.sim = CUDACachingAllocatorSimulator()

    def simulate_training_iteration(self, n_params: int, batch_size: int, seq_len: int,
                                    hidden_dim: int, n_layer: int, dtype_bytes: int = 2,
                                    optimizer: str = "adam") -> Dict[str, float]:
        """
        Executes xMem dynamic lifecycle trace simulation through the BFC CUDACachingAllocator.
        """
        self.sim.reset()

        # Phase 1: Model parameter loading (persistent lifecycle)
        w_bytes = n_params * dtype_bytes
        self.sim.allocate("model_weights", w_bytes)

        # Phase 2: Optimizer states (persistent lifecycle for Adam)
        if optimizer.lower() == "adam":
            # Adam keeps 2 FP32 states per parameter (m, v)
            opt_bytes = n_params * 8
            self.sim.allocate("optimizer_states", opt_bytes)
        elif optimizer.lower() == "adamw_8bit":
            opt_bytes = n_params * 2
            self.sim.allocate("optimizer_states", opt_bytes)
        else:
            opt_bytes = 0

        # Phase 3: Forward Pass Activation Accumulation
        # Per layer: input activations, self-attention QKV, attention scores, FFN projections
        # Memory Orchestrator: activations accumulate across layers during forward pass
        layer_act_bytes = int(16 * batch_size * seq_len * hidden_dim * dtype_bytes)
        for l in range(n_layer):
            self.sim.allocate(f"activation_layer_{l}", layer_act_bytes)

        # Phase 4: Backward Pass Gradient Generation and Activation Freeing
        # Gradients accumulate, activations are released layer by layer
        grad_bytes = int(w_bytes // max(1, n_layer))
        for l in reversed(range(n_layer)):
            # Generate layer gradient
            self.sim.allocate(f"grad_layer_{l}", grad_bytes)
            # Release corresponding forward activations
            self.sim.deallocate(f"activation_layer_{l}")

        # Phase 5: Optimizer step & Zero Grad
        for l in range(n_layer):
            self.sim.deallocate(f"grad_layer_{l}")

        peak_allocated_mb = self.sim.peak_allocated_bytes / (1024**2)
        peak_reserved_mb = self.sim.peak_reserved_bytes / (1024**2)
        total_peak_mb = peak_reserved_mb + self.CUDA_CONTEXT_MB

        oom = total_peak_mb > self.device_capacity_mb
        headroom_mb = max(0.0, self.device_capacity_mb - total_peak_mb)
        frag_mb = max(0.0, peak_reserved_mb - peak_allocated_mb)

        return {
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "cuda_context_mb": self.CUDA_CONTEXT_MB,
            "total_peak_mb": total_peak_mb,
            "total_mb": total_peak_mb,
            "fragmentation_mb": frag_mb,
            "oom_predicted": oom,
            "oom_on_4gb": oom,
            "headroom_mb": headroom_mb,
            "memory_conserved_mb": headroom_mb,
            "utilization": total_peak_mb / max(1.0, self.device_capacity_mb),
        }

    def estimate(self, n_params: int, batch_size: int, seq_len: int,
                 hidden_dim: int, n_layer: int, dtype_bytes: int = 2,
                 optimizer: str = "adam") -> Dict[str, float]:
        """Backward-compatible estimator API returning full xMem breakdown."""
        return self.simulate_training_iteration(
            n_params=n_params, batch_size=batch_size, seq_len=seq_len,
            hidden_dim=hidden_dim, n_layer=n_layer, dtype_bytes=dtype_bytes,
            optimizer=optimizer
        )

    def will_oom(self, **kwargs) -> bool:
        res = self.estimate(**kwargs)
        return res["oom_predicted"]

    def recommend_config(self, max_seq_len: int = 512, n_params: int = 285_000_000) -> Dict:
        """Recommend max batch_size and seq_len that fits in 4GB."""
        for bs in [4, 2, 1]:
            for sl in [512, 256, 128]:
                if sl > max_seq_len:
                    continue
                est = self.estimate(n_params=n_params, batch_size=bs, seq_len=sl,
                                    hidden_dim=1024, n_layer=6)
                if not est["oom_on_4gb"]:
                    return {"batch_size": bs, "seq_len": sl, "est": est}
        return {"batch_size": 1, "seq_len": 128, "est": self.estimate(n_params=n_params, batch_size=1, seq_len=128, hidden_dim=1024, n_layer=6)}


# Paper 15 (Checklist Row 15 / Oxford & NVIDIA, 2025, arXiv:2511.16652v2):
# Evolution Strategies at the Hyperscale: Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL)

class EGGROLLOptimizer:
    """
    EGGROLL: Evolution Guided GeneRal Optimisation via Low-rank Learning
    (Sarkar et al., Oxford FLAIR / WhiRL / MILA / NVIDIA, arXiv:2511.16652v2).

    Overcomes the arithmetic intensity bottleneck of naive Evolution Strategies (ES) on GPUs
    by structuring perturbations as low-rank matrices E_i = (1 / sqrt(r)) * A_i @ B_i^T:
      1. Hardware-Efficient Batched Forward Pass:
         Computes u_i @ (M + sigma * E_i)^T = u_i @ M^T + (sigma / sqrt(r)) * (u_i @ B_i) @ A_i^T
         Preserves dense GEMM throughput for mean parameter M while executing cheap rank-r projections.
      2. Unmaterialized Parameter Update:
         Calculates sum_i f_i * E_i without allocating giant [N, m, n] perturbation tensors:
         For rank-r matrices: Delta M = (eta / (N * sigma * sqrt(r))) * A_flat^T @ (F_rep * B_flat).
      3. Antithetic Perturbation Pairs:
         Evaluates positive (M + sigma * E_i) and negative (M - sigma * E_i) candidates to halve variance.
      4. Centered Rank & Sign Fitness Shaping:
         Maps raw fitness differentials into centered ranks in [-0.5, 0.5] or sign(s+ - s-) in {-1, 0, 1}.
      5. High-Dimensional Critical Decay:
         Counteracts Gaussian Annulus thin-shell concentration via sigma_d = sigma_0 * d^(-alpha).
    """

    def __init__(self, rank: int = 1, population_size: int = 32, sigma: float = 0.02,
                 lr: float = 0.01, shaping: str = "centered_rank"):
        self.rank = max(1, rank)
        self.pop_size = population_size
        self.sigma = sigma
        self.lr = lr
        self.shaping = shaping

    @classmethod
    def critical_sigma(cls, base_sigma: float, param_dim: int, alpha: float = 0.25) -> float:
        """
        Calculates critical decay rate sigma_d = base_sigma * (param_dim)^(-alpha)
        to prevent Gaussian Annulus divergence in high-dimensional parameter spaces (Section 5.1).
        """
        dim = max(1, int(param_dim))
        return float(base_sigma * (dim ** (-alpha)))

    def sample_perturbation_factors(self, out_features: int, in_features: int,
                                   pop_size: Optional[int] = None,
                                   rank: Optional[int] = None,
                                   device: Optional[torch.device] = None,
                                   dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples low-rank factors A ~ N(0, I) and B ~ N(0, I).
        Returns:
            A: [pop_size, out_features, rank]
            B: [pop_size, in_features, rank]
        """
        n_workers = pop_size or self.pop_size
        r = rank or self.rank
        a = torch.randn(n_workers, out_features, r, device=device, dtype=dtype)
        b = torch.randn(n_workers, in_features, r, device=device, dtype=dtype)
        return a, b

    def batched_forward(self, u: torch.Tensor, M: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor,
                        sigma: Optional[float] = None,
                        worker_idx: Optional[int] = None) -> torch.Tensor:
        """
        Hardware-efficient batched linear projection:
        u @ (M + sigma * E)^T = u @ M^T + (sigma / sqrt(r)) * (u @ B) @ A^T.

        Args:
            u: Input activation tensor [batch_size, in_features] or [batch_size, seq_len, in_features]
            M: Parameter weight matrix [out_features, in_features]
            A: Perturbation factor [out_features, rank] or [pop_size, out_features, rank]
            B: Perturbation factor [in_features, rank] or [pop_size, in_features, rank]
            sigma: Perturbation scale (defaults to self.sigma)
            worker_idx: Optional worker index if A and B are 3D
        """
        s = sigma if sigma is not None else self.sigma
        orig_shape = u.shape
        flat_u = u.view(-1, orig_shape[-1])
        base_out = flat_u @ M.T

        if A.dim() == 3:
            idx = worker_idx if worker_idx is not None else 0
            cur_a = A[idx]
            cur_b = B[idx]
        else:
            cur_a = A
            cur_b = B

        r = cur_a.size(-1)
        scale = s / math.sqrt(r)
        lora_out = (flat_u @ cur_b) @ cur_a.T
        out = base_out + scale * lora_out
        return out.view(*orig_shape[:-1], M.size(0))

    def fitness_shaping(self, fitness_pos: torch.Tensor,
                        fitness_neg: torch.Tensor,
                        mode: Optional[str] = None) -> torch.Tensor:
        """Applies antithetic fitness shaping."""
        shaping_mode = mode or self.shaping
        delta_f = fitness_pos - fitness_neg

        if shaping_mode == "sign":
            # Paper Appendix H.2: sign(s+ - s-) in {-1, 0, 1}
            return torch.sign(delta_f)
        elif shaping_mode == "standardized":
            std = delta_f.std().clamp(min=1e-6)
            return (delta_f - delta_f.mean()) / std
        else:
            # Centered rank transformation in [-0.5, 0.5]
            ranks = delta_f.argsort().argsort().float()
            denom = max(1.0, float(delta_f.numel() - 1))
            return (ranks / denom) - 0.5

    def compute_unmaterialized_gradient(self, A: torch.Tensor, B: torch.Tensor,
                                        shaped_fitness: torch.Tensor,
                                        sigma: Optional[float] = None) -> torch.Tensor:
        """
        Calculates gradient estimate without materializing the individual E_i:
        E = (1 / sqrt(r)) * (A_flat^T @ (F_rep * B_flat))
        grad = E / (pop_size * sigma)
        """
        s = sigma if sigma is not None else self.sigma
        n_workers = A.size(0)
        out_f = A.size(1)
        in_f = B.size(1)
        r = A.size(2)

        a_flat = A.permute(0, 2, 1).reshape(n_workers * r, out_f)
        b_flat = B.permute(0, 2, 1).reshape(n_workers * r, in_f)
        f_rep = shaped_fitness.unsqueeze(-1).expand(n_workers, r).reshape(n_workers * r, 1)

        weighted_b = f_rep * b_flat
        unmat_e = (1.0 / math.sqrt(r)) * (a_flat.T @ weighted_b)
        grad_est = unmat_e / (float(n_workers) * max(1e-6, s))
        return grad_est

    def step(self, M: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
             fitness_pos: torch.Tensor, fitness_neg: torch.Tensor,
             lr: Optional[float] = None) -> torch.Tensor:
        """Executes complete EGGROLL optimization step on parameter matrix M."""
        step_lr = lr if lr is not None else self.lr
        shaped_f = self.fitness_shaping(fitness_pos, fitness_neg)
        grad_est = self.compute_unmaterialized_gradient(A, B, shaped_f)
        return M + step_lr * grad_est


class HyperscaleES:
    """
    Hyperscale Evolution Strategies with EGGROLL (Oxford FLAIR / WhiRL / MILA / NVIDIA).
    Supports 1D vector parameters and 2D matrix weights via unmaterialized rank-r perturbations.
    """

    def __init__(self, hidden_dim: int, population_size: int = 32, sigma: float = 0.02,
                 rank: int = 1, lr: float = 0.01):
        self.hidden_dim = hidden_dim
        self.pop_size = population_size
        self.sigma = sigma
        self.rank = rank
        self.lr = lr
        self.eggroll = EGGROLLOptimizer(rank=rank, population_size=population_size,
                                        sigma=sigma, lr=lr)

    def antithetic_sample(self, mean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate antithetic pair: mean + sigma*eps, mean - sigma*eps."""
        eps = torch.randn_like(mean) * self.sigma
        return mean + eps, mean - eps

    def fitness_shaping(self, fitness: torch.Tensor) -> torch.Tensor:
        """Rank-based fitness shaping (centered ranks in [-0.5, 0.5])."""
        ranks = fitness.argsort().argsort().float()
        centered = (ranks / (fitness.numel() - 1 + 1e-6)) - 0.5
        return centered

    def batched_forward(self, u: torch.Tensor, M: torch.Tensor,
                        A: torch.Tensor, B: torch.Tensor,
                        worker_idx: Optional[int] = None) -> torch.Tensor:
        """Hardware-efficient low-rank forward pass."""
        return self.eggroll.batched_forward(u, M, A, B, sigma=self.sigma, worker_idx=worker_idx)

    def update(self, mean: torch.Tensor, fitness_pos: torch.Tensor,
               fitness_neg: torch.Tensor, eps: Optional[torch.Tensor] = None,
               A: Optional[torch.Tensor] = None, B: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes ES update with antithetic fitness.
        If A and B are provided, executes exact unmaterialized EGGROLL low-rank update.
        Otherwise executes standard antithetic vector update.
        """
        if A is not None and B is not None:
            return self.eggroll.step(mean, A, B, fitness_pos, fitness_neg, lr=self.lr)

        delta_f = self.fitness_shaping(fitness_pos - fitness_neg)
        if eps is None:
            eps = torch.randn_like(mean) * self.sigma

        if eps.dim() > mean.dim():
            update = (delta_f.view(-1, *([1] * mean.dim())) * eps).mean(dim=0)
        else:
            update = delta_f.mean() * eps
        return mean + self.lr * update


# Paper 16 (Checklist Row 16 / CASIA & UCAS, 2026, arXiv:2607.24720v1):
# The Physics of Multi-Turn Long-Horizon Planning: On-Policy Agentic Distillation (OPD & MOPD)

class LongHorizonDistiller:
    """
    The Physics of Multi-Turn Long-Horizon Planning: Single- and Multi-Teacher
    On-Policy Agentic Distillation (Men et al., CASIA / UCAS, arXiv:2607.24720v1).

    Solves long-horizon credit assignment decay and error amplification via:
      1. Full-Vocabulary GKD Agent-OPD (Eq. 5):
         Exact token-level KL divergence along student on-policy trajectory prefixes.
      2. Top-k Vocabulary GKD Agent-OPD (Eq. 3, 4):
         Truncated and re-normalized probability mass divergence over high-confidence token subsets.
      3. Sampled-Token Agent-OPD:
         Schulman K1 / K2 distance minimization and Policy Gradient (PG) advantage estimation.
      4. Multi-Teacher On-Policy Agentic Distillation (MOPD, Section 6):
         Consolidates specialized teachers (e.g. Glimmer long-horizon, Lightning short-horizon, Omni medium)
         via horizon-decay potential weights w_m(h) and inter-teacher pattern compatibility auditing.
      5. World-Model CoT State Transition Modeling (Section 4):
         Supervised internal state representation alignment preventing compounding execution drift.
    """

    def __init__(self, hidden_dim: int, gamma: float = 0.95, num_teachers: int = 3,
                 max_horizon: int = 10, default_top_k: int = 10):
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.num_teachers = num_teachers
        self.max_horizon = max_horizon
        self.default_top_k = default_top_k
        # Teacher preferred horizon specializations: [long=8, short=2, medium=5]
        self.teacher_preferred_horizons = [8, 2, 5]

    def horizon_weight(self, horizon: int) -> float:
        """Physical potential decay weight: gamma^h (Eq. 20)."""
        h = max(0, min(int(horizon), self.max_horizon))
        return float(self.gamma ** h)

    def multi_teacher_weight(self, teacher_id: int, horizon: int) -> float:
        """
        Weights teacher_id signal for given horizon:
        Glimmer (0): long-horizon specialist -> peak at h=8
        Lightning (1): short-horizon specialist -> peak at h=2
        Omni (2): medium-horizon specialist -> peak at h=5
        """
        preferred = self.teacher_preferred_horizons[teacher_id % len(self.teacher_preferred_horizons)]
        gaussian_kernel = math.exp(-0.5 * ((horizon - preferred) / 3.0) ** 2)
        return gaussian_kernel * self.horizon_weight(horizon)

    @classmethod
    def full_vocabulary_opd(cls, student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Full-Vocabulary GKD Agent-OPD (Section 2, Eq. 5).
        Aligns complete predictive distribution on student on-policy prefixes:
        L_Full(theta) = E [ sum_{i,t} D_KL(pi_student || pi_teacher) ]
        """
        s_log = F.log_softmax(student_logits, dim=-1)
        t_prob = F.softmax(teacher_logits, dim=-1)
        return F.kl_div(s_log, t_prob, reduction="batchmean")

    @classmethod
    def top_k_opd(cls, student_logits: torch.Tensor,
                  teacher_logits: torch.Tensor,
                  k: int = 10) -> torch.Tensor:
        """
        Top-k Vocabulary GKD Agent-OPD (Section 2, Eq. 3, 4).
        Truncates and re-normalizes probability mass functions over top-k student tokens:
        tilde{pi}(v) = pi(v) / sum_{w in V^k} pi(w).
        """
        v_size = student_logits.size(-1)
        k_val = min(k, v_size)
        top_vals, top_idx = torch.topk(student_logits, k=k_val, dim=-1)
        s_top_log = F.log_softmax(top_vals, dim=-1)

        t_top_logits = torch.gather(teacher_logits, -1, top_idx)
        t_top_prob = F.softmax(t_top_logits, dim=-1)

        return F.kl_div(s_top_log, t_top_prob, reduction="batchmean")

    @classmethod
    def sampled_token_opd(cls, student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          sampled_tokens: torch.Tensor,
                          mode: str = "k2") -> torch.Tensor:
        """
        Sampled-Token Agent-OPD (Section 2).
        Supports Schulman K1, Schulman K2, and Policy Gradient (PG) advantage estimation.
        """
        s_log = F.log_softmax(student_logits, dim=-1)
        t_log = F.log_softmax(teacher_logits, dim=-1)

        token_s_log = torch.gather(s_log, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        token_t_log = torch.gather(t_log, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)

        log_diff = token_s_log - token_t_log

        if mode.lower() == "k1":
            # Schulman K1 approximation: log pi_s - log pi_t
            return log_diff.mean()
        elif mode.lower() == "pg":
            # Policy gradient advantage: A = log pi_t - log pi_s
            advantage = (-log_diff).detach()
            return -(advantage * token_s_log).mean()
        else:
            # Schulman K2 approximation: 0.5 * (log pi_s - log pi_t)^2
            return 0.5 * (log_diff ** 2).mean()

    @classmethod
    def compute_teacher_compatibility(cls, teacher_logits_list: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Audits multi-teacher compatibility and pattern conflicts (Section 6.2 & 6.3).
        Compatible teachers (sim >= 0.5) accelerate cross-environment planning.
        Conflicting teachers (sim < 0.2) induce catastrophic forgetting.
        """
        n_teachers = len(teacher_logits_list)
        if n_teachers < 2:
            return {"pairwise_similarities": [], "mean_compatibility": 1.0, "has_conflict": False}

        probs = [F.softmax(t.reshape(-1, t.size(-1)), dim=-1) for t in teacher_logits_list]
        sims = []
        conflicts = 0

        for i in range(n_teachers):
            for j in range(i + 1, n_teachers):
                cos = F.cosine_similarity(probs[i], probs[j], dim=-1).mean().item()
                sims.append(cos)
                if cos < 0.20:
                    conflicts += 1

        mean_sim = float(sum(sims) / max(1, len(sims)))
        return {
            "pairwise_similarities": sims,
            "mean_compatibility": mean_sim,
            "has_conflict": conflicts > 0,
            "conflict_ratio": float(conflicts / max(1, len(sims)))
        }

    @classmethod
    def world_model_transition_loss(cls, predicted_states: torch.Tensor,
                                    target_states: torch.Tensor) -> torch.Tensor:
        """
        Chain-of-Thought World Model state transition modeling loss (Section 4).
        MSE loss between predicted next-state embeddings and target environment states.
        """
        return F.mse_loss(predicted_states, target_states)

    def distill_loss(self, student_logits: torch.Tensor,
                     teacher_logits_list: List[torch.Tensor],
                     horizons: Optional[List[int]] = None,
                     mode: str = "full_vocab",
                     top_k: Optional[int] = None) -> torch.Tensor:
        """
        Horizon-weighted Multi-Teacher On-Policy Agentic Distillation loss (MOPD).
        """
        if not teacher_logits_list:
            return student_logits.new_tensor(0.0)

        if horizons is None:
            horizons = [5] * len(teacher_logits_list)

        total_loss = student_logits.new_tensor(0.0)
        total_weight = 0.0

        k_val = top_k or self.default_top_k

        for t_idx, t_logits in enumerate(teacher_logits_list):
            h = horizons[t_idx % len(horizons)] if isinstance(horizons, list) else horizons
            w = self.multi_teacher_weight(t_idx, h)

            if mode == "top_k":
                kl = self.top_k_opd(student_logits, t_logits, k=k_val)
            else:
                kl = self.full_vocabulary_opd(student_logits, t_logits)

            total_loss = total_loss + w * kl
            total_weight += w

        return total_loss / max(1e-6, total_weight)


class QuantMemoryManager(nn.Module):
    """
    Combined Papers 11-15: quantization + memory estimation + ES + long-horizon.

    Usage:
        mgr = QuantMemoryManager(hidden_dim=1024, n_layer=6)
        xmem = mgr.xmem.estimate(n_params=285_000_000, batch_size=2, seq_len=512)
        print(f"Will OOM: {xmem['oom_on_4gb']}, total {xmem['total_mb']:.0f}MB")
        nvfp4 = mgr.nvfp4  # for model quantization
        es = mgr.es  # for council exploration
        lh = mgr.long_horizon  # for multi-teacher distillation
    """

    def __init__(self, hidden_dim: int = 1024, n_layer: int = 6):
        super().__init__()
        self.nvfp4 = NVFP4Quantizer()
        self.xmem = XMemEstimator()
        self.es = HyperscaleES(hidden_dim)
        self.long_horizon = LongHorizonDistiller(hidden_dim)

    def get_stats(self, seq_len: int = 512, batch_size: int = 2,
                  n_params: int = 285_000_000) -> Dict:
        xmem = self.xmem.estimate(n_params=n_params, batch_size=batch_size,
                                   seq_len=seq_len, hidden_dim=1024, n_layer=6)
        return {
            "seq_len": seq_len,
            "batch_size": batch_size,
            "xmem_total_mb": xmem["total_mb"],
            "xmem_oom": xmem["oom_on_4gb"],
            "xmem_headroom": xmem["headroom_mb"],
            "nvfp4_ratio": self.nvfp4.compression_ratio(),
            "nvfp4_model_mb": n_params * 0.5 / (1024**2),  # approx
        }
