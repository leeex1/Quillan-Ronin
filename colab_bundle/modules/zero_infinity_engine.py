#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeRO-Infinity Engine: Breaking the GPU Memory Wall for Extreme Scale Deep Learning
================================================================================
Formal implementation of ZeRO-Infinity (Rajbhandari, Ruwase, Rasley, Smith, He; Microsoft,
SC '21 / arXiv:2104.07857).

Mathematical & Architectural Formulations:
------------------------------------------
1. Heterogeneous Memory Hierarchy & Model Scale (Sections 4 & 5.1):
   - Static Model State Breakdown (Mixed-Precision Adam):
       * FP16 Parameters: 2 bytes / param
       * FP16 Gradients: 2 bytes / param
       * FP32 Master Parameters: 4 bytes / param
       * FP32 First Momentum: 4 bytes / param
       * FP32 Second Momentum (Variance): 4 bytes / param
       * Total Static States: 16 bytes / param
   - Activation Checkpointing Memory:
       A_ckpt = c_i * bsz * seq * hd * 2 bytes (per Transformer block)
   - Heterogeneous Capacity across GPU, CPU, and NVMe:
       M_total = N_DP * (M_GPU + M_CPU + M_NVMe)
       Max Trainable Parameters: P_max = M_total / 16

2. Arithmetic Intensity (AIT) & Computational Efficiency (Section 4, Eqs. 7-8):
   - Computation per Iteration:
       FLOPs = 2 * 4 * bsz * seq * P = 8 * bsz * seq * P = 96 * bsz * seq * n_l * hd^2
   - Parameter & Gradient Data Movement:
       Movement = 2P (forward + backward) + 1P (activation recompute) + 1P (grad store) = 4P elements = 8P bytes
   - Arithmetic Intensity:
       AIT = FLOPs / Data_Bytes = (8 * bsz * seq * P) / (8P) = bsz * seq (FLOPs/byte)
   - Efficiency w.r.t. Achievable Bandwidth B_eff:
       Efficiency = T_comp / (T_comp + T_comm)
                  = (FLOPs / Peak_FLOPS) / ((FLOPs / Peak_FLOPS) + (Data_Bytes / B_eff))

3. Bandwidth-Centric Partitioning (Section 6.1):
   - Broadcast-based approach (ZeRO-Offload): Single PCIe bottleneck B_eff = B_PCIe (~12 GB/s).
   - ZeRO-Infinity All-Gather approach: Each GPU loads 1/N_DP from local NVMe/CPU concurrently,
     then participates in high-speed All-Gather over NVLink (>= 70 GB/s).
   - Aggregate effective offload bandwidth scales linearly with DP degree:
       B_eff = min(N_DP * B_local, B_allgather)

4. Overlap-Centric Pipeline Engine (Section 6.2):
   - Overlaps 3-stage parameter reconstruction:
       * nc-transfer: NVMe -> CPU for operator i+3
       * cg-transfer: CPU -> GPU for operator i+2
       * gg-transfer: GPU All-Gather for operator i+1
       * Concurrently with GPU computation of operator i
       T_step = max(T_comp(i), T_gg(i+1), T_cg(i+2), T_nc(i+3))

5. Memory-Centric Tiling (Section 5.1.3):
   - Decomposes large linear operator Y = X W (W in R^{d_in x d_out}) into K tiles W = [W_1, ..., W_K].
   - Model State Working Memory reduced by factor K: MSWM_tiled = (d_in * d_out * 2) / K.
   - Forward: Y_k = X W_k executed sequentially with tile prefetch/release.
   - Backward: dX = sum_k dY_k W_k^T, dW_k = X^T dY_k.
   - Enables operators of arbitrary dimension (e.g. hd > 100,000) on a single GPU without tensor parallelism.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Heterogeneous Memory Hierarchy & Model Scale
# ==============================================================================

@dataclass
class HeterogeneousMemoryBreakdown:
    """Represents the memory distribution across GPU, CPU, and NVMe (Section 4 & 5.1)."""
    num_params: int
    hidden_dim: int
    num_layers: int
    batch_size: int
    seq_len: int
    num_data_parallel_ranks: int

    # Static memory per parameter for mixed-precision Adam = 16 bytes
    fp16_params_bytes: int = 0
    fp16_grads_bytes: int = 0
    fp32_optimizer_bytes: int = 0
    total_static_bytes: int = 0
    activation_checkpoint_bytes: int = 0

    def __post_init__(self):
        # 2 bytes fp16 params + 2 bytes fp16 grads
        self.fp16_params_bytes = self.num_params * 2
        self.fp16_grads_bytes = self.num_params * 2
        # 12 bytes fp32 Adam: 4B master weights, 4B momentum, 4B variance
        self.fp32_optimizer_bytes = self.num_params * 12
        self.total_static_bytes = self.num_params * 16

        # Activation checkpointing: c_i * bsz * seq * hd * 2 bytes per layer (1 checkpoint per block)
        self.activation_checkpoint_bytes = (
            self.num_layers * self.batch_size * self.seq_len * self.hidden_dim * 2
        )

    def memory_per_rank(self) -> Dict[str, float]:
        """Returns static model states and activations per DP rank in Gigabytes (GB)."""
        dp = self.num_data_parallel_ranks
        return {
            "static_states_gb": (self.total_static_bytes / dp) / (1024 ** 3),
            "activation_ckpt_gb": (self.activation_checkpoint_bytes / dp) / (1024 ** 3),
            "total_per_rank_gb": ((self.total_static_bytes + self.activation_checkpoint_bytes) / dp) / (1024 ** 3)
        }

    @staticmethod
    def max_trainable_parameters(
        gpu_mem_gb: float,
        cpu_mem_gb: float,
        nvme_mem_gb: float,
        num_ranks: int
    ) -> int:
        """
        Calculates maximum trainable parameters fitting in aggregate heterogeneous memory:
        P_max = (N_ranks * (M_GPU + M_CPU + M_NVMe)) / 16 bytes.
        """
        total_bytes = num_ranks * (gpu_mem_gb + cpu_mem_gb + nvme_mem_gb) * (1024 ** 3)
        return int(total_bytes // 16)


# ==============================================================================
# 2. Arithmetic Intensity & Computational Efficiency
# ==============================================================================

class ArithmeticIntensityModel:
    """
    Arithmetic Intensity (AIT) and Efficiency formulation (Section 4, Eqs. 7-8).
    """

    @staticmethod
    def compute_flops(num_params: int, batch_size: int, seq_len: int) -> float:
        """
        computation_per_iter = 2 * 4 * bsz * seq * parameters (Eq. 7).
        Forward = 2 * bsz * seq * params
        Backward = 2 * Forward
        Activation Recompute = 1 * Forward
        Total = 4 * Forward = 8 * bsz * seq * params.
        """
        return float(8 * batch_size * seq_len * num_params)

    @staticmethod
    def compute_data_movement_bytes(num_params: int) -> float:
        """
        Data movement: 2P (fwd + bwd) + 1P (recompute) + 1P (grad store) = 4P elements.
        In FP16 (2 bytes/elem): Total = 8 * P bytes.
        """
        return float(8 * num_params)

    @staticmethod
    def calculate_ait(batch_size: int, seq_len: int) -> float:
        """
        Arithmetic Intensity (AIT) = FLOPs / Data_Bytes = (8 * bsz * seq * P) / (8P) = bsz * seq (FLOPs/byte).
        """
        return float(batch_size * seq_len)

    @staticmethod
    def calculate_efficiency(
        num_params: int,
        batch_size: int,
        seq_len: int,
        effective_bandwidth_gb_s: float,
        peak_device_tflops: float
    ) -> float:
        """
        Efficiency = T_comp / (T_comp + T_comm).
        T_comp = FLOPs / (peak_device_tflops * 10^12)
        T_comm = Data_Bytes / (effective_bandwidth_gb_s * 10^9)
        """
        flops = ArithmeticIntensityModel.compute_flops(num_params, batch_size, seq_len)
        data_bytes = ArithmeticIntensityModel.compute_data_movement_bytes(num_params)

        t_comp = flops / (peak_device_tflops * 1e12)
        t_comm = data_bytes / (effective_bandwidth_gb_s * 1e9)

        if t_comp + t_comm == 0:
            return 0.0
        return float(t_comp / (t_comp + t_comm))


# ==============================================================================
# 3. Bandwidth-Centric Partitioning
# ==============================================================================

class BandwidthCentricPartitioning:
    """
    Bandwidth-Centric Partitioning vs Broadcast-based Transfer (Section 6.1).
    Proves effective transfer bandwidth scales linearly with DP degree.
    """

    @staticmethod
    def broadcast_effective_bandwidth(pcie_bandwidth_gb_s: float = 12.0) -> float:
        """In broadcast-based offload, single owner GPU reads over PCIe: B_eff = B_PCIe."""
        return float(pcie_bandwidth_gb_s)

    @staticmethod
    def zero_infinity_effective_bandwidth(
        num_ranks: int,
        local_bandwidth_gb_s: float = 12.0,
        allgather_network_bandwidth_gb_s: float = 70.0
    ) -> float:
        """
        ZeRO-Infinity partitions individual parameters across all DP ranks.
        All ranks read their 1/N_DP shard concurrently over local PCIe/NVMe, then All-Gather:
        B_eff = min(N_DP * B_local, B_allgather).
        """
        aggregate_local = num_ranks * local_bandwidth_gb_s
        return float(min(aggregate_local, allgather_network_bandwidth_gb_s))


# ==============================================================================
# 4. Overlap-Centric Pipeline Engine
# ==============================================================================

class ThreeStageOverlapEngine:
    """
    Overlap-Centric Pipeline Engine (Section 6.2).
    Simulates three-stage pipelined parameter reconstruction:
      - nc-transfer: NVMe -> CPU for operator i+3
      - cg-transfer: CPU -> GPU for operator i+2
      - gg-transfer: GPU All-Gather for operator i+1
      - Executed concurrently with GPU computation of operator i.
    """

    @staticmethod
    def simulate_pipeline(
        layer_compute_times_ms: List[float],
        nvme_to_cpu_times_ms: List[float],
        cpu_to_gpu_times_ms: List[float],
        allgather_times_ms: List[float]
    ) -> Dict[str, float]:
        n_layers = len(layer_compute_times_ms)
        if not (n_layers == len(nvme_to_cpu_times_ms) == len(cpu_to_gpu_times_ms) == len(allgather_times_ms)):
            raise ValueError("All transfer time arrays must have the same length as compute times.")

        # Sequential baseline (no overlap):
        total_sequential_ms = (
            sum(layer_compute_times_ms) +
            sum(nvme_to_cpu_times_ms) +
            sum(cpu_to_gpu_times_ms) +
            sum(allgather_times_ms)
        )

        # Overlapped pipeline execution:
        # In steady state for layer i:
        # T_step(i) = max(T_comp(i), T_gg(i+1), T_cg(i+2), T_nc(i+3))
        total_overlapped_ms = 0.0
        # Warmup: fill 3-stage pipeline
        total_overlapped_ms += nvme_to_cpu_times_ms[0]
        total_overlapped_ms += max(nvme_to_cpu_times_ms[min(1, n_layers - 1)], cpu_to_gpu_times_ms[0])
        total_overlapped_ms += max(
            nvme_to_cpu_times_ms[min(2, n_layers - 1)],
            cpu_to_gpu_times_ms[min(1, n_layers - 1)],
            allgather_times_ms[0]
        )

        # Steady-state overlapped loop
        for i in range(n_layers):
            t_comp = layer_compute_times_ms[i]
            t_gg = allgather_times_ms[i + 1] if i + 1 < n_layers else 0.0
            t_cg = cpu_to_gpu_times_ms[i + 2] if i + 2 < n_layers else 0.0
            t_nc = nvme_to_cpu_times_ms[i + 3] if i + 3 < n_layers else 0.0
            step_ms = max(t_comp, t_gg, t_cg, t_nc)
            total_overlapped_ms += step_ms

        speedup = total_sequential_ms / total_overlapped_ms if total_overlapped_ms > 0 else 1.0
        hidden_comm_ms = total_sequential_ms - total_overlapped_ms

        return {
            "sequential_ms": total_sequential_ms,
            "overlapped_ms": total_overlapped_ms,
            "speedup": speedup,
            "hidden_communication_pct": (hidden_comm_ms / total_sequential_ms) * 100.0
        }


# ==============================================================================
# 5. Memory-Centric Tiling (Layer Slicing)
# ==============================================================================

class MemoryCentricTiledLinear(nn.Module):
    """
    Memory-Centric Tiling for Large Linear Operators (Section 5.1.3).
    Partitions an arbitrarily large linear layer W in R^{d_in x d_out} into K column tiles:
    W = [W_1, W_2, ..., W_K].
    Executes sequentially: Y_k = X W_k, fetching and releasing each tile one at a time.
    Reduces peak Model State Working Memory (MSWM) by a factor of K.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tiles: int = 4,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tiles = num_tiles
        self.tile_out_features = out_features // num_tiles

        if out_features % num_tiles != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by num_tiles ({num_tiles})")

        # Represented as a list of sequential parameter tiles
        self.weight_tiles = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, self.tile_out_features) * 0.02)
            for _ in range(num_tiles)
        ])

        if bias:
            self.bias_tiles = nn.ParameterList([
                nn.Parameter(torch.zeros(self.tile_out_features))
                for _ in range(num_tiles)
            ])
        else:
            self.bias_tiles = None

    def get_full_weight(self) -> torch.Tensor:
        """Reconstructs the full weight matrix for validation against standard linear layers."""
        return torch.cat(list(self.weight_tiles), dim=1)

    def working_memory_bytes_per_tile(self) -> int:
        """Returns the memory footprint of a single tile: (d_in * d_out / K) * 2 bytes (FP16)."""
        return (self.in_features * self.tile_out_features) * 2

    def original_working_memory_bytes(self) -> int:
        """Returns the memory footprint of the un-tiled weight matrix: (d_in * d_out) * 2 bytes (FP16)."""
        return (self.in_features * self.out_features) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes tiled forward pass:
        Fetches each tile W_k, computes partial projection Y_k = x @ W_k, and concatenates.
        """
        outputs = []
        for k in range(self.num_tiles):
            w_k = self.weight_tiles[k]
            y_k = torch.matmul(x, w_k)
            if self.bias_tiles is not None:
                y_k = y_k + self.bias_tiles[k]
            outputs.append(y_k)

        return torch.cat(outputs, dim=-1)
