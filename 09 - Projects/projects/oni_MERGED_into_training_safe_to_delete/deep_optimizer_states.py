r"""
Deep Optimizer States: Towards Scalable Training of Transformer Models Using Interleaved Offloading
(Avinash Maurya, Jie Ye, M. Mustafa Rafique, Franck Cappello, Bogdan Nicolae - RIT, IIT, Argonne National Lab, Middleware '24, arXiv:2410.21316).

Mathematical Foundations & Core Algorithmic Invariants:
--------------------------------------------------------
1. Interleaved Analytical Performance Model (Eq. 1):
   In hybrid CPU-GPU optimizer offloading, static partitioning (e.g. TwinFlow) suffers from:
     (a) idle CPU while GPU updates GPU-resident subgroups;
     (b) blocking H2D transfers of updated parameters;
     (c) sequential FP32 -> FP16 downscaling stalls.
   DOS computes the optimal CPU-to-GPU update ratio k:1 (stride k) by equating the CPU
   compute + downscale pipeline with GPU compute + bidirectional PCIe transfer time:
       k * (S / U_c + S / D_c) = max(3S / B, 3S / B + k * S / (2 * B)) + S / U_g
   Solving for k (invariant to subgroup size S):
       k = (3 / B + 1 / U_g) / (1 / U_c + 1 / D_c - 1 / (2 * B))
   where:
     - S: number of parameters per subgroup
     - U_c: CPU optimizer update throughput (parameters/sec)
     - U_g: GPU optimizer update throughput (parameters/sec)
     - D_c: CPU FP32 -> FP16 downscaling throughput (parameters/sec)
     - B: PCIe bidirectional transfer throughput (parameters/sec)

2. Optimizer Update Scheduling Algorithm (Algorithm 1):
   Input: Partitioned subgroups <S>, update stride k, static GPU residents <R>.
   For each subgroup i in S:
     - If i in R or (i + 1) % k == 0:
         target = GPU update;
         asynchronously downscale accumulated CPU-updated subgroups on CPU.
     - Else if i not in R and i % k == 0:
         (previous subgroup was updated on GPU)
         asynchronously flush out previous GPU subgroup (D2H m, v, p and D2D FP16 downscale);
         asynchronously prefetch in next GPU subgroup (H2D m, v, p);
         target = CPU update;
     - Else:
         target = CPU update;
         enqueue i for CPU downscale.

3. On-the-Fly GPU Gradient Conversion & Pinned Flushing (Section 4.1):
   ZeRO-3 flushes FP16 gradients to host at ~2.5 GB/s due to unpinned temporary host buffers
   and host FP16 -> FP32 conversion stalls (22x slower than peak PCIe).
   DOS executes in-place chunk-wise FP16 -> FP32 conversion directly on the GPU (at 1.2 TB/s VRAM),
   then flushes FP32 gradients straight to pinned host memory at peak DMA speeds (>50 GB/s),
   eliminating host allocation bottlenecks.

4. Numerical Equivalence Invariant:
   Interleaved execution produces identical mathematical states to full-GPU AdamW:
     m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
     v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
     theta_t = theta_{t-1} - lr * ( (m_t / (1 - beta1^t)) / (sqrt(v_t / (1 - beta2^t)) + eps) + wd * theta_{t-1} )
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class DOSHardwareProfile:
    """Hardware throughput specifications for performance model (parameters/sec)."""
    u_c: float = 2.5e7   # CPU Adam update throughput (~25M param/s)
    u_g: float = 5.0e8   # GPU Adam update throughput (~500M param/s)
    d_c: float = 1.5e8   # CPU FP32 -> FP16 downscale throughput (~150M param/s)
    b_pcie: float = 6.25e9  # PCIe Gen4/Gen5 transfer throughput (params/s, e.g. 25 GB/s / 4B ~ 6.25G)

    def compute_optimal_stride(self) -> float:
        """
        Computes optimal CPU-to-GPU update ratio k using Equation 1:
            k = (3 / B + 1 / U_g) / (1 / U_c + 1 / D_c - 1 / (2 * B))
        """
        numerator = (3.0 / self.b_pcie) + (1.0 / self.u_g)
        denominator = (1.0 / self.u_c) + (1.0 / self.d_c) - (1.0 / (2.0 * self.b_pcie))
        if denominator <= 0:
            return 1.0  # Fast PCIe / compute fallback
        return max(1.0, numerator / denominator)

    def compute_optimal_stride_int(self) -> int:
        """Returns integer update stride k >= 1."""
        return max(1, int(round(self.compute_optimal_stride())))


class DOSSubgroup:
    """
    Represents a single partitioned optimizer subgroup (shard of model parameters).
    Stores FP32 master parameters, FP16 model weights, momentum, variance, and gradients.
    """
    def __init__(
        self,
        subgroup_id: int,
        param_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        self.subgroup_id = subgroup_id
        self.numel = param_tensor.numel()
        self.shape = param_tensor.shape
        self.device = device

        # Master weights in FP32
        self.p_fp32 = param_tensor.detach().clone().float().to(device)
        # Model weights in FP16 (used in forward/backward)
        self.p_fp16 = self.p_fp32.half().to(device)
        # Optimizer states in FP32
        self.m_fp32 = torch.zeros_like(self.p_fp32)
        self.v_fp32 = torch.zeros_like(self.p_fp32)
        # Gradient buffer
        self.grad_fp32 = torch.zeros_like(self.p_fp32)

    def set_grad(self, grad_tensor: torch.Tensor):
        self.grad_fp32.copy_(grad_tensor.float().to(self.device))


class DOSInterleavedScheduler:
    """
    Optimizer update scheduling algorithm (Algorithm 1).
    Plans and tracks interleaved CPU vs GPU updates, static GPU residents,
    and asynchronous prefetching / flushing pipelines.
    """
    def __init__(
        self,
        num_subgroups: int,
        stride_k: int,
        static_gpu_residents: Optional[Set[int]] = None,
    ):
        self.num_subgroups = num_subgroups
        self.stride_k = max(1, stride_k)
        self.static_gpu_residents = static_gpu_residents or set()

    def plan_schedule(self) -> List[Dict[str, Union[int, str, List[int]]]]:
        """
        Executes Algorithm 1 to derive execution targets and pipeline hooks.

        Returns list of step descriptors:
          - subgroup_id: int
          - target: 'GPU' or 'CPU'
          - is_static_gpu: bool
          - async_downscale_ids: List[int]
          - flush_out_id: Optional[int]
          - prefetch_in_id: Optional[int]
        """
        schedule = []
        fp32_fp16_conv_queue: List[int] = []
        gpu_scheduled_order: List[int] = []

        # Identify all GPU targets ahead of time to support prefetch/flush lookahead
        for i in range(self.num_subgroups):
            if i in self.static_gpu_residents or (i + 1) % self.stride_k == 0:
                gpu_scheduled_order.append(i)

        for i in range(self.num_subgroups):
            is_static = i in self.static_gpu_residents
            is_stride_gpu = (i + 1) % self.stride_k == 0

            if is_static or is_stride_gpu:
                # GPU update
                downscale_tasks = list(fp32_fp16_conv_queue)
                fp32_fp16_conv_queue.clear()
                step = {
                    "subgroup_id": i,
                    "target": "GPU",
                    "is_static_gpu": is_static,
                    "async_downscale_ids": downscale_tasks,
                    "flush_out_id": None,
                    "prefetch_in_id": None,
                }
                schedule.append(step)
            else:
                # CPU candidate
                flush_id = None
                prefetch_id = None
                if i not in self.static_gpu_residents and i % self.stride_k == 0 and i > 0:
                    # Previous subgroup was updated on GPU
                    # Flushes out previous GPU update and prefetches next GPU update
                    curr_gpu_idx = -1
                    for g_idx, g_id in enumerate(gpu_scheduled_order):
                        if g_id < i:
                            curr_gpu_idx = g_idx
                    if curr_gpu_idx >= 0:
                        flush_id = gpu_scheduled_order[curr_gpu_idx]
                    if curr_gpu_idx + 1 < len(gpu_scheduled_order):
                        prefetch_id = gpu_scheduled_order[curr_gpu_idx + 1]

                fp32_fp16_conv_queue.append(i)
                step = {
                    "subgroup_id": i,
                    "target": "CPU",
                    "is_static_gpu": False,
                    "async_downscale_ids": [],
                    "flush_out_id": flush_id,
                    "prefetch_in_id": prefetch_id,
                }
                schedule.append(step)

        return schedule


class DeepOptimizerStates(nn.Module):
    """
    Deep Optimizer States Middleware Engine.
    Executes interleaved CPU-GPU AdamW updates with on-the-fly gradient conversion
    and full-duplex asynchronous offloading.
    """
    def __init__(
        self,
        subgroups: List[DOSSubgroup],
        hw_profile: Optional[DOSHardwareProfile] = None,
        stride_k: Optional[int] = None,
        static_gpu_residents: Optional[Set[int]] = None,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        gpu_device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.subgroups = {sg.subgroup_id: sg for sg in subgroups}
        self.hw_profile = hw_profile or DOSHardwareProfile()
        self.stride_k = stride_k or self.hw_profile.compute_optimal_stride_int()
        self.static_gpu_residents = static_gpu_residents or set()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.gpu_device = gpu_device
        self.step_count = 0

        self.scheduler = DOSInterleavedScheduler(
            num_subgroups=len(subgroups),
            stride_k=self.stride_k,
            static_gpu_residents=self.static_gpu_residents,
        )

    @staticmethod
    def on_the_fly_gradient_flush(
        grad_fp16_gpu: torch.Tensor,
        dest_fp32_host: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements chunk-wise on-the-fly GPU FP16 -> FP32 conversion
        and fast pinned DMA flush (Section 4.1).
        Converts in GPU high-bandwidth VRAM (1.2 TB/s) then writes to pinned host memory.
        """
        grad_fp32_gpu = grad_fp16_gpu.float()  # Fast on GPU
        dest_fp32_host.copy_(grad_fp32_gpu.to(dest_fp32_host.device))
        return dest_fp32_host

    def _adamw_update_tensor(
        self,
        p: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core AdamW update formula."""
        beta1, beta2 = self.betas
        # Update biased 1st and 2nd moment estimate
        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

        # Bias correction
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        step_size = self.lr / bias_correction1
        denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

        # Weight decay
        if self.weight_decay != 0:
            p.add_(p, alpha=-self.lr * self.weight_decay)

        # Step update
        p.addcdiv_(m, denom, value=-step_size)
        return p, m, v

    def step(self) -> Dict[str, Union[int, float, List]]:
        """
        Performs one full training update step across all subgroups
        following Algorithm 1 interleaved schedule.
        """
        self.step_count += 1
        schedule = self.scheduler.plan_schedule()

        stats = {
            "step": self.step_count,
            "stride_k": self.stride_k,
            "gpu_updates": 0,
            "cpu_updates": 0,
            "downscales": 0,
            "flushes": 0,
            "prefetches": 0,
        }

        for item in schedule:
            sg_id = item["subgroup_id"]
            sg = self.subgroups[sg_id]
            target = item["target"]

            if target == "GPU":
                stats["gpu_updates"] += 1
                # Move optimizer state to GPU if not resident
                p_dev = sg.p_fp32.to(self.gpu_device)
                m_dev = sg.m_fp32.to(self.gpu_device)
                v_dev = sg.v_fp32.to(self.gpu_device)
                g_dev = sg.grad_fp32.to(self.gpu_device)

                # Compute GPU update
                p_dev, m_dev, v_dev = self._adamw_update_tensor(p_dev, m_dev, v_dev, g_dev, self.step_count)

                # Update FP16 model parameters on GPU (model_G16 <- p_G32.half())
                sg.p_fp16 = p_dev.half().to(sg.device)

                # Store back FP32 master variables
                sg.p_fp32.copy_(p_dev.to(sg.device))
                sg.m_fp32.copy_(m_dev.to(sg.device))
                sg.v_fp32.copy_(v_dev.to(sg.device))

                # Asynchronously downscale pending CPU subgroups
                for d_id in item["async_downscale_ids"]:
                    d_sg = self.subgroups[d_id]
                    d_sg.p_fp16.copy_(d_sg.p_fp32.half())
                    stats["downscales"] += 1

            else:
                stats["cpu_updates"] += 1
                # Pipeline flush and prefetch telemetry
                if item["flush_out_id"] is not None:
                    stats["flushes"] += 1
                if item["prefetch_in_id"] is not None:
                    stats["prefetches"] += 1

                # CPU update
                self._adamw_update_tensor(
                    sg.p_fp32,
                    sg.m_fp32,
                    sg.v_fp32,
                    sg.grad_fp32,
                    self.step_count,
                )

        # Flush any remaining downscales
        for sg in self.subgroups.values():
            sg.p_fp16.copy_(sg.p_fp32.half())

        return stats
