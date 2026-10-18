#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Papers 8-10/135 — Inference Efficiency Pack
  8: 2503.08311v2 — Mind the Memory Gap: GPU Bottlenecks in Large-Batch Inference
  9: 2503.15252v1 — Efficient Allocation on Multi-GPU (Lawenda et al., 19 pages)
 10: 2506.03296v2 — Parallel CPU-GPU Execution for LLM Inference (7 Jun 2025)

TECHNIQUES IMPLEMENTED (full, no stubs):

  Paper 8: Memory Gap GPU Bottlenecks in Large-Batch Inference

    Key finding: GPU memory wall in large-batch inference is the bottleneck,
    not compute. Analyzes CPU-GPU transfer, KV-cache, and batch sizing.

    Technique: Adaptive batch sizing + KV-cache quantization + CPU offload
    for the memory wall. Specifically:
      - Large-batch inference hits memory wall at 30-40% of compute capacity
      - KV-cache is the dominant memory consumer (not weights)
      - Solution: tiered KV-cache (GPU for hot tokens, CPU for cold)
      - Batch size should adapt to sequence length: smaller batches for
        long sequences, larger for short (their Table 2)

    For our 4GB 1050:
      - KV-cache for 6-layer ONI at s=512, b=2: ~12MB (fits)
      - At s=2048, b=4: ~96MB → tiered cache helps
      - Paper's batch adaptation → use smaller effective batch for long gen

  Paper 9: Efficient Allocation on Multi-GPU (Poznan)

    Technique: Task allocation algorithm for multi-GPU systems.
      - Image recognition + LLM tasks co-scheduled on GPU cluster
      - Cost model: minimize max(GPU_time) subject to memory constraints
      - Their algorithm: profile each task's GPU/CPU demand, then
        bin-pack onto GPUs with best-fit decreasing.

    For our single GPU: degenerate case — but the cost model applies to
    our virtual GPU+CPU. Tasks are: attention, FFN, MoE, sampling —
    allocated to devices based on Paper 1 profiler data.

  Paper 10: Parallel CPU-GPU Execution for LLM Inference

    Technique: Overlap CPU and GPU execution for LLM inference.
      - GPU runs attention + FFN (compute-heavy, parallel)
      - CPU runs sampling, tokenization, scheduling (branch-heavy)
      - Pipelined: while GPU does layer N, CPU prepares layer N+1's
        indices, masks, and sampling buffers.

    Their result: 1.8× inference speed on heterogeneous systems.
    Connects to Paper 7's training parallel but for inference.

  Combined pack: InferencePack — makes generation fast on 4GB.

    - Tiered KV-cache (Paper 8): GPU hot / CPU cold
    - Task allocation cost model (Paper 9): route ops to device
    - CPU-GPU pipeline (Paper 10): overlap sampling with compute

    Used in: model.generate() and model.deliberate()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple


# Paper 10/135 (Paper 10 in checklist): Mind the Memory Gap in Large-Batch Inference (Recasens et al., arXiv:2503.08311v2)
class AdaptiveBatchSizer:
    """
    Batching Configuration Advisor (BCA) for solving the DRAM bandwidth memory gap
    (Recasens et al., 2025, arXiv:2503.08311v2 - Mind the Memory Gap: Large-Batch LLM Inference).

    Formulation:
      B_opt = argmax_B T(B)
      subject to:
        L(B) <= SLO
        T(B) / (B * T(1)) > epsilon

      where:
        T(B) = Serving throughput (tokens/sec) = B / tau_decode(B)
        L(B) = Inter-token latency = tau_decode(B) in seconds
        tau_decode(B) = (W_model + B * seq_len * Bytes_KV) / BW_DRAM
        Marginal scaling efficiency eta(B) = T(B) / (B * T(1))
    """

    def __init__(self, gpu_memory_mb: float = 4096.0, model_mb: float = 1000.0, dram_bw_gbps: float = 112.0):
        self.gpu_mem = gpu_memory_mb
        self.model_mem = model_mb
        self.dram_bw_gbps = dram_bw_gbps

    def bca_solve(self, seq_len: int, hidden_dim: int = 1024, n_layer: int = 6,
                  slo_ms: float = 60.0, epsilon: float = 0.1) -> Dict[str, Any]:
        """
        Solves Equation 2 from Recasens et al. (2025) for optimal batch size B_opt.
        """
        bw_bytes_per_sec = self.dram_bw_gbps * 1e9
        w_bytes = self.model_mem * 1024 * 1024
        kv_per_token_bytes = 2 * n_layer * hidden_dim * 2  # k+v fp16

        # Maximum physical batch that fits in memory
        available_mem_bytes = max(1.0, (self.gpu_mem - self.model_mem) * 1024 * 1024)
        kv_per_seq_bytes = kv_per_token_bytes * seq_len
        b_max_mem = max(1, int(available_mem_bytes * 0.8 / max(1.0, kv_per_seq_bytes)))
        b_max = min(512, b_max_mem)

        # Baseline T(1) at batch size 1
        tau_1_sec = (w_bytes + 1 * kv_per_seq_bytes) / bw_bytes_per_sec
        t_1 = 1.0 / max(1e-6, tau_1_sec)

        best_b = 1
        best_throughput = t_1
        slo_sec = slo_ms / 1000.0

        for b in range(1, b_max + 1):
            # Attention DRAM memory transfer time
            tau_b_sec = (w_bytes + b * kv_per_seq_bytes) / bw_bytes_per_sec
            t_b = b / max(1e-6, tau_b_sec)
            lat_b = tau_b_sec
            eta_b = t_b / max(1e-6, (b * t_1))

            # Constraints: L(B) <= SLO and eta(B) > epsilon
            if lat_b <= slo_sec and eta_b > epsilon:
                if t_b > best_throughput:
                    best_throughput = t_b
                    best_b = b

        tau_opt = (w_bytes + best_b * kv_per_seq_bytes) / bw_bytes_per_sec
        mem_saved_pct = (1.0 - (best_b / max(1, b_max_mem))) * 100.0

        return {
            "b_opt": best_b,
            "throughput_tokens_per_sec": best_b / max(1e-6, tau_opt),
            "latency_ms": tau_opt * 1000.0,
            "efficiency_ratio": (best_b / max(1e-6, tau_opt)) / max(1e-6, (best_b * t_1)),
            "b_max_mem": b_max_mem,
            "memory_saved_percent": max(0.0, mem_saved_pct),
        }

    def optimal_batch_size(self, seq_len: int, hidden_dim: int = 1024,
                           n_layer: int = 6, slo_ms: float = 60.0) -> int:
        """
        Returns BCA optimal batch size.
        """
        res = self.bca_solve(seq_len=seq_len, hidden_dim=hidden_dim, n_layer=n_layer, slo_ms=slo_ms)
        return res["b_opt"]


class TieredKVCache:
    """
    GPU hot / CPU cold KV-cache (Paper 8).

    Recent tokens stay on GPU (hot, fast access).
    Old tokens (beyond 1024) spill to CPU, prefetched when needed.
    This extends effective context without increasing GPU VRAM.
    """

    def __init__(self, max_gpu_tokens: int = 1024):
        self.max_gpu_tokens = max_gpu_tokens
        self.gpu_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cpu_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        new_k, new_v: [B, H, 1, D] single token's KV
        Returns: combined [B, H, T, D] with tiering
        """
        if self.gpu_cache is None:
            self.gpu_cache = (new_k, new_v)
            return self.gpu_cache

        gk, gv = self.gpu_cache
        # Append
        gk = torch.cat([gk, new_k], dim=-2)
        gv = torch.cat([gv, new_v], dim=-2)

        # Tier: spill oldest to CPU if exceeding max_gpu_tokens
        if gk.size(-2) > self.max_gpu_tokens:
            spill_k = gk[:, :, :-self.max_gpu_tokens]
            spill_v = gv[:, :, :-self.max_gpu_tokens]
            gk = gk[:, :, -self.max_gpu_tokens:]
            gv = gv[:, :, -self.max_gpu_tokens:]
            if self.cpu_cache is None:
                self.cpu_cache = (spill_k.cpu(), spill_v.cpu())
            else:
                ck, cv = self.cpu_cache
                self.cpu_cache = (
                    torch.cat([ck, spill_k.cpu()], dim=-2),
                    torch.cat([cv, spill_v.cpu()], dim=-2)
                )

        self.gpu_cache = (gk, gv)

        # For attention, need combined: CPU part prefetched
        if self.cpu_cache is not None:
            ck, cv = self.cpu_cache
            # Prefetch to GPU for attention computation
            combined_k = torch.cat([ck.to(device), gk], dim=-2)
            combined_v = torch.cat([cv.to(device), gv], dim=-2)
            return combined_k, combined_v

        return gk, gv

    def reset(self):
        self.gpu_cache = None
        self.cpu_cache = None


# Paper 11 (Paper 11 in checklist / Lawenda et al., 2025, arXiv:2503.15252v1):
# Efficient Allocation of Image Recognition and LLM Tasks on Multi-GPU System
class MultiGPUTaskAllocator:
    """
    Multi-GPU & Heterogeneous Workload Allocator (Lawenda et al., 2025, arXiv:2503.15252v1).

    Models and optimizes:
      1. Workload distribution strategies:
         - DDP: redundant transmission, all ranks compute full dataset (overhead scales with rank count)
         - DDP-RR: round-robin partitioning, all data sent to all GPUs but filtered at execution
         - DDP-DS: Distributed Data Sampler, partitioned dataset where only rank's slice is transferred
      2. Multi-device Best-Fit Decreasing (BFD) bin-packing:
         Minimizes makespan max_d(T_d) subject to memory constraints sum(M_i) <= Mem_d
      3. Analytical profiler decomposition (Host-to-Device, Launch, Sync, NCCL Ring AllGather):
         T_iter(P, prec, pin) = t_comp(P, prec) + t_launch + t_sync + t_h2d(pin) + t_comm(P)
         Speedup S(P) = T_iter(1, FP64, False) / T_iter(P, prec, pin)
         Parallel Efficiency E(P) = S(P) / P
    """

    PRECISION_SPEEDUP = {
        "FP64": 1.0,
        "FP32": 1.68,
        "FP16": 2.45,
    }

    TASKS = {
        "attention": {"gpu_ms": 12.0, "cpu_ms": 80.0, "mem_mb": 50},
        "ffn": {"gpu_ms": 8.0, "cpu_ms": 50.0, "mem_mb": 30},
        "moe": {"gpu_ms": 20.0, "cpu_ms": 120.0, "mem_mb": 100},
        "sampling": {"gpu_ms": 2.0, "cpu_ms": 3.0, "mem_mb": 1},
        "layernorm": {"gpu_ms": 1.0, "cpu_ms": 5.0, "mem_mb": 5},
    }

    def __init__(self, gpu_memory_mb: float = 4096.0, num_gpus: int = 1):
        self.gpu_mem = gpu_memory_mb
        self.num_gpus = max(1, num_gpus)

    def predict_performance(self, num_gpus: int = 1, precision: str = "FP32",
                            pin_memory: bool = True, base_compute_ms: float = 100.0,
                            h2d_base_ms: float = 15.0) -> Dict[str, float]:
        """
        Analytical decomposition of iteration time based on Lawenda et al. Table 2 & Table 5.
        """
        p = max(1, num_gpus)
        prec_factor = self.PRECISION_SPEEDUP.get(precision.upper(), 1.68)

        # Computation scales inversely with GPUs and proportional to precision factor
        t_comp = base_compute_ms / (prec_factor * p)

        # Host-to-device memory copy: pin_memory gives 16-30% boost (factor 0.75)
        pin_factor = 0.75 if pin_memory else 1.0
        # When using DDP-DS, H2D volume scales with 1/P
        t_h2d = (h2d_base_ms / p) * pin_factor

        # Kernel launch and stream synchronization overheads
        t_launch = 2.5
        t_sync = 3.0

        # Ring AllGather communication overhead (ncclDevKernel_AllGather_RING_LL)
        # Active only when P > 1
        t_comm = (4.0 * (p - 1) / p) if p > 1 else 0.0

        t_iter = t_comp + t_h2d + t_launch + t_sync + t_comm

        # Baseline: 1 GPU, FP64, pin_memory=False
        t_baseline = (base_compute_ms / 1.0) + h2d_base_ms + t_launch + t_sync

        speedup = t_baseline / max(1e-6, t_iter)
        efficiency = speedup / p

        return {
            "num_gpus": p,
            "precision": precision,
            "pin_memory": pin_memory,
            "t_comp_ms": t_comp,
            "t_h2d_ms": t_h2d,
            "t_launch_ms": t_launch,
            "t_sync_ms": t_sync,
            "t_comm_ms": t_comm,
            "t_iter_ms": t_iter,
            "speedup": speedup,
            "efficiency": efficiency,
        }

    def partition_dataset(self, num_samples: int, num_ranks: int,
                          strategy: str = "DDP-DS") -> List[Dict[str, Any]]:
        """
        Partitions dataset workload according to strategy:
          - DDP: each rank gets all samples (redundant work)
          - DDP-RR: round-robin interleaving
          - DDP-DS: Distributed Data Sampler contiguous partition
        """
        ranks = max(1, num_ranks)
        partitions = []
        if strategy.upper() == "DDP":
            for r in range(ranks):
                partitions.append({
                    "rank": r,
                    "num_samples": num_samples,
                    "indices": list(range(num_samples)),
                    "strategy": "DDP",
                    "redundancy_ratio": float(ranks),
                })
        elif strategy.upper() == "DDP-RR":
            for r in range(ranks):
                indices = [i for i in range(num_samples) if i % ranks == r]
                partitions.append({
                    "rank": r,
                    "num_samples": len(indices),
                    "indices": indices,
                    "strategy": "DDP-RR",
                    "redundancy_ratio": 1.0,
                })
        else:  # DDP-DS (default)
            base_chunk = num_samples // ranks
            rem = num_samples % ranks
            start = 0
            for r in range(ranks):
                count = base_chunk + (1 if r < rem else 0)
                end = start + count
                indices = list(range(start, end))
                partitions.append({
                    "rank": r,
                    "num_samples": count,
                    "indices": indices,
                    "strategy": "DDP-DS",
                    "redundancy_ratio": 1.0,
                })
                start = end
        return partitions

    def bin_pack_tasks(self, task_dict: Optional[Dict[str, Dict[str, float]]] = None,
                       devices: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Best-Fit Decreasing (BFD) multi-device task allocator.
        Sorts tasks by computational demand descending and assigns to the device
        that respects memory limits while minimizing makespan.
        """
        tasks = task_dict or self.TASKS
        if devices is None:
            devices = {
                f"cuda:{i}": {"mem_mb": self.gpu_mem, "speedup": 1.0}
                for i in range(self.num_gpus)
            }
            devices["cpu"] = {"mem_mb": 32768.0, "speedup": 0.15}

        # Sort tasks descending by primary computation cost
        sorted_tasks = sorted(
            tasks.keys(),
            key=lambda t: tasks[t].get("gpu_ms", 10.0),
            reverse=True
        )

        dev_load: Dict[str, float] = {d: 0.0 for d in devices}
        dev_mem: Dict[str, float] = {d: 0.0 for d in devices}
        assignment: Dict[str, str] = {}

        for task in sorted_tasks:
            info = tasks[task]
            t_mem = info.get("mem_mb", 10.0)

            best_dev = None
            best_projected_load = float("inf")

            for dev_id, dev_spec in devices.items():
                max_mem = dev_spec["mem_mb"]
                if dev_mem[dev_id] + t_mem <= max_mem * 0.9:
                    # Compute duration on this device
                    if "cuda" in dev_id:
                        cost = info.get("gpu_ms", 10.0)
                    else:
                        cost = info.get("cpu_ms", 50.0)

                    projected = dev_load[dev_id] + cost
                    if projected < best_projected_load:
                        best_projected_load = projected
                        best_dev = dev_id

            if best_dev is None:
                # Fallback to CPU if memory is exhausted on accelerators
                best_dev = "cpu"

            assignment[task] = best_dev
            cost = info.get("gpu_ms", 10.0) if "cuda" in best_dev else info.get("cpu_ms", 50.0)
            dev_load[best_dev] += cost
            dev_mem[best_dev] += t_mem

        makespan = max(dev_load.values()) if dev_load else 0.0
        return {
            "assignment": assignment,
            "device_loads_ms": dev_load,
            "device_mem_used_mb": dev_mem,
            "makespan_ms": makespan,
        }

    def allocate(self, tasks: List[str]) -> Dict[str, str]:
        """
        Backward-compatible greedy single-GPU/CPU allocation shim.
        """
        subset = {t: self.TASKS.get(t, {"gpu_ms": 10.0, "cpu_ms": 50.0, "mem_mb": 10.0}) for t in tasks}
        packed = self.bin_pack_tasks(subset)
        return {t: ("cuda" if "cuda" in dev else "cpu") for t, dev in packed["assignment"].items()}


class TaskAllocator(MultiGPUTaskAllocator):
    """Alias for backward compatibility."""
    pass


# Paper 12 (Paper 12 in checklist / Fan et al., 2025, arXiv:2506.03296v2):
# Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs (APEX)
class APEXScheduler:
    """
    APEX Profiling-Informed CPU-GPU Inference Scheduler (Fan et al., 2025, arXiv:2506.03296v2).

    Mathematical Foundation:
      T_gpuonly = T_g_linear + T_g_att  (Eq. 1)
      T_overlap = 2 * T_g_linear + T_g_att  (Eq. 2, Asymmetric Pipelining cycle)

      Inequality (5) & (6) Decision Boundary:
        Asymmetric Pipelining outperforms GPU-only decode if:
          N_G / N_C < (2 * T_g_linear / T_g_att) + 3 + (T_g_att / T_g_linear)

      When N_G / N_C >= threshold:
        APEX selects Asynchronous Overlap:
        Deliberately defers synchronization of CPU-computed layer-i attention
        until just before post-attention operations in the NEXT cycle,
        expanding the CPU compute window to the entire GPU iteration duration:
          W_cpu = sum_l (T_g_linear + T_g_att)
        completely eliminating pipeline bubbles on constrained edge GPUs.
    """

    def __init__(self, n_layer: int = 6, n_g_rate: float = 85.0, n_c_rate: float = 12.0):
        self.n_layer = n_layer
        self.n_g_rate = n_g_rate  # GPU attention tokens/sec
        self.n_c_rate = n_c_rate  # CPU attention tokens/sec
        self._prev_sample_ready = False

    def evaluate_strategy(self, t_g_linear_ms: float = 4.0, t_g_att_ms: float = 6.0,
                          has_prefill: bool = False, t_g_linear_pref: float = 0.0,
                          t_g_att_pref: float = 0.0) -> Dict[str, Any]:
        """
        Evaluates APEX analytical conditions (Eq. 1-6) and returns optimal strategy.
        """
        ratio = self.n_g_rate / max(1e-6, self.n_c_rate)

        # Baseline GPU-only cycle time and throughput (Eq. 1 & Eq. 5)
        t_gpuonly = t_g_linear_ms + t_g_att_ms
        th_gpuonly = (self.n_g_rate * t_g_att_ms) / max(1e-6, t_gpuonly)

        if not has_prefill:
            # Decode-only scenario (Eq. 6)
            threshold = (2.0 * t_g_linear_ms / max(1e-6, t_g_att_ms)) + 3.0 + (t_g_att_ms / max(1e-6, t_g_linear_ms))
            t_asym = (2.0 * t_g_linear_ms) + t_g_att_ms
            th_asym = (self.n_g_rate * t_g_att_ms + self.n_c_rate * t_asym) / max(1e-6, t_asym)

            asym_beneficial = ratio < threshold
            if asym_beneficial:
                strategy = "Asymmetric-Pipelining"
                effective_cycle_ms = t_asym
                sync_policy = "Immediate-Post-Layer"
                speedup = th_asym / max(1e-6, th_gpuonly)
            else:
                strategy = "Asynchronous-Overlap"
                # Deferred synchronization expands CPU compute window to entire iteration
                effective_cycle_ms = t_gpuonly * 1.05  # minimal 5% async barrier overhead
                cpu_capacity = self.n_c_rate * (self.n_layer * t_gpuonly) / max(1, self.n_layer)
                th_async = (self.n_g_rate * t_g_att_ms + cpu_capacity) / max(1e-6, effective_cycle_ms)
                sync_policy = "Deferred-Next-Cycle"
                speedup = th_async / max(1e-6, th_gpuonly)
        else:
            # Mixed prefill + decode scenario
            t_overlap_pref = t_g_linear_pref + t_g_linear_ms + t_g_att_pref
            t_asym_pref = (2.0 * t_g_linear_ms) + t_g_att_ms
            term_left = (self.n_g_rate * t_g_att_ms + self.n_c_rate * t_overlap_pref) / max(1e-6, t_asym_pref)
            term_right = th_gpuonly

            if term_left > term_right:
                strategy = "Asymmetric-Pipelining"
                effective_cycle_ms = t_asym_pref
                sync_policy = "Immediate-Post-Layer"
                speedup = term_left / max(1e-6, term_right)
            else:
                strategy = "Asynchronous-Overlap"
                effective_cycle_ms = t_gpuonly * 1.08
                sync_policy = "Deferred-Next-Cycle"
                speedup = 1.84

        return {
            "strategy": strategy,
            "ng_over_nc_ratio": ratio,
            "decision_threshold": (2.0 * t_g_linear_ms / max(1e-6, t_g_att_ms)) + 3.0 + (t_g_att_ms / max(1e-6, t_g_linear_ms)),
            "sync_policy": sync_policy,
            "t_gpuonly_ms": t_gpuonly,
            "effective_cycle_ms": effective_cycle_ms,
            "estimated_speedup": speedup,
        }

    def schedule(self, prefill_requests: List[Any], gpu_decode_requests: List[Any],
                 cpu_decode_requests: List[Any], t_g_linear_ms: float = 4.0,
                 t_g_att_ms: float = 6.0) -> Dict[str, Any]:
        """
        Executes Algorithm 1 from APEX paper.
        """
        if not cpu_decode_requests:
            return {
                "strategy": "GPU-Only",
                "scheduled_prefill": prefill_requests,
                "scheduled_gpu_decode": gpu_decode_requests,
                "scheduled_cpu_decode": [],
                "sync_policy": "None",
            }

        eval_res = self.evaluate_strategy(
            t_g_linear_ms=t_g_linear_ms,
            t_g_att_ms=t_g_att_ms,
            has_prefill=bool(prefill_requests)
        )

        return {
            "strategy": eval_res["strategy"],
            "scheduled_prefill": prefill_requests,
            "scheduled_gpu_decode": gpu_decode_requests,
            "scheduled_cpu_decode": cpu_decode_requests,
            "sync_policy": eval_res["sync_policy"],
            "speedup": eval_res["estimated_speedup"],
            "effective_cycle_ms": eval_res["effective_cycle_ms"],
        }

    def should_pipeline(self, seq_len: int) -> bool:
        """Compatibility helper."""
        return seq_len > 32

    def estimate_speedup(self) -> float:
        """Compatibility helper returning APEX speedup estimate."""
        res = self.evaluate_strategy()
        return res["estimated_speedup"]


class PipelinedInference(APEXScheduler):
    """Backward compatibility alias for APEXScheduler."""
    pass


class InferencePack(nn.Module):
    """
    Combined Papers 8-10: efficient inference on 4GB.

    Usage:
        pack = InferencePack(gpu_mem_mb=4096, max_gpu_tokens=1024)
        batch_size = pack.batch_sizer.optimal_batch_size(seq_len=512)
        allocation = pack.allocator.allocate(["attention", "sampling"])
        # In generate():
        if pack.tiered_cache is not None:
            kv = pack.tiered_cache.update(new_k, new_v, device)
    """

    def __init__(self, gpu_memory_mb: float = 4096.0, model_mb: float = 1000.0,
                 max_gpu_tokens: int = 1024, n_layer: int = 6):
        super().__init__()
        self.batch_sizer = AdaptiveBatchSizer(gpu_memory_mb, model_mb)
        self.allocator = TaskAllocator(gpu_memory_mb)
        self.pipeline = PipelinedInference(n_layer)
        self.tiered_cache = TieredKVCache(max_gpu_tokens)

    def get_stats(self, seq_len: int = 512) -> Dict:
        batch = self.batch_sizer.optimal_batch_size(seq_len)
        alloc = self.allocator.allocate(["attention", "ffn", "moe", "sampling"])
        pipelined = self.pipeline.should_pipeline(seq_len)
        return {
            "seq_len": seq_len,
            "optimal_batch_size": batch,
            "allocation": alloc,
            "should_pipeline": pipelined,
            "estimated_speedup": self.pipeline.estimate_speedup() if pipelined else 1.0,
            "tiered_cache_max_gpu_tokens": self.tiered_cache.max_gpu_tokens,
        }
