"""
DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs
(Zeyu Zhu, Gang Li, Peisong Wang, Zitao Mo, Minnan Pei, Zhuoran Song, Xiaoyao Liang, Jian Cheng - CAS IA & SJTU, arXiv:2602.03495v1).

Mathematical Foundations & Algorithmic Invariants:
---------------------------------------------------
1. 0-1 Integer Optimization Objective for CPU-GPU Balancing (Eq. 3-9):
     min max(T_gpu, T_cpu)
   where:
     T_cpu = sum_{i=1}^N t_cpu(w_i) * C_i
     T_gpu = sum_{i=1}^N t_gpu(w_i) * G_i
     t_gpu(w_i) = max(Trans_expert(w_i), compute_expert(w_i))
     Trans_expert(w_i) = 0 if expert i is cached on GPU else t_trans
   Constraints:
     C_i + G_i in {0, 1} (mutual exclusion, Eq. 8)
     sum (C_i + G_i) = expert_num (activation constraint, Eq. 7)
     sum G_i * size(E_i) <= M_gpu (GPU memory constraint, Eq. 9)

2. Greedy Assignment Strategy (Algorithm 1):
   - Sorts activated experts by descending difference: delta = |t_gpu(w_i) - t_cpu(w_i)|.
   - Greedily allocates to the device minimizing cumulative latency (T_gpu + t_gpu vs T_cpu + t_cpu).
   - Delivers up to 92% of optimal integer program performance with only 5% solving overhead.

3. Residual-Based Prefetching (Eq. 10, 11, Fig. 10):
   - Exploits inter-layer similarity via residual connections:
       h_tilde^(l) = hidden_states^(l) + res_vec^(l)
       predict_expert^(l+1) = gate_func^(l+1)(h_tilde^(l))
     where res_vec^(l) = (1 / N) * sum (hidden_states^(l+1) - hidden_states^(l)).
   - Accurately predicts high-workload experts for layer l+1 to overlap PCIe transfer with layer l computation.

4. Workload-Aware Cache Replacement (Algorithm 2, Eq. 12, Fig. 11):
   - Tracks accumulated workload scores s_k = sum_{i=1}^{w_size} workload_i^k over sliding window w_size.
   - At window boundaries, evicts u_size lowest-score experts on GPU and promotes u_size highest-score experts from CPU.
   - Captures temporal locality of high-workload experts, elevating cache hit rate from ~25% to >56%.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardwareProfile:
    """
    Hardware execution and transmission latency profile for heterogeneous local PC setups
    (e.g. RTX 3090/4090 GPU + Host CPU + PCIe 4.0 x16).
    """
    def __init__(
        self,
        gpu_tflops: float = 35.0,        # RTX 3090/4090 FP16/INT8 TFLOPS
        cpu_tflops: float = 1.5,         # Modern Multi-Core Host CPU TFLOPS
        pcie_bandwidth_gb_s: float = 32.0, # PCIe 4.0 x16 bandwidth in GB/s
        expert_param_mb: float = 64.0,   # Parameter size of a single expert in MB
        base_kernel_launch_ms: float = 0.05
    ):
        self.gpu_tflops = gpu_tflops
        self.cpu_tflops = cpu_tflops
        self.pcie_bandwidth_gb_s = pcie_bandwidth_gb_s
        self.expert_param_mb = expert_param_mb
        self.base_kernel_launch_ms = base_kernel_launch_ms
        
        # t_trans in milliseconds: (MB / 1024) / (GB/s) * 1000
        self.t_trans_ms = (self.expert_param_mb / 1024.0) / self.pcie_bandwidth_gb_s * 1000.0

    def compute_cpu_time(self, workload_tokens: int, flop_per_token: float = 2e6) -> float:
        """Returns CPU execution time in ms for given token workload."""
        if workload_tokens <= 0:
            return 0.0
        gflops_needed = (workload_tokens * flop_per_token) / 1e9
        return (gflops_needed / (self.cpu_tflops * 1000.0)) * 1000.0

    def compute_gpu_time(
        self,
        workload_tokens: int,
        is_cached: bool,
        flop_per_token: float = 2e6
    ) -> float:
        """
        Returns GPU execution time in ms (Paper Eq. 5 & 6):
        t_gpu(w_i) = max(Trans_expert(w_i), compute_expert(w_i))
        """
        if workload_tokens <= 0:
            return 0.0
        compute_ms = self.base_kernel_launch_ms + ((workload_tokens * flop_per_token) / 1e9) / (self.gpu_tflops * 1000.0) * 1000.0
        trans_ms = 0.0 if is_cached else self.t_trans_ms
        return max(trans_ms, compute_ms)


class DALIGreedyAssignment:
    """
    Implements DALI Greedy Assignment Strategy (Algorithm 1, Eq. 3-9).
    Solves 0-1 integer optimization problem min max(T_gpu, T_cpu) at runtime.
    """
    def __init__(self, hw_profile: HardwareProfile, max_gpu_experts: int = 16):
        self.hw = hw_profile
        self.max_gpu_experts = max_gpu_experts

    def assign_experts(
        self,
        workloads: Dict[int, int],
        cached_gpu_experts: Set[int]
    ) -> Dict[str, Any]:
        """
        Assigns activated experts between CPU and GPU.
        - workloads: {expert_id: token_count}
        - cached_gpu_experts: set of expert_ids currently resident in GPU memory.
        """
        num_experts = len(workloads)
        c_cpu = [0] * num_experts
        g_gpu = [0] * num_experts
        t_cpu_total = 0.0
        t_gpu_total = 0.0

        t_gpu_list = []
        t_cpu_list = []
        delta_list = []

        for e_id in range(num_experts):
            w = workloads.get(e_id, 0)
            if w == 0:
                t_g = 0.0
                t_c = 0.0
            else:
                is_cached = (e_id in cached_gpu_experts)
                t_g = self.hw.compute_gpu_time(w, is_cached=is_cached)
                t_c = self.hw.compute_cpu_time(w)
            
            t_gpu_list.append(t_g)
            t_cpu_list.append(t_c)
            delta_list.append(abs(t_g - t_c))

        # Sort activated experts by descending delta (Line 5)
        sorted_indices = sorted(
            [i for i in range(num_experts) if workloads.get(i, 0) > 0],
            key=lambda idx: delta_list[idx],
            reverse=True
        )

        gpu_count = 0
        for idx in sorted_indices:
            t_g = t_gpu_list[idx]
            t_c = t_cpu_list[idx]

            # Greedy choice: assign to device minimizing cumulative time (Lines 12-18)
            # Subject to GPU capacity constraint (Eq. 9)
            can_assign_gpu = (gpu_count < self.max_gpu_experts)
            if can_assign_gpu and ((t_gpu_total + t_g) <= (t_cpu_total + t_c)):
                g_gpu[idx] = 1
                t_gpu_total += t_g
                gpu_count += 1
            else:
                c_cpu[idx] = 1
                t_cpu_total += t_c

        makespan = max(t_gpu_total, t_cpu_total)
        imbalance_ratio = abs(t_gpu_total - t_cpu_total) / (makespan + 1e-12)

        return {
            "c_cpu": c_cpu,
            "g_gpu": g_gpu,
            "t_gpu_total_ms": t_gpu_total,
            "t_cpu_total_ms": t_cpu_total,
            "makespan_ms": makespan,
            "imbalance_ratio": imbalance_ratio,
            "assigned_gpu": [i for i, v in enumerate(g_gpu) if v == 1],
            "assigned_cpu": [i for i, v in enumerate(c_cpu) if v == 1]
        }


class DALIResidualPrefetcher:
    """
    Implements Residual-Based Prefetching (Section 4.2, Eq. 10 & 11).
    Leverages inter-layer residual vector to predict high-workload experts for layer l+1.
    """
    def __init__(self, hidden_dim: int, num_layers: int):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Layer-specific residual vectors res_vec^(l) in R^d
        self.residual_vectors: Dict[int, torch.Tensor] = {
            l: torch.zeros(hidden_dim) for l in range(num_layers - 1)
        }

    def calibrate_residual_vector(
        self,
        layer_idx: int,
        h_current: torch.Tensor,
        h_next: torch.Tensor
    ):
        """
        Calibrates res_vec^(l) offline (Eq. 11):
        res_vec^(l) = (1 / N) * sum (hidden_states^(l+1) - hidden_states^(l))
        """
        assert layer_idx < self.num_layers - 1
        delta = (h_next - h_current).view(-1, self.hidden_dim)
        self.residual_vectors[layer_idx] = delta.mean(dim=0).detach()

    def predict_next_layer_experts(
        self,
        layer_idx: int,
        h_current: torch.Tensor,
        next_gate_weight: torch.Tensor,
        top_k: int = 2
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Predicts layer l+1 activated experts using adjusted hidden states (Eq. 10).
        Returns predicted token-to-expert mapping and top-k high-workload expert IDs.
        """
        assert layer_idx < self.num_layers - 1
        res_vec = self.residual_vectors[layer_idx].to(h_current.device)
        # 1. Residual correction: h_tilde^(l) = hidden_states^(l) + res_vec^(l)
        h_tilde = h_current + res_vec.view(1, 1, -1)

        # 2. Gate prediction: predict_expert^(l+1) = gate_func^(l+1)(h_tilde)
        logits = torch.matmul(h_tilde, next_gate_weight)  # (B, S, num_experts)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)

        # 3. Identify predicted high-workload experts
        flat_indices = topk_indices.flatten()
        counts = torch.bincount(flat_indices, minlength=next_gate_weight.shape[-1])
        sorted_experts = torch.argsort(counts, descending=True).tolist()
        predicted_high_workload = sorted_experts[:top_k]

        return topk_indices, predicted_high_workload


class DALIWorkloadAwareCache:
    """
    Implements Workload-Aware Cache Replacement (Section 4.3, Algorithm 2, Eq. 12).
    Maintains GPU expert cache with sliding window w_size and replacement batch u_size.
    """
    def __init__(
        self,
        num_experts: int,
        cache_size: int = 4,
        w_size: int = 4,
        u_size: int = 2
    ):
        self.num_experts = num_experts
        self.cache_size = min(cache_size, num_experts)
        self.w_size = w_size
        self.u_size = min(u_size, self.cache_size)

        # Initially cache the first cache_size experts on GPU
        self.expert_on_gpu: Set[int] = set(range(self.cache_size))
        self.expert_on_cpu: Set[int] = set(range(self.cache_size, num_experts))
        
        # Accumulated workload score vector s in R^N
        self.scores = [0.0] * num_experts
        self.tokens_in_window = 0
        self.total_cache_hits = 0
        self.total_accesses = 0

    def record_access(self, expert_workload_vector: List[int]) -> bool:
        """
        Records a step of token workloads and applies sliding window replacement (Algorithm 2).
        Returns True if a cache replacement occurred at this step.
        """
        assert len(expert_workload_vector) == self.num_experts
        replaced = False

        # Accumulate workload scores (Line 6)
        for e_id, w in enumerate(expert_workload_vector):
            self.scores[e_id] += float(w)
            if w > 0:
                self.total_accesses += 1
                if e_id in self.expert_on_gpu:
                    self.total_cache_hits += 1

        self.tokens_in_window += 1

        # Check if window completed (Line 9)
        if self.tokens_in_window >= self.w_size:
            # 1. Find u_size lowest-score experts on GPU to evict (Line 11)
            gpu_candidates = list(self.expert_on_gpu)
            gpu_candidates.sort(key=lambda idx: self.scores[idx])
            evict_from_gpu = set(gpu_candidates[:self.u_size])

            # 2. Find u_size highest-score experts on CPU to promote (Line 10)
            cpu_candidates = list(self.expert_on_cpu)
            cpu_candidates.sort(key=lambda idx: self.scores[idx], reverse=True)
            promote_from_cpu = set(cpu_candidates[:self.u_size])

            # Apply replacement (Lines 12-14)
            self.expert_on_gpu = (self.expert_on_gpu - evict_from_gpu) | promote_from_cpu
            self.expert_on_cpu = (self.expert_on_cpu - promote_from_cpu) | evict_from_gpu

            # Reset scores for new sliding window (Line 15)
            self.scores = [0.0] * self.num_experts
            self.tokens_in_window = 0
            replaced = True

        return replaced

    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.total_cache_hits / self.total_accesses


class DALIMoEOrchestrator(nn.Module):
    """
    End-to-End DALI MoE Layer Orchestrator combining:
    - Greedy Dynamic Assignment between Host CPU and Local PC GPU
    - Residual-Based Prefetching
    - Workload-Aware GPU Expert Cache
    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        cache_size: int = 4,
        w_size: int = 4,
        u_size: int = 2,
        hw_profile: Optional[HardwareProfile] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.hw = hw_profile or HardwareProfile()

        # Gate router
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        # Experts (simulated as linear transformations)
        self.experts = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_experts)
        ])

        # DALI sub-engines
        self.greedy_assigner = DALIGreedyAssignment(self.hw, max_gpu_experts=cache_size)
        self.cache = DALIWorkloadAwareCache(
            num_experts=num_experts,
            cache_size=cache_size,
            w_size=w_size,
            u_size=u_size
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes MoE layer with dynamic DALI scheduling.
        x: (B, S, D)
        """
        B, S, D = x.shape
        flat_x = x.view(-1, D)
        total_tokens = flat_x.shape[0]

        # 1. Routing Gate
        logits = self.gate(flat_x)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)

        # 2. Compute token workloads per expert
        workloads: Dict[int, int] = {e: 0 for e in range(self.num_experts)}
        flat_topk = topk_indices.flatten().tolist()
        for idx in flat_topk:
            workloads[idx] += 1

        workload_vector = [workloads[e] for e in range(self.num_experts)]

        # 3. Workload-Aware Cache Update
        self.cache.record_access(workload_vector)

        # 4. Greedy Assignment
        assignment = self.greedy_assigner.assign_experts(
            workloads=workloads,
            cached_gpu_experts=self.cache.expert_on_gpu
        )

        # 5. Execute computation
        out_flat = torch.zeros_like(flat_x)
        for e_id in range(self.num_experts):
            if workloads[e_id] == 0:
                continue
            # Find tokens assigned to this expert
            mask = (topk_indices == e_id).any(dim=-1)
            if not mask.any():
                continue
            e_tokens = flat_x[mask]
            e_out = self.experts[e_id](e_tokens)

            # Weight by gate score
            # Find which position (0 .. top_k-1) this expert occupies for each token
            pos_mask = (topk_indices[mask] == e_id)
            weights = topk_scores[mask][pos_mask].view(-1, 1)
            out_flat[mask] += e_out * weights

        out = out_flat.view(B, S, D)
        
        telemetry = {
            "workloads": workloads,
            "assignment": assignment,
            "cache_hit_rate": self.cache.hit_rate,
            "cached_gpu_experts": list(self.cache.expert_on_gpu)
        }
        return out, telemetry
