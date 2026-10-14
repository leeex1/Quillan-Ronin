#!/usr/bin/env python3
# EvoMoE (2505.23830) + GLM-5.3 MoE-DSA config — 34 heterogeneous council w/ shared expert + noaux_tc
import torch, torch.nn as nn, torch.nn.functional as F

class EvoMoE(nn.Module):
    """Heterogeneous grouped experts: 34 experts in 3 groups (small/med/large) per MoHGE/MoDSE.
    GLM-5.3 adoptions: 1 shared expert (DeepSeekMoE fine-grained) + noaux_tc sigmoid scoring + routed scaling."""
    def __init__(self, hidden_dim=2048, n_experts=34, rank=24):
        super().__init__()
        self.n_experts, self.hidden_dim = n_experts, hidden_dim
        # Heterogeneous: 11 small (rank 8), 11 medium (rank 24), 12 large (rank 48) per MoDSE
        self.groups = [(0, 11, 8), (11, 22, 24), (22, 34, 48)]
        self.experts = nn.ModuleList()
        for i in range(n_experts):
            g_rank = [r for s, e, r in self.groups if s <= i < e][0]
            self.experts.append(nn.Sequential(
                nn.Linear(hidden_dim, g_rank, bias=False),
                nn.Linear(g_rank, hidden_dim, bias=False)))
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)
        self.evolution_scale = 0.01  # EGGROLL-style perturbation
        # GLM-5.3 config → EvoMoE: 1 shared expert (DeepSeekMoE fine-grained) + noaux_tc scaling
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False))
        self.shared_gate = nn.Parameter(torch.tensor(2.5))   # GLM routed_scaling_factor

    def evolve_from_base(self, base_state):
        # Evolutionary update: perturb base to create diverse experts (EGGROLL crossover)
        for expert in self.experts:
            for p in expert.parameters():
                p.data.add_(torch.randn_like(p) * self.evolution_scale)

    def forward(self, x):
        # Token-aware routing (EvoMoE): each token gets its own expert mix
        logits = self.router(x)  # [B,S,34]
        # noaux_tc (GLM topk_method): sigmoid scoring, no aux load-loss needed
        gates = torch.sigmoid(logits)
        top_gates, top_idx = gates.topk(4, dim=-1)
        top_gates = top_gates / top_gates.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for b in range(x.shape[0]):
            for s in range(x.shape[1]):
                for k in range(4):
                    eid = top_idx[b, s, k].item()
                    out[b, s] += top_gates[b, s, k] * self.experts[eid](x[b, s])
        out = out + self.shared_expert(x) * torch.sigmoid(self.shared_gate)  # shared expert path
        return out