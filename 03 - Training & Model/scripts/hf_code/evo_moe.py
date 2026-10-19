#!/usr/bin/env python3
# EvoMoE (2505.23830) — Expert Evolution for 33 Council (HNMoE)
# Evolves diverse experts from single base via evolutionary perturbations + token-aware routing
import torch, torch.nn as nn, torch.nn.functional as F

class EvoMoE(nn.Module):
    """Heterogeneous grouped experts: 33 experts in 3 groups (small/med/large) per MoHGE/MoDSE"""
    def __init__(self, hidden_dim=2048, n_experts=33, rank=24):
        super().__init__()
        self.n_experts, self.hidden_dim = n_experts, hidden_dim
        # Heterogeneous: 11 small (rank 8), 11 medium (rank 24), 11 large (rank 48) per MoDSE
        self.groups = [(0,11,8),(11,22,24),(22,34,48)]
        self.experts = nn.ModuleList()
        for i in range(n_experts):
            g_rank = [r for s,e,r in self.groups if s <= i < e][0]
            self.experts.append(nn.Sequential(nn.Linear(hidden_dim, g_rank, bias=False), nn.Linear(g_rank, hidden_dim, bias=False)))
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)
        self.evolution_scale = 0.01  # EGGROLL-style perturbation

    def evolve_from_base(self, base_state):
        # Evolutionary update: perturb base to create diverse experts (EGGROLL crossover)
        for expert in self.experts:
            for p in expert.parameters():
                p.data.add_(torch.randn_like(p) * self.evolution_scale)

    def forward(self, x):
        # Token-aware routing (EvoMoE): each token gets its own expert mix
        logits = self.router(x)  # [B,S,33]
        gates = F.softmax(logits, dim=-1)
        # Top-4 heterogeneous routing (like Mixtral but grouped)
        top_gates, top_idx = gates.topk(4, dim=-1)
        
        # Vectorized dispatch across token slots to replace O(B*S*4) nested Python loop
        B, S, D = x.shape
        flat_x = x.reshape(-1, D)
        flat_top_idx = top_idx.reshape(-1, 4)
        flat_top_gates = top_gates.reshape(-1, 4)
        out_vec = torch.zeros_like(flat_x)
        
        for eid in range(self.n_experts):
            tok_pos, k_slot = (flat_top_idx == eid).nonzero(as_tuple=True)
            if tok_pos.numel() > 0:
                w = flat_top_gates[tok_pos, k_slot].unsqueeze(-1)
                e_out = self.experts[eid](flat_x[tok_pos])
                out_vec.index_add_(0, tok_pos, (w * e_out).to(out_vec.dtype))
                
        return out_vec.reshape(B, S, D)

