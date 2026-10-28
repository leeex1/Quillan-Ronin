#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EvoMoE (2505.23830) — Expert Evolution & Individualized Micro-Agent Swarm for 34 Council (HNMoE)
=================================================================================================
Enforces the Canonical Sovereign Council & Swarm Laws across all 34 Council Members:
  - Law 2: All 34 Council members (grouped into 11x rank-8, 11x rank-24, 12x rank-48) debate
           queries using their own micro-agent swarms.
  - Law 3: Each Council member configures its swarm via parent projection W1 [r, D] and
           individualizes each micro-agent m in 0..r-1 via distinct phase/diversity gates.
  - Law 4: Each micro-agent path is incorporated by the parent Council member via W2 [D, r]
           to form that Council member's proposal e_all [BT, 34, D].
  - Law 5: All 34 Council proposals report back to Quillan Core via Throne pull consensus.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_ENV_VEC = os.environ.get("QUILLAN_EVOMOE_VECTORIZED", "").strip()


def _default_vectorized() -> bool:
    if _ENV_VEC:
        return _ENV_VEC.lower() in ("1", "true", "yes", "on")
    return True


class EvoMoE(nn.Module):
    """Heterogeneous grouped Council experts: 34 experts in 3 rank groups (8 / 24 / 48 micro-agents)."""

    PAD_WASTE_LIMIT = 8.0

    def __init__(self, hidden_dim=2048, n_experts=34, rank=24):
        super().__init__()
        self.n_experts, self.hidden_dim = n_experts, hidden_dim
        # Heterogeneous: 11 small (rank 8), 11 medium (rank 24), 12 large (rank 48) per MoDSE
        self.groups = [(0, 11, 8), (11, 22, 24), (22, 34, 48)]
        self.experts = nn.ModuleList()
        for i in range(n_experts):
            g_rank = [r for s, e, r in self.groups if s <= i < e][0]
            self.experts.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, g_rank, bias=False),
                    nn.Linear(g_rank, hidden_dim, bias=False),
                )
            )
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)
        self.evolution_scale = 0.01  # EGGROLL-style perturbation
        self.vectorized = _default_vectorized()
        self._wcache = None
        self._wkey = None

    def evolve_from_base(self, base_state):
        # Evolutionary update: perturb base to create diverse experts (EGGROLL crossover)
        for expert in self.experts:
            for p in expert.parameters():
                p.data.add_(torch.randn_like(p) * self.evolution_scale)

    def _group_weights(self, dtype: torch.dtype):
        """Returns [(lo, hi, r, W1[n,r,D], W2[n,D,r], micro_profile[1,n,r]), ...].
        Caches only when grad is disabled so multi-step grad_accum never reuses a freed graph."""
        if not torch.is_grad_enabled():
            key = (dtype,) + tuple(p._version for e in self.experts for p in e.parameters())
            if self._wkey == key and self._wcache is not None:
                return self._wcache
        else:
            self._wcache, self._wkey = None, None

        out = []
        dev = self.experts[0][0].weight.device
        for lo, hi, r in self.groups:
            hi = min(hi, self.n_experts)
            n = hi - lo
            if n <= 0:
                continue
            W1 = torch.stack([self.experts[lo + i][0].weight for i in range(n)], dim=0).to(dtype)  # [n, r, D]
            W2 = torch.stack([self.experts[lo + i][1].weight for i in range(n)], dim=0).to(dtype)  # [n, D, r]
            # Deterministic per-(expert, micro-agent) individualization signature in [0.85, 1.15]
            e_idx = torch.arange(lo, hi, device=dev, dtype=torch.float32).view(1, n, 1)
            m_idx = torch.arange(r, device=dev, dtype=torch.float32).view(1, 1, r)
            micro_profile = (1.0 + 0.15 * torch.sin(e_idx * 0.73 + m_idx * 1.17)).to(dtype)
            out.append((lo, hi, r, W1, W2, micro_profile))

        if not torch.is_grad_enabled():
            self._wcache, self._wkey = out, key
        return out

    def forward_council(self, x: torch.Tensor, pull: torch.Tensor = None, gov_scale: float = 1.0):
        """
        Executes all 34 Council members and their individualized micro-agent swarms in 3 batched einsums.
        Returns:
          moe_out: [BT, D] pull-weighted Council synthesis
          combined_pull: [BT, 34] fp32 consensus pull weights
          e_pooled: [34, D] mean expert representations for AQCS Born-rule consensus
        """
        if x.dim() == 3:
            B, S, D = x.shape
            flat_x = x.reshape(-1, D)
        else:
            flat_x = x
            D = flat_x.size(-1)

        r_logits = self.router(flat_x).float()
        evo_gates = F.softmax(r_logits, dim=-1)  # [BT, 34]
        if pull is not None:
            combined_pull = 0.5 * pull.float() + 0.5 * evo_gates
            combined_pull = combined_pull / combined_pull.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            combined_pull = evo_gates

        group_outputs = []
        for lo, hi, r, W1, W2, micro_profile in self._group_weights(flat_x.dtype):
            # Law 3a: Council parent projects query into its r micro-agents -> [BT, n, r]
            h_micro = torch.einsum("td,erd->ter", flat_x, W1)
            # Law 2 & Law 3b: Individualize each micro-agent and run nonlinear swarm debate
            h_ind = h_micro * micro_profile
            swarm_idea = h_ind + (0.25 * gov_scale) * torch.tanh(h_ind)
            # Law 4: Parent Council member incorporates returned micro-agent paths -> [BT, n, D]
            e_group = torch.einsum("ter,edr->ted", swarm_idea, W2)
            group_outputs.append(e_group)

        e_all = torch.cat(group_outputs, dim=1)  # [BT, 34, D]
        # Law 5: All 34 Council proposals report back to Throne pull synthesis
        moe_out = torch.einsum("te,ted->td", combined_pull.to(flat_x.dtype), e_all)
        e_pooled = e_all.mean(dim=0)  # [34, D]
        return moe_out, combined_pull, e_pooled

    def forward(self, x: torch.Tensor, pull: torch.Tensor = None, gov_scale: float = 1.0):
        is_3d = (x.dim() == 3)
        if is_3d:
            B, S, D = x.shape
        moe_out, _, _ = self.forward_council(x, pull=pull, gov_scale=gov_scale)
        return moe_out.reshape(B, S, D) if is_3d else moe_out
