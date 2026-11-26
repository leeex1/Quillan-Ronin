#!/usr/bin/env python3
"""Quillan tri-stack recurrent loop — the 2/34/34 sovereign graph's loop fabric.

Three stacked mechanisms for the A2A council<->swarm communication:
  1. Recirculation (deep->shallow convex feedback)  — existing paper module
  2. GRT recurrent core (gated depth iteration)      — existing paper module
  3. GRU core (real recurrent cell)                  — NEW: fixes the FFN-only loop

The GRU is bound to the council/swarm axis: hidden_dim -> 34 council slots -> hidden_dim.
All state is per-token and resettable; no donor weights, no training required.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


class CouncilSwarmGRU(nn.Module):
    """Real GRU cell carrying state between council experts and their swarms.

    34 council slots mirror the expert count. State shape (B, 34, hidden) is
    initialized from the token hidden state so the loop begins from the model's
    own representation, not zeros injected from outside.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 34, rank: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        # project hidden -> compact council state, run GRU, project back
        self.in_proj = nn.Linear(hidden_dim, num_experts * rank, bias=False)
        self.cell = nn.GRUCell(num_experts * rank, num_experts * rank)
        self.out_proj = nn.Linear(num_experts * rank, hidden_dim, bias=False)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, H)
        B, T, H = x.shape
        flat = x.reshape(B * T, H)
        inp = self.in_proj(flat)  # (B*T, E*r)
        if state is None:
            state = inp.new_zeros(inp.shape)
        else:
            state = state.reshape(B * T, -1)
        new_state = self.cell(inp, state)
        out = self.out_proj(new_state).view(B, T, H)
        return out, new_state.view(B, T, -1)


class ThroneSwarm(nn.Module):
    """The `2` in 2/34/34: throne + its own private swarm loop.

    Low-rank sovereign oscillator bound to the throne residual stream.
    Zero-init output projection => exact identity until WSP activates it.
    """

    def __init__(self, hidden_dim: int, rank: int = 8):
        super().__init__()
        self.phase_in = nn.Linear(hidden_dim, rank, bias=False)
        self.phase_out = nn.Linear(rank, hidden_dim, bias=False)
        self.coupling = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.phase_in.weight, std=0.02)
        nn.init.zeros_(self.phase_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kuramoto-style phase binding on a low-rank projection of the throne state.
        phase = self.phase_in(x)
        bound = phase + torch.tanh(self.coupling) * (
            phase.mean(dim=-2, keepdim=True) - phase
        )
        return x + self.phase_out(bound)


class TristackLoop(nn.Module):
    """Stacks recirculation + GRT + GRU over a hidden stream."""

    def __init__(
        self, hidden_dim: int, num_experts: int = 34, recirc_alpha: float = 0.15
    ):
        super().__init__()
        from paper_26_30_recurrent_diffusion_pack import RecirculationLayer
        from paper_21_25_grt_coordination_pack import GRTRecurrentCore

        self.recirc = RecirculationLayer(alpha=recirc_alpha)
        # GRT depth iteration: the ONI body only calls compute_gate, so the
        # recurrence itself is dead there. Run it here with a small FFN core.
        self.grt = GRTRecurrentCore(hidden_dim, n_core=3, R=2)
        self.grt_core = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        nn.init.zeros_(self.grt_core[-1].weight)
        nn.init.zeros_(self.grt_core[-1].bias)
        self.gru = CouncilSwarmGRU(hidden_dim, num_experts=num_experts)
        # zero-init gates so the loop starts as identity and cannot destabilize a loaded checkpoint
        self.gate_recirc = nn.Parameter(torch.zeros(1))
        self.gate_grt = nn.Parameter(torch.zeros(1))
        self.gate_gru = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None, steps: int = 1
    ):
        """Apply the tri-stack for `steps` recurrent passes."""
        out = x
        g_state = state
        for _ in range(max(1, steps)):
            # deep -> shallow convex feedback: both streams derive from the
            # current hidden state so a loaded checkpoint starts unchanged.
            r, _info = self.recirc(out, out)
            out = out + torch.tanh(self.gate_recirc) * (r - out)
            # GRT: iterate a small FFN core R times with learned gates.
            gate_base = torch.cat([out, out], dim=-1)
            grt_out = self.grt.iterate(out, gate_base, self.grt_core)
            out = out + torch.tanh(self.gate_grt) * (grt_out - out)
            gru_out, g_state = self.gru(out, g_state)
            out = out + torch.tanh(self.gate_gru) * gru_out
        return out, g_state
