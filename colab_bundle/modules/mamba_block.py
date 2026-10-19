#!/usr/bin/env python3
# Mamba (2312.00752) — Linear-time selective SSM for Quillan world model long-horizon
import torch, torch.nn as nn

class MambaBlock(nn.Module):
    """Selective SSM per Mamba Alg. 2: O(n) instead of O(n^2) attention"""
    def __init__(self, hidden_dim=2048, d_state=16, expand=2):
        super().__init__()
        self.hidden_dim, self.d_state = hidden_dim, d_state
        self.in_proj = nn.Linear(hidden_dim, hidden_dim*expand, bias=False)
        self.conv1d = nn.Conv1d(hidden_dim*expand, hidden_dim*expand, kernel_size=4, groups=hidden_dim*expand, padding=3)
        # Selective params: input-dependent B,C,Δ
        self.x_proj = nn.Linear(hidden_dim*expand, d_state*2, bias=False)
        self.dt_proj = nn.Linear(d_state, hidden_dim*expand, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_state, hidden_dim*expand))
        self.D = nn.Parameter(torch.ones(hidden_dim*expand))
        self.out_proj = nn.Linear(hidden_dim*expand, hidden_dim, bias=False)
    def forward(self, x):
        # x: [B,S,D]
        xz = self.in_proj(x)  # [B,S,2D]
        x_conv = self.conv1d(xz.transpose(1,2))[:,:,:xz.shape[1]].transpose(1,2)
        x_conv = torch.nn.functional.silu(x_conv)
        # Selective scan (simplified): state space recurrence with input-dependent A,B,C
        # Full Mamba uses parallel associative scan; here O(n) loop for clarity
        B, S, D = x_conv.shape
        state = torch.zeros(B, self.d_state, D, device=x.device, dtype=x.dtype)
        A = -torch.exp(self.A_log)  # [d_state, D]
        outs = []
        for t in range(S):
            Bc, Cc = self.x_proj(x_conv[:,t]).chunk(2, dim=-1)  # [B,d_state]
            dt = torch.nn.functional.softplus(self.dt_proj(Bc))  # [B,D]
            dA = torch.exp(dt.unsqueeze(1) * A)  # [B,d_state,D]
            dB = Bc.unsqueeze(-1) * dt.unsqueeze(1)  # [B,d_state,D]
            state = state * dA + dB * x_conv[:,t].unsqueeze(1)
            y = (state * Cc.unsqueeze(-1)).sum(dim=1) + self.D * x_conv[:,t]
            outs.append(y)
        y = torch.stack(outs, dim=1)
        return self.out_proj(y)
