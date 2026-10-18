#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — SOVEREIGN MUON-K2 + ADAMW + CCRL HYBRID OPTIMIZER
---------------------------------------------------------------------------------------
CPU-Hardened Implementation:
  1. Low-Rank Muon: Matrix Orthogonalization via Newton-Schulz for 2D LoRA/Swarm adapters (Rank <= 256).
  2. AdamW: Standard adaptive second-moment updates for larger weight matrices & embeddings.
  3. CCRL Layer: Curvature-Conditioned Recursive Learning with dynamic gradient clipping.
"""

import math
import torch
from typing import List, Dict, Any, Tuple, Optional
from torch.optim.optimizer import Optimizer

def zeropower_via_newtonschulz5_lowrank(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes approximate polar decomposition via 5th-order Newton-Schulz
    for matrices where min(dim) <= 256.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    orig_shape = G.shape
    if G.ndim > 2:
        G = G.reshape(-1, G.shape[-1])
        
    X = G.to(dtype=torch.float32)
    norm = torch.linalg.norm(X) + eps
    X = X / norm

    transposed = False
    if X.size(0) < X.size(1):
        X = X.T
        transposed = True

    # X is (N, K) where K <= N
    # Compute Gram matrix A = X.T @ X of size (K, K)
    K = X.size(1)
    if K > 512:
        # Fallback to normalized gradient if inner dimension is too large for CPU RAM
        return (G / (norm + 1e-7)).reshape(orig_shape)

    for _ in range(steps):
        A = X.T @ X
        B = b * A + c * (A @ A)
        X = X @ (a * torch.eye(K, device=X.device, dtype=X.dtype) + B)

    if transposed:
        X = X.T

    return X.reshape(orig_shape).to(dtype=G.dtype)

class SovereignMuonK2AdamW(Optimizer):
    def __init__(
        self,
        muon_params: List[torch.nn.Parameter],
        adamw_params: List[torch.nn.Parameter],
        lr_muon: float = 0.02,
        lr_adamw: float = 3e-4,
        momentum_muon: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        ccrl_limit: float = 5.0,
        eps: float = 1e-8,
        ns_steps: int = 5
    ):
        defaults = dict(
            lr_muon=lr_muon,
            lr_adamw=lr_adamw,
            momentum_muon=momentum_muon,
            betas=betas,
            weight_decay=weight_decay,
            ccrl_limit=ccrl_limit,
            eps=eps,
            ns_steps=ns_steps
        )
        
        param_groups = []
        if muon_params:
            param_groups.append({"params": muon_params, "is_muon": True, "lr": lr_muon})
        if adamw_params:
            param_groups.append({"params": adamw_params, "is_muon": False, "lr": lr_adamw})
            
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            is_muon = group.get("is_muon", False)
            lr = group.get("lr", 1e-3)
            wd = group.get("weight_decay", 0.0)
            
            if is_muon:
                momentum = group["momentum_muon"]
                ns_steps = group["ns_steps"]
                ccrl_limit = group["ccrl_limit"]

                for p in group["params"]:
                    if p.grad is None: continue
                    grad = p.grad

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p)

                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    
                    # CCRL Curvature Regularization
                    norm_g = torch.linalg.norm(buf)
                    if norm_g > ccrl_limit:
                        buf.mul_(ccrl_limit / (norm_g + 1e-8))

                    # Newton-Schulz orthogonalization
                    update = zeropower_via_newtonschulz5_lowrank(buf, steps=ns_steps)
                    
                    if wd > 0:
                        p.mul_(1.0 - lr * wd)
                        
                    p.add_(update, alpha=-lr)

            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None: continue
                    grad = p.grad

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]

                    # Zero-allocation in-place denominator computation
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                    step_size = lr / bias_correction1

                    if wd > 0:
                        p.mul_(1.0 - lr * wd)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def create_quillan_muonk2_optimizer(
    model: torch.nn.Module,
    lr_muon: float = 0.02,
    lr_adamw: float = 3e-4,
    weight_decay: float = 0.01,
    ccrl_limit: float = 5.0
) -> SovereignMuonK2AdamW:
    """Partitions parameters into Low-Rank Muon (LoRA/Swarms) and AdamW (Bridges/Norms/Dense)."""
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Apply Muon to 2D LoRA matrices and Swarms where min dimension <= 256
        if p.ndim == 2 and min(p.shape) <= 256 and any(k in name for k in ['lora', 'swarm', 'expert_swarms']):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return SovereignMuonK2AdamW(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr_muon=lr_muon,
        lr_adamw=lr_adamw,
        weight_decay=weight_decay,
        ccrl_limit=ccrl_limit
    )
