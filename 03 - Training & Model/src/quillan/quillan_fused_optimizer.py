#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuillanFusedOptimizer (v5.4.0-oni — 2026 Sovereign Release)
==========================================================
Unified Sovereign Muon-K2 + Sophia-H + AdamW + CCRL Hybrid Optimizer.
Inherits from `torch.optim.Optimizer` for 100% native PyTorch compatibility,
scheduler integration, DDP readiness, and robust checkpoint serialization.

Core Optimization Pipeline:
  1. MUON (Newton-Schulz Orthogonalization):
     Applied to 2D and 3D weight matrices (Linear, Conv, MoE expert weights [B, M, N], Swarms).
     Prevents gradient collapse along low-variance directions and accelerates convergence.
  2. SOPHIA-H (Second-Order Curvature Damping):
     Diagonal Hessian EMA curvature estimation (Gauss-Newton approximation) clipping update steps.
  3. CCRL (Curvature-Conditioned Recursive Learning):
     Gradient Centralization (zero-mean non-output dimensions) + gradient norm bounding.
  4. ADAMW (Adaptive Second-Moment Updates):
     Applied to 1D parameters (biases, RMSNorm/LayerNorm scales) and embedding tables.
  5. BITNET DEAD-ZONE:
     Enforces post-step ternary weight sparsity by zeroing values within [-threshold, +threshold].
  6. PID THERMO GOVERNOR:
     Closed-loop learning rate damping based on system loss / health telemetry.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

LOG = logging.getLogger(__name__)


# ─── Gradient Utilities ───────────────────────────────────────────────────────

def centralize_gradient(g: torch.Tensor) -> torch.Tensor:
    """Zero-mean gradient over all non-output dimensions (Gradient Centralization)."""
    if g.dim() < 2:
        return g
    reduce_dims = list(range(1, g.dim()))
    return g - g.mean(dim=reduce_dims, keepdim=True)


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate polar decomposition via 5th-order / 3rd-order Newton-Schulz iteration.

    Supports 2D [M, N] and batched 3D [B, M, N] (MoE expert tensors).
    Ensures min(dim) <= max(dim) before iterating (transposes if needed, restores after).
    Coefficients: a=3.4445, b=-4.7750, c=2.0315 (5th order) or a=1.5, b=-0.5 (3rd order).
    """
    if G.numel() == 0:
        return G
    g_norm = G.norm()
    if g_norm.item() == 0.0 or torch.isnan(g_norm) or torch.isinf(g_norm):
        return G

    orig_dtype = G.dtype
    orig_shape = G.shape
    is_3d = G.dim() == 3

    X = G.to(dtype=torch.float32)
    norm = X.norm() + eps
    X = X / norm

    transposed = False
    if is_3d:
        if X.shape[1] > X.shape[2]:
            X = X.mT
            transposed = True
        # Batched 3D: Gram matrix A = X @ X.mT
        a, b = 1.5, -0.5
        for _ in range(min(steps, 4)):
            A = X @ X.mT
            X = a * X + b * (A @ X)
        if transposed:
            X = X.mT
    else:
        if X.size(0) > X.size(1):
            X = X.T
            transposed = True

        K = X.size(0)
        if K <= 256 and steps >= 4:
            # 5th-order iteration on small inner dimension
            a, b, c = 3.4445, -4.7750, 2.0315
            eye_k = torch.eye(K, device=X.device, dtype=torch.float32)
            for _ in range(steps):
                A = X @ X.T
                B = b * A + c * (A @ A)
                X = (a * eye_k + B) @ X
        else:
            # 3rd-order iteration: O(N * K^2)
            a, b = 1.5, -0.5
            for _ in range(min(steps, 4)):
                A = X @ X.T
                X = a * X + b * (A @ X)

        if transposed:
            X = X.T

    return X.reshape(orig_shape).to(dtype=orig_dtype)


# ─── PID Thermo Governor & Noise Scale EMA ────────────────────────────────────

class PIDGovernor:
    """PID-based LR damping. integrity=1.0 -> no damping; integrity=0 -> full stop."""

    def __init__(self, kp: float = 0.1, ki: float = 0.01, kd: float = 0.001) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def step(self, integrity: float = 1.0) -> float:
        error = 1.0 - max(0.0, min(1.0, integrity))
        self._integral += error
        d_term = error - self._prev_error
        self._prev_error = error
        scale = 1.0 - (self.kp * error + self.ki * self._integral + self.kd * d_term)
        return max(0.0, min(1.0, scale))

    def state_dict(self) -> Dict[str, float]:
        return {"integral": self._integral, "prev_error": self._prev_error}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self._integral = state.get("integral", 0.0)
        self._prev_error = state.get("prev_error", 0.0)


class NoiseScaleEMA:
    """Tracks gradient noise via exponential moving average of gradient norms."""

    def __init__(self, decay: float = 0.99) -> None:
        self._ema: float = 1.0
        self._decay = decay

    def step(self, grads: Iterable[Optional[torch.Tensor]]) -> float:
        norms = [g.norm().item() ** 2 for g in grads if g is not None]
        if norms:
            current = float(sum(norms) ** 0.5)
            self._ema = self._decay * self._ema + (1.0 - self._decay) * current
        return self._ema

    def state_dict(self) -> Dict[str, float]:
        return {"ema": self._ema}

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self._ema = state.get("ema", 1.0)


# ─── Main Sovereign Optimizer ────────────────────────────────────────────────

class QuillanFusedOptimizer(Optimizer):
    """Sovereign unified optimizer combining Muon-K2, Sophia-H, CCRL, and AdamW.

    Fully compatible with PyTorch's native `Optimizer` API, learning rate schedulers,
    gradient clipping, and `torch.save`/`torch.load` state dictionaries.
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], List[Dict[str, Any]]],
        lr: float = 3e-4,
        lr_muon: float = 0.02,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        momentum_muon: float = 0.95,
        ns_steps: int = 5,
        ccrl_limit: float = 5.0,
        gradient_centralization: bool = True,
        use_sophia: bool = True,
        sophia_rho: float = 0.05,
        sophia_beta: float = 0.999,
        bitnet_deadzone: float = 0.0,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.001,
        noise_decay: float = 0.99,
    ) -> None:
        defaults = dict(
            lr=lr,
            lr_muon=lr_muon,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            momentum_muon=momentum_muon,
            ns_steps=ns_steps,
            ccrl_limit=ccrl_limit,
            gradient_centralization=gradient_centralization,
            use_sophia=use_sophia,
            sophia_rho=sophia_rho,
            sophia_beta=sophia_beta,
            bitnet_deadzone=bitnet_deadzone,
        )

        # Handle raw parameter lists by classifying if not already grouped
        param_list = list(params)
        if param_list and not isinstance(param_list[0], dict):
            muon_group = []
            adamw_group = []
            for p in param_list:
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:
                    muon_group.append(p)
                else:
                    adamw_group.append(p)
            param_groups = []
            if muon_group:
                param_groups.append({"params": muon_group, "is_muon": True, "lr": lr_muon})
            if adamw_group:
                param_groups.append({"params": adamw_group, "is_muon": False, "lr": lr})
        else:
            param_groups = param_list

        super().__init__(param_groups, defaults)

        self._pid = PIDGovernor(pid_kp, pid_ki, pid_kd)
        self._noise = NoiseScaleEMA(noise_decay)
        self._global_step: int = 0

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        integrity: float = 1.0,
    ) -> Optional[float]:
        """Perform a single optimization step with CCRL and PID thermal governing."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        pid_scale = self._pid.step(integrity)
        self._global_step += 1

        all_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_grads.append(p.grad)
        self._noise.step(all_grads)

        for group in self.param_groups:
            is_muon = group.get("is_muon", False)
            base_lr = group.get("lr", 1e-4)
            effective_lr = base_lr * pid_scale
            wd = group.get("weight_decay", 0.0)
            eps = group.get("eps", 1e-8)

            if is_muon:
                momentum = group.get("momentum_muon", 0.95)
                ns_steps = group.get("ns_steps", 5)
                ccrl_limit = group.get("ccrl_limit", 5.0)
                gc = group.get("gradient_centralization", True)
                use_sophia = group.get("use_sophia", True)
                sophia_rho = group.get("sophia_rho", 0.05)
                sophia_beta = group.get("sophia_beta", 0.999)
                deadzone = group.get("bitnet_deadzone", 0.0)

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

                    # 1. Gradient Centralization
                    if gc and g.dim() >= 2:
                        g = centralize_gradient(g)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p)
                        if use_sophia:
                            state["hessian_ema"] = torch.zeros_like(p)

                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    # 2. CCRL Norm Bounding
                    norm_buf = buf.norm()
                    if norm_buf > ccrl_limit:
                        buf.mul_(ccrl_limit / (norm_buf + eps))

                    # 3. Newton-Schulz Orthogonalization
                    update = newton_schulz(buf, steps=ns_steps, eps=eps)

                    # 4. Sophia-H Curvature Damping
                    if use_sophia and "hessian_ema" in state:
                        hess = state["hessian_ema"]
                        # Gauss-Newton diagonal estimate: g^2
                        hess.mul_(sophia_beta).addcmul_(g, g, value=1.0 - sophia_beta)
                        denom = hess.clamp(min=sophia_rho / 100.0)
                        update = update / denom
                        update = update.clamp(-sophia_rho, sophia_rho)

                    # 5. Decoupled Weight Decay
                    if wd > 0.0:
                        p.mul_(1.0 - effective_lr * wd)

                    # 6. Apply Orthogonal Step
                    p.add_(update.to(dtype=p.dtype), alpha=-effective_lr)

                    # 7. BitNet Dead-Zone Sparsity
                    if deadzone > 0.0:
                        p[p.abs() < deadzone] = 0.0

            else:
                # AdamW for scalars, norms, and embeddings
                beta1, beta2 = group.get("betas", (0.9, 0.95))

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    step_t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bias_correction1 = 1.0 - (beta1 ** step_t)
                    bias_correction2 = 1.0 - (beta2 ** step_t)

                    step_size = effective_lr / bias_correction1
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                    if wd > 0.0:
                        p.mul_(1.0 - effective_lr * wd)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd["pid"] = self._pid.state_dict()
        sd["noise_ema"] = self._noise.state_dict()
        sd["global_step"] = self._global_step
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "pid" in state_dict:
            self._pid.load_state_dict(state_dict["pid"])
        if "noise_ema" in state_dict:
            self._noise.load_state_dict(state_dict["noise_ema"])
        if "global_step" in state_dict:
            self._global_step = state_dict["global_step"]
        super().load_state_dict(state_dict)


def create_quillan_fused_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    lr_muon: float = 0.02,
    weight_decay: float = 0.01,
    ccrl_limit: float = 5.0,
    ns_steps: int = 5,
    use_sophia: bool = True,
    bitnet_deadzone: float = 0.0,
) -> QuillanFusedOptimizer:
    """Partitions model parameters into Muon (matrices >= 2D) and AdamW (1D/scalars)."""
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Apply Muon strictly to internal representation matrices (attention projections, MoE expert MLPs, LoRA).
        # Standard Muon specification strictly excludes embeddings, the lm_head/unembedding projection, and router gates.
        is_excluded = any(k in name.lower() for k in ["embed", "wte", "wpe", "embedding", "lm_head", "head", "router", "gate"])
        if p.ndim >= 2 and not is_excluded:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    param_groups = [
        {"params": muon_params, "is_muon": True, "lr": lr_muon, "weight_decay": weight_decay},
        {"params": adamw_params, "is_muon": False, "lr": lr, "weight_decay": weight_decay},
    ]

    return QuillanFusedOptimizer(
        params=param_groups,
        lr=lr,
        lr_muon=lr_muon,
        weight_decay=weight_decay,
        ccrl_limit=ccrl_limit,
        ns_steps=ns_steps,
        use_sophia=use_sophia,
        bitnet_deadzone=bitnet_deadzone,
    )


# Backward-compatible canonical aliases
SovereignMuonK2AdamW = QuillanFusedOptimizer
create_quillan_muonk2_optimizer = create_quillan_fused_optimizer
