#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sm61_linear.py — nn.Linear-compatible module backed by the sm_61 DP4A kernel.

  SM61Linear.from_linear(layer) converts a trained nn.Linear:
    weights quantized ONCE at load (~4x smaller: a 0.6B fp32 model fits
    a 4 GB card as int8); activations quantized per-row every forward.

  Routing, in order:
    1. grad needed  -> plain fp32 F.linear (correct backward; the DP4A
       kernel is forward-only and must never see autograd).
    2. no_grad + compiled extension available -> HeterogeneousDispatcher
       (GPU DP4A slice + CPU vGPU slice concurrently, split_ratio tuned
       against YOUR measured throughput).
    3. no_grad, no extension -> pure-CPU vgpu_qgemm_cpu reference.

Tune split_ratio against measured wall-clock on your box; there is no
universal right value.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize import quantize_per_output, quantize_per_row

try:
    import quillan_sm61_qgemm  # compiled extension (setup.py build_ext --inplace)
    from vgpu_backend import HeterogeneousDispatcher

    _EXT_OK = True
except Exception:
    quillan_sm61_qgemm = None  # type: ignore
    HeterogeneousDispatcher = None  # type: ignore
    _EXT_OK = False

try:
    from vgpu_backend import vgpu_qgemm_cpu

    _CPU_OK = True
except Exception:
    vgpu_qgemm_cpu = None  # type: ignore
    _CPU_OK = False


class SM61Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        split_ratio: float = 0.9,
        clip_percentile: Optional[float] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.split_ratio = split_ratio
        self.clip_percentile = clip_percentile
        # Pre-quantized weight ([N, K] int8) + per-output scales + fp32 bias.
        self.register_buffer("w_i8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("w_scale", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias_fp", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias_fp = None
        self._dispatcher = None
        if _EXT_OK:
            self._dispatcher = HeterogeneousDispatcher(
                cuda_op=quillan_sm61_qgemm.qgemm_forward,
                split_ratio=split_ratio,
            )

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        split_ratio: float = 0.9,
        clip_percentile: Optional[float] = None,
    ) -> "SM61Linear":
        mod = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            split_ratio=split_ratio,
            clip_percentile=clip_percentile,
        )
        with torch.no_grad():
            w_q, w_s = quantize_per_output(layer.weight.detach(), clip_percentile)
            mod.w_i8.copy_(w_q)
            mod.w_scale.copy_(w_s)
            if layer.bias is not None:
                mod.bias_fp.copy_(layer.bias.detach().float())
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gradient path: stay in fp32 so backward() is exact. Rebuild the
        # fp32 weight from the quantized copy (dequant is differentiable-free
        # but this branch only runs when grad is genuinely needed).
        if torch.is_grad_enabled() and (x.requires_grad or self.training):
            w_fp = self.w_i8.float() * self.w_scale.view(-1, 1)
            return F.linear(x.float(), w_fp, self.bias_fp)

        with torch.no_grad():
            orig_shape = x.shape
            if x.dim() > 2:
                x_2d = x.reshape(-1, x.shape[-1])
            else:
                x_2d = x
            x_q, x_s = quantize_per_row(x_2d.detach(), self.clip_percentile)
            if self._dispatcher is not None:
                y = self._dispatcher(x_q, self.w_i8, x_s, self.w_scale)
            elif _CPU_OK:
                y = vgpu_qgemm_cpu(
                    x_q.cpu(), self.w_i8.cpu(), x_s.cpu(), self.w_scale.cpu()
                )
            else:
                raise RuntimeError("No compute path: extension missing and vgpu_backend unavailable.")
            y = y.float()
            if self.bias_fp is not None:
                y = y + self.bias_fp
            if len(orig_shape) > 2:
                y = y.reshape(*orig_shape[:-1], self.out_features)
            return y.to(x.dtype) if x.dtype != torch.float32 else y

    def extra_repr(self) -> str:
        path = "dp4a+cpu" if self._dispatcher is not None else "cpu"
        return f"in={self.in_features}, out={self.out_features}, path={path}, split={self.split_ratio}"
