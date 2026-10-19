#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize.py — the missing quant helpers for the sm_61 DP4A path.

Symmetric per-row (activations) / per-output (weights) int8 quantization
matching exactly what qgemm_binding.cpp expects:
  x_int8 [M, K] int8, x_scale [M] fp32  (one scale per activation row)
  w_int8 [N, K] int8, w_scale [N] fp32  (one scale per output channel)

Optional percentile clipping tames outlier rows without changing the API.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _resolve_amax(vals: torch.Tensor, clip_percentile: Optional[float]) -> torch.Tensor:
    if clip_percentile is None:
        return vals
    lo = 100.0 - float(clip_percentile)
    hi = float(clip_percentile)
    # torch.quantile needs fp32+; keep it simple and explicit.
    lo_v = torch.quantile(vals.float(), lo / 100.0)
    hi_v = torch.quantile(vals.float(), hi / 100.0)
    bound = torch.maximum(lo_v.abs(), hi_v.abs())
    return torch.clamp(vals, -bound, bound)


@torch.no_grad()
def quantize_per_row(
    x: torch.Tensor,
    clip_percentile: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [M, K] fp -> (int8 [M, K], fp32 [M])."""
    xf = x.detach().float()
    xf = _resolve_amax(xf, clip_percentile)
    amax = xf.abs().amax(dim=-1).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(xf / scale.view(-1, 1)), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


@torch.no_grad()
def quantize_per_output(
    w: torch.Tensor,
    clip_percentile: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N, K] fp -> (int8 [N, K], fp32 [N]). One scale per output."""
    wf = w.detach().float()
    wf = _resolve_amax(wf, clip_percentile)
    amax = wf.abs().amax(dim=1).clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(wf / scale.view(-1, 1)), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


@torch.no_grad()
def quantize_roundtrip_error(x: torch.Tensor) -> float:
    """Max relative error of quantize->dequantize; sanity metric only."""
    q, s = quantize_per_row(x)
    ref = q.float() * s.view(-1, 1)
    err = (ref - x.float()).abs() / (x.float().abs() + 1e-8)
    return float(err.max().item())
