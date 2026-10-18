#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin Hardware-Tiered Kernel Subsystem
=============================================
Unified export interface for CPU SIMD, GPU Inductor, and CUDA acceleration kernels.
"""

from .fused_ste import (
    FusedSTEActivationFunction,
    fused_ste_activation,
)
from .ternary_gemm import (
    pack_ternary_weights,
    unpack_ternary_weights,
    HardwareTernaryLinear,
)
from .moe_dispatch import (
    fixed_capacity_moe_dispatch,
    scatter_add_moe_output,
    mark_dynamic_dims,
)

__all__ = [
    "FusedSTEActivationFunction",
    "fused_ste_activation",
    "pack_ternary_weights",
    "unpack_ternary_weights",
    "HardwareTernaryLinear",
    "fixed_capacity_moe_dispatch",
    "scatter_add_moe_output",
    "mark_dynamic_dims",
]
