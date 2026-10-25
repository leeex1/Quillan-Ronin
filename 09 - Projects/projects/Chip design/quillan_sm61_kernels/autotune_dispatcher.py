"""
Autotuning dispatcher for sm_61 DP4A GEMM kernels.

This module provides:
1. Autotuning: benchmark different kernel configurations
2. Shape-specific dispatch: select best kernel based on (M, N, K)
3. Persistent cache: store results for fast lookup
4. Fallback: sensible defaults for unseen shapes
"""

import torch
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import time

# Kernel configuration database
KERNEL_CONFIGS = {
    "64_64_32_1_1": {"tm": 64, "tn": 64, "tk": 32, "bm": 1, "bn": 1, "name": "small"},
    "128_64_32_1_1": {"tm": 128, "tn": 64, "tk": 32, "bm": 1, "bn": 1, "name": "medium_baseline"},
    "128_64_32_2_2": {"tm": 128, "tn": 64, "tk": 32, "bm": 2, "bn": 2, "name": "medium_regblock"},
    "256_64_32_1_1": {"tm": 256, "tn": 64, "tk": 32, "bm": 1, "bn": 1, "name": "large"},
    "256_64_32_2_2": {"tm": 256, "tn": 64, "tk": 32, "bm": 2, "bn": 2, "name": "large_regblock"},
    "128_128_32_1_1": {"tm": 128, "tn": 128, "tk": 32, "bm": 1, "bn": 1, "name": "square"},
    "256_32_32_1_1": {"tm": 256, "tn": 32, "tk": 32, "bm": 1, "bn": 1, "name": "tall_skinny"},
    "64_256_32_1_1": {"tm": 64, "tn": 256, "tk": 32, "bm": 1, "bn": 1, "name": "wide_short"},
}

# Heuristic selection based on matrix shape
HEURISTIC_RULES = [
    # (condition, config_key)
    (lambda m, n, k: m <= 128 and n <= 128 and k <= 128, "64_64_32_1_1"),
    (lambda m, n, k: m > 256 and n < 64, "256_32_32_1_1"),  # Tall and skinny
    (lambda m, n, k: m < 64 and n > 256, "64_256_32_1_1"),  # Wide and short
    (lambda m, n, k: m >= 256 and n >= 128, "128_128_32_1_1"),  # Large square-ish
    (lambda m, n, k: m >= 256, "256_64_32_1_1"),  # Large M
    (lambda m, n, k: True, "128_64_32_1_1"),  # Default
]


class Autotuner:
    """Autotuning system for kernel configuration selection."""

    def __init__(self, cache_file: str = "autotune_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, str] = self._load_cache()
        self.baseline_kernel = None  # Will be loaded from compiled extension

    def _load_cache(self) -> Dict[str, str]:
        """Load autotuning cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save autotuning cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def _shape_key(self, M: int, N: int, K: int) -> str:
        """Generate a cache key for matrix shape."""
        # Quantize to nearby powers of 2 for cache reuse
        def quantize(x):
            if x <= 64: return 64
            if x <= 128: return 128
            if x <= 256: return 256
            if x <= 512: return 512
            if x <= 1024: return 1024
            if x <= 2048: return 2048
            return 4096

        return f"{quantize(M)}_{quantize(N)}_{quantize(K)}"

    def benchmark_config(
        self,
        config_key: str,
        M: int, N: int, K: int,
        x_int8: torch.Tensor,
        w_int8: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        num_warmup: int = 5,
        num_iters: int = 20
    ) -> float:
        """Benchmark a single kernel configuration."""
        import quillan_sm61_qgemm_autotune  # Load autotune extension

        # Get the kernel function
        kernel_func = getattr(quillan_sm61_qgemm_autotune, f"qgemm_launch_{config_key}")

        # Warmup
        for _ in range(num_warmup):
            y = kernel_func(x_int8, w_int8, x_scale, w_scale)
            torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            y = kernel_func(x_int8, w_int8, x_scale, w_scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return elapsed / num_iters

    def autotune_shape(
        self,
        M: int, N: int, K: int,
        num_warmup: int = 5,
        num_iters: int = 20
    ) -> str:
        """Autotune best kernel configuration for a given shape."""
        shape_key = self._shape_key(M, N, K)

        # Check cache
        if shape_key in self.cache:
            return self.cache[shape_key]

        # Generate test data
        x_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device='cuda')
        w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device='cuda')
        x_scale = torch.randn(M, dtype=torch.float32, device='cuda')
        w_scale = torch.randn(N, dtype=torch.float32, device='cuda')

        # Benchmark all configurations
        best_config = None
        best_time = float('inf')

        print(f"Autotuning shape ({M}, {N}, {K})...")
        for config_key, config in KERNEL_CONFIGS.items():
            try:
                elapsed = self.benchmark_config(
                    config_key, M, N, K,
                    x_int8, w_int8, x_scale, w_scale,
                    num_warmup, num_iters
                )
                print(f"  {config_key}: {elapsed*1000:.3f}ms")

                if elapsed < best_time:
                    best_time = elapsed
                    best_config = config_key
            except Exception as e:
                print(f"  {config_key}: FAILED ({e})")

        if best_config is None:
            # Fallback to heuristic
            best_config = self._select_heuristic(M, N, K)
            print(f"  All configs failed, using heuristic: {best_config}")

        # Cache result
        self.cache[shape_key] = best_config
        self._save_cache()

        print(f"  Best: {best_config} ({best_time*1000:.3f}ms)")
        return best_config

    def _select_heuristic(self, M: int, N: int, K: int) -> str:
        """Select kernel configuration using heuristic rules."""
        for condition, config_key in HEURISTIC_RULES:
            if condition(M, N, K):
                return config_key
        return "128_64_32_1_1"  # Default

    def get_config(self, M: int, N: int, K: int, use_cache: bool = True) -> str:
        """Get best configuration for a shape, with fallback to heuristic."""
        if use_cache:
            shape_key = self._shape_key(M, N, K)
            if shape_key in self.cache:
                return self.cache[shape_key]

        # Use heuristic as fallback
        return self._select_heuristic(M, N, K)


class DispatchedGEMM:
    """Dispatched GEMM with automatic kernel selection."""

    def __init__(self, cache_file: str = "autotune_cache.json"):
        self.autotuner = Autotuner(cache_file)
        self.use_autotune = True

    def __call__(
        self,
        x_int8: torch.Tensor,
        w_int8: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        force_config: Optional[str] = None
    ) -> torch.Tensor:
        """
        Dispatched GEMM call.

        Args:
            x_int8: [M, K] int8 tensor
            w_int8: [N, K] int8 tensor (pre-transposed)
            x_scale: [M] float32 tensor
            w_scale: [N] float32 tensor
            force_config: Optional override to force specific kernel config

        Returns:
            y: [M, N] fp16 tensor
        """
        M, K = x_int8.shape
        N, _ = w_int8.shape

        # Select configuration
        if force_config is not None:
            config_key = force_config
        elif self.use_autotune:
            config_key = self.autotuner.get_config(M, N, K)
        else:
            config_key = self.autotuner._select_heuristic(M, N, K)

        # Call the selected kernel
        import quillan_sm61_qgemm_autotune
        kernel_func = getattr(quillan_sm61_qgemm_autotune, f"qgemm_launch_{config_key}")
        return kernel_func(x_int8, w_int8, x_scale, w_scale)

    def autotune_current_shape(
        self,
        x_int8: torch.Tensor,
        w_int8: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        num_warmup: int = 5,
        num_iters: int = 20
    ) -> str:
        """Autotune for the current tensor shapes."""
        M, K = x_int8.shape
        N, _ = w_int8.shape
        return self.autotuner.autotune_shape(M, N, K, num_warmup, num_iters)

    def disable_autotune(self):
        """Disable autotuning, use heuristics only."""
        self.use_autotune = False

    def enable_autotune(self):
        """Enable autotuning with cache."""
        self.use_autotune = True


# Global dispatcher instance
_dispatcher = None


def get_dispatcher(cache_file: str = "autotune_cache.json") -> DispatchedGEMM:
    """Get or create the global dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = DispatchedGEMM(cache_file)
    return _dispatcher


def qgemm_dispatched(
    x_int8: torch.Tensor,
    w_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    force_config: Optional[str] = None
) -> torch.Tensor:
    """
    Dispatched GEMM with automatic kernel selection.

    This is the main entry point for users. It automatically selects
    the best kernel configuration based on matrix shape and cached
    autotuning results.

    Example:
        >>> y = qgemm_dispatched(x_i8, w_i8, x_scale, w_scale)
    """
    dispatcher = get_dispatcher()
    return dispatcher(x_int8, w_int8, x_scale, w_scale, force_config)
