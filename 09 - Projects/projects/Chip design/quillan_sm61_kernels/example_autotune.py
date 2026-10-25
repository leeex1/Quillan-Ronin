"""
Example usage of the autotuning GEMM dispatcher.

This demonstrates:
1. Basic usage with automatic dispatch
2. Manual kernel selection
3. Running autotuning for specific shapes
4. Comparing performance across configurations
"""

import torch
from autotune_dispatcher import get_dispatcher, qgemm_dispatched

# Helper function for quantization
def quantize_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row int8 quantization."""
    max_abs = x.abs().amax(dim=1, keepdim=True)
    scale = max_abs / 127.0
    x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale.squeeze(0)

def quantize_per_col(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-column int8 quantization."""
    max_abs = w.abs().amax(dim=0, keepdim=True)
    scale = max_abs / 127.0
    w_q = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return w_q, scale.squeeze(0)


def example_basic_usage():
    """Basic usage with automatic dispatch."""
    print("\n" + "="*60)
    print("Example 1: Basic usage with automatic dispatch")
    print("="*60)

    # Create test data
    M, N, K = 512, 256, 128
    x = torch.randn(M, K, dtype=torch.float16, device='cuda')
    w = torch.randn(N, K, dtype=torch.float16, device='cuda')

    # Quantize
    x_i8, x_scale = quantize_per_row(x)
    w_i8, w_scale = quantize_per_col(w)

    # Run with automatic dispatch
    y = qgemm_dispatched(x_i8, w_i8, x_scale, w_scale)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output dtype: {y.dtype}")
    print("✓ Automatic dispatch successful")


def example_manual_selection():
    """Manual kernel selection for comparison."""
    print("\n" + "="*60)
    print("Example 2: Manual kernel selection")
    print("="*60)

    # Create test data
    M, N, K = 256, 128, 64
    x = torch.randn(M, K, dtype=torch.float16, device='cuda')
    w = torch.randn(N, K, dtype=torch.float16, device='cuda')

    # Quantize
    x_i8, x_scale = quantize_per_row(x)
    w_i8, w_scale = quantize_per_col(w)

    dispatcher = get_dispatcher()

    # Compare different configurations
    configs = ["128_64_32_1_1", "128_64_32_2_2", "256_64_32_1_1"]

    for config in configs:
        y = dispatcher(x_i8, w_i8, x_scale, w_scale, force_config=config)
        torch.cuda.synchronize()
        print(f"  {config}: shape={y.shape}")

    print("✓ Manual selection successful")


def example_autotune_shape():
    """Run autotuning for a specific shape."""
    print("\n" + "="*60)
    print("Example 3: Autotune for specific shape")
    print("="*60)

    # Create test data
    M, N, K = 1024, 512, 256
    x = torch.randn(M, K, dtype=torch.float16, device='cuda')
    w = torch.randn(N, K, dtype=torch.float16, device='cuda')

    # Quantize
    x_i8, x_scale = quantize_per_row(x)
    w_i8, w_scale = quantize_per_col(w)

    dispatcher = get_dispatcher()

    # Run autotuning
    print("Running autotuning (this may take a minute)...")
    best_config = dispatcher.autotune_current_shape(
        x_i8, w_i8, x_scale, w_scale,
        num_warmup=3,
        num_iters=10
    )

    print(f"✓ Best config: {best_config}")


def example_performance_comparison():
    """Compare performance across different shapes."""
    print("\n" + "="*60)
    print("Example 4: Performance comparison")
    print("="*60)

    import time

    shapes = [
        (128, 64, 64),
        (256, 128, 128),
        (512, 256, 256),
        (1024, 512, 512),
    ]

    dispatcher = get_dispatcher()
    dispatcher.disable_autotune()  # Use heuristics for fair comparison

    for M, N, K in shapes:
        x = torch.randn(M, K, dtype=torch.float16, device='cuda')
        w = torch.randn(N, K, dtype=torch.float16, device='cuda')

        x_i8, x_scale = quantize_per_row(x)
        w_i8, w_scale = quantize_per_col(w)

        # Warmup
        for _ in range(5):
            y = dispatcher(x_i8, w_i8, x_scale, w_scale)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(20):
            y = dispatcher(x_i8, w_i8, x_scale, w_scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"  {M:4d}x{N:4d}x{K:4d}: {elapsed/20*1000:.3f}ms")

    print("✓ Performance comparison complete")


def main():
    print("="*60)
    print("Quillan sm_61 Autotuning GEMM Examples")
    print("="*60)

    try:
        example_basic_usage()
        example_manual_selection()
        # example_autotune_shape()  # Uncomment to run autotuning
        example_performance_comparison()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
