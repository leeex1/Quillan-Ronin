"""
Run autotuning for common matrix shapes.

This script benchmarks all kernel configurations on a set of representative
matrix shapes and stores the results in autotune_cache.json for fast dispatch.

Usage:
    python run_autotune.py
"""

import torch
from autotune_dispatcher import Autotuner, get_dispatcher

# Common matrix shapes to autotune
TEST_SHAPES = [
    # Small matrices
    (64, 64, 64),
    (128, 64, 64),
    (64, 128, 64),

    # Medium matrices
    (256, 128, 128),
    (512, 256, 256),
    (256, 512, 256),

    # Large matrices
    (1024, 512, 512),
    (2048, 1024, 1024),

    # Tall and skinny
    (2048, 64, 128),
    (4096, 128, 256),

    # Wide and short
    (64, 2048, 128),
    (128, 4096, 256),

    # Square
    (512, 512, 512),
    (1024, 1024, 1024),
]


def main():
    print("=" * 60)
    print("Quillan sm_61 GEMM Autotuning")
    print("=" * 60)
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print()

    autotuner = Autotuner("autotune_cache.json")

    for M, N, K in TEST_SHAPES:
        print(f"\n{'='*60}")
        print(f"Shape: M={M}, N={N}, K={K}")
        print(f"{'='*60}")

        try:
            best_config = autotuner.autotune_shape(M, N, K, num_warmup=3, num_iters=10)
            print(f"✓ Best config: {best_config}")
        except Exception as e:
            print(f"✗ Failed: {e}")
            # Use heuristic as fallback
            heuristic = autotuner._select_heuristic(M, N, K)
            autotuner.cache[autotuner._shape_key(M, N, K)] = heuristic
            autotuner._save_cache()
            print(f"✓ Using heuristic: {heuristic}")

    print("\n" + "=" * 60)
    print("Autotuning complete!")
    print(f"Cache saved to: autotune_cache.json")
    print("=" * 60)

    # Print summary
    print("\nCache summary:")
    for shape_key, config in sorted(autotuner.cache.items()):
        print(f"  {shape_key}: {config}")


if __name__ == "__main__":
    main()
