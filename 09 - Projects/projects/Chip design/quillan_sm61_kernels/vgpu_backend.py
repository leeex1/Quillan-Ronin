"""
vgpu_backend.py

There's no such thing as a free-floating "virtual GPU" without either (a)
NVIDIA vGPU/MIG hardware-level partitioning of a real GPU, or (b) a software
compute path on a different device (CPU here) that you offload work to. This
module implements (b): a numerically-matching CPU reference implementation
of the same quantized GEMM op, plus a dispatcher that splits a batch across
the real CUDA kernel and this CPU path, running both concurrently.

This is the honest version of "offload compute to a second unit" when you
only have one physical GPU: overlap GPU compute with CPU compute on
different slices of the same batch, instead of leaving the CPU idle while
the GPU works.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import torch


def vgpu_qgemm_cpu(
    x_int8: torch.Tensor,
    w_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """CPU reference implementation matching the CUDA kernel's semantics
    exactly: int32 accumulation of the int8 dot products, then per-row /
    per-column dequantization to fp16. All tensors must be CPU tensors.

    Y[m, n] = ( sum_k X[m,k] * W[n,k] ) * x_scale[m] * w_scale[n]
    """
    if x_int8.is_cuda or w_int8.is_cuda:
        raise ValueError("vgpu_qgemm_cpu expects CPU tensors; move data with .cpu() first")

    x = x_int8.to(torch.int32).numpy()
    w = w_int8.to(torch.int32).numpy()
    acc = x @ w.T  # [M, K] @ [K, N] -> [M, N] int32, matches dp4a accumulation
    xs = x_scale.numpy().reshape(-1, 1)
    ws = w_scale.numpy().reshape(1, -1)
    y = acc.astype(np.float32) * xs * ws
    return torch.from_numpy(y).to(torch.float16)


class HeterogeneousDispatcher:
    """Splits a batch's rows between the real CUDA kernel and the CPU vGPU
    path, runs both concurrently on separate threads, and recombines the
    results in the original row order.

    Parameters
    ----------
    cuda_op : callable(x_i8_cuda, w_i8_cuda, x_scale_cuda, w_scale_cuda) -> Tensor (CUDA, fp16)
        Typically `quillan_sm61_qgemm.qgemm_forward` from the compiled extension.
    split_ratio : float in [0, 1]
        Fraction of the batch's rows sent to the CUDA path; the remainder
        goes to the CPU vGPU path. There's no universal "correct" value --
        it depends on your GPU's throughput relative to your CPU's, and on
        how much VRAM headroom you have. Measure both paths' standalone
        throughput on your hardware and set this to balance completion time,
        not to some fixed ratio.
    """

    def __init__(self, cuda_op: Callable, split_ratio: float = 0.8):
        if not (0.0 <= split_ratio <= 1.0):
            raise ValueError("split_ratio must be in [0, 1]")
        self.cuda_op = cuda_op
        self.split_ratio = split_ratio

    def __call__(
        self,
        x_int8: torch.Tensor,
        w_int8: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
    ) -> torch.Tensor:
        M = x_int8.shape[0]
        split = int(round(M * self.split_ratio))

        if split <= 0:
            return vgpu_qgemm_cpu(x_int8.cpu(), w_int8.cpu(), x_scale.cpu(), w_scale.cpu())
        if split >= M:
            return self.cuda_op(
                x_int8.cuda(non_blocking=True),
                w_int8.cuda(non_blocking=True),
                x_scale.cuda(non_blocking=True),
                w_scale.cuda(non_blocking=True),
            ).cpu()

        results: list[torch.Tensor | None] = [None, None]
        errors: list[Exception | None] = [None, None]

        def run_cuda():
            try:
                xa = x_int8[:split].cuda(non_blocking=True)
                xsa = x_scale[:split].cuda(non_blocking=True)
                wa = w_int8.cuda(non_blocking=True)
                wsa = w_scale.cuda(non_blocking=True)
                results[0] = self.cuda_op(xa, wa, xsa, wsa).cpu()
            except Exception as e:  # surface on join, don't swallow silently
                errors[0] = e

        def run_cpu():
            try:
                results[1] = vgpu_qgemm_cpu(
                    x_int8[split:].cpu(), w_int8.cpu(),
                    x_scale[split:].cpu(), w_scale.cpu(),
                )
            except Exception as e:
                errors[1] = e

        t_cuda = threading.Thread(target=run_cuda)
        t_cpu = threading.Thread(target=run_cpu)
        t_cuda.start()
        t_cpu.start()
        t_cuda.join()
        t_cpu.join()

        for e in errors:
            if e is not None:
                raise e

        return torch.cat([results[0], results[1]], dim=0)


if __name__ == "__main__":
    # Self-contained correctness check of the CPU path against a plain
    # PyTorch int32 matmul reference (does not require a GPU to run).
    torch.manual_seed(0)
    M, K, N = 17, 64, 9

    x_i8 = torch.randint(-127, 128, (M, K), dtype=torch.int8)
    w_i8 = torch.randint(-127, 128, (N, K), dtype=torch.int8)
    x_scale = torch.rand(M, dtype=torch.float32) * 0.05 + 0.001
    w_scale = torch.rand(N, dtype=torch.float32) * 0.05 + 0.001

    y_cpu = vgpu_qgemm_cpu(x_i8, w_i8, x_scale, w_scale)

    ref = (x_i8.to(torch.int32) @ w_i8.to(torch.int32).T).to(torch.float32)
    ref = ref * x_scale.view(-1, 1) * w_scale.view(1, -1)
    ref = ref.to(torch.float16)

    abs_err = (y_cpu.float() - ref.float()).abs()
    rel_err = (abs_err / (ref.float().abs() + 1e-8)).max().item()
    print(f"[vgpu_backend self-test] M={M} K={K} N={N}  max_abs_err={abs_err.max().item():.6f}"
          f"  max_rel_err={rel_err:.6f}")
    # Use relative error, not absolute: at larger K the int32 accumulator
    # magnitude grows, and fp16's ~10-bit mantissa introduces absolute
    # rounding error that scales with magnitude. That's expected fp16
    # output-precision behavior, not a bug in the dequant arithmetic
    # (verified separately: the fp32-precision result before the fp16 cast
    # matches the reference exactly).
    assert rel_err < 1e-2, "CPU vGPU path diverged from reference matmul"
    print("[vgpu_backend self-test] PASSED")
