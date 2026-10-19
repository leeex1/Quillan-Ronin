# sm_61 Quantized GEMM + CPU vGPU Offload

Two pieces:

1. **`sm61_qgemm.cu` + `qgemm_binding.cpp` + `setup.py`** — a real CUDA kernel
   for INT8×INT8 GEMM using Pascal's native `__dp4a` instruction, wrapped as
   a PyTorch extension. Targets compute capability 6.1 (GTX 1050 / GP107)
   explicitly.
2. **`vgpu_backend.py`** — there is no free-floating "virtual GPU" without
   either real NVIDIA vGPU/MIG hardware partitioning, or a software compute
   path on a *different* device that you offload to. This is the latter: a
   numerically-matching CPU implementation of the same op, plus a
   dispatcher that runs a slice of the batch on CUDA and a slice on CPU
   concurrently, then recombines.

## Why DP4A specifically

Pascal (sm_61) has no tensor cores — those start at Volta (sm_70). DP4A
(4-wide int8 dot-product-accumulate in one instruction) is the fastest
integer primitive Pascal actually has, and it's genuinely present on GTX
1050-class hardware. This kernel is built around it rather than trying to
fake tensor-core-style behavior that the chip doesn't support.

## Build

Requirements: a local CUDA toolkit whose major version matches (or is
compatible with) the CUDA runtime your installed PyTorch was built against.
Check with:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

Then:

```bash
cd quillan_kernels
python setup.py build_ext --inplace
```

`setup.py` explicitly passes:

```
-gencode=arch=compute_61,code=sm_61        # native Pascal machine code
-gencode=arch=compute_61,code=compute_61   # embedded PTX
```

The PTX embedding is the actual "bridge to modern CUDA" mechanism — if a
future driver/toolkit combination doesn't ship native sm_61 SASS anymore
(this hasn't happened as of current CUDA 12.x releases; Pascal remains
supported), the driver can JIT-compile the embedded PTX at load time
instead. This is standard NVIDIA forward-compatibility, not something
custom to this kernel — it just needs to be requested explicitly, which
most default build configs won't do for an old architecture.

**What this does NOT solve:** if NVIDIA fully drops Pascal driver support
at some future CUDA major version, no amount of kernel code fixes that —
that's a driver-support decision, not a compilation-target problem. Check
NVIDIA's current CUDA toolkit release notes for your card's support status
before relying on this long-term.

## Use

```python
import torch
import quillan_sm61_qgemm  # the compiled extension

# x: [M, K] fp16/fp32 activations -> quantize per-row before calling
# w: [N, K] fp16/fp32 weights (pre-transposed) -> quantize per-column once, offline

x_i8, x_scale = quantize_per_row(x)      # you write this: symmetric int8 quant
w_i8, w_scale = quantize_per_col(w)      # do this once at load time, not per forward

y_fp16 = quillan_sm61_qgemm.qgemm_forward(
    x_i8.cuda(), w_i8.cuda(), x_scale.cuda(), w_scale.cuda()
)
```

Quantization helpers aren't included — how you compute per-row/per-column
scales (symmetric vs asymmetric, clipping percentile, etc.) is a modeling
decision that depends on what you're quantizing, and I'm not going to
silently pick one for you and bury the choice in this file.

## Offloading with the vGPU path

```python
from vgpu_backend import HeterogeneousDispatcher
import quillan_sm61_qgemm

dispatcher = HeterogeneousDispatcher(
    cuda_op=quillan_sm61_qgemm.qgemm_forward,
    split_ratio=0.8,  # tune this against YOUR measured GPU vs CPU throughput
)

y = dispatcher(x_i8, w_i8, x_scale, w_scale)  # x_i8/w_i8 can start on CPU
```

`split_ratio` is not something I can pick correctly for you sight-unseen —
it depends on your specific CPU core count, your GPU's actual measured
throughput for this op, and your batch sizes. Benchmark both paths
standalone on your hardware first, then set the ratio to balance wall-clock
completion time between the two threads.

## What's actually verified here

- The CPU-path dequant arithmetic (`vgpu_backend.py`) was checked against an
  independent per-element reference loop in this environment: exact at
  fp32, ~0.045% relative error after the fp16 output cast (expected fp16
  rounding, not a logic error). Run `python vgpu_backend.py` yourself to
  reproduce.
- The CUDA kernel (`sm61_qgemm.cu`) was **not** compiled or run anywhere in
  producing this — there's no GPU or `nvcc` in the environment this was
  written in. It's structurally correct CUDA/DP4A code following the
  documented instruction semantics, but you need to build and test it
  against your actual GTX 1050 before trusting it in anything real. Start
  by testing `qgemm_forward` against a plain `torch.matmul` on
  dequantized fp32 tensors for a few random small shapes and diff the
  outputs.

## Known limitations of the kernel as written

- Reference-quality, not tuned: no double-buffering, no bank-conflict
  swizzling beyond basic alignment, one output element per thread. Profile
  against `cuBLAS`'s int8 GEMM before assuming this is faster in practice —
  it may not be for larger shapes.
- No backward pass. This is a forward-only inference kernel. If you need
  gradients through it, you'd want a straight-through-estimator wrapper
  around the quantization step (dequantize -> real matmul -> requantize),
  not through this kernel itself.
- `M`, `N`, `K` are read as `int`; this will silently misbehave past
  ~2^31 elements in a dimension. Not a realistic concern for a linear
  layer, worth knowing if you repurpose this for something larger.
