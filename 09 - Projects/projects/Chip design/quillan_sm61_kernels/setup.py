"""
Build script for the sm_61 quantized GEMM PyTorch extension.

Usage:
    python setup.py build_ext --inplace
    # or
    pip install -e .

This explicitly targets compute capability 6.1 (Pascal / GTX 1050 family)
regardless of what TORCH_CUDA_ARCH_LIST your installed PyTorch build itself
used -- that only affects PyTorch's own bundled kernels, not extensions you
compile yourself. What DOES matter is that the local `nvcc` you build this
with is from a CUDA toolkit version compatible with your installed PyTorch's
CUDA runtime (check `python -c "import torch; print(torch.version.cuda)"`
and match your toolkit major version to it, or at minimum stay within
NVIDIA's minor-version compatibility guarantees).

We embed both SASS (sm_61) and PTX (compute_61) so the driver can either run
the native binary directly or JIT the PTX if it ever needs to on a newer
architecture/driver combo -- this is the actual "bridge to modern CUDA"
mechanism: PTX forward-compatibility, not magic.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")

setup(
    name="quillan_sm61_qgemm",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="quillan_sm61_qgemm",
            sources=["qgemm_binding.cpp", "sm61_qgemm.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_61,code=sm_61",   # native Pascal SASS
                    "-gencode=arch=compute_61,code=compute_61",  # PTX fallback
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
