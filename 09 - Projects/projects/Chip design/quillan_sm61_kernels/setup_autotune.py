"""
Build script for the sm_61 autotuning GEMM PyTorch extension.

This builds the multi-kernel version with 8 different configurations
for automatic dispatch based on matrix shape.

Usage:
    python setup_autotune.py build_ext --inplace
    # or
    pip install -e .
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")

setup(
    name="quillan_sm61_qgemm_autotune",
    version="0.2.0",
    ext_modules=[
        CUDAExtension(
            name="quillan_sm61_qgemm_autotune",
            sources=["qgemm_autotune_binding.cpp", "sm61_qgemm_autotune.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_61,code=sm_61",
                    "-gencode=arch=compute_61,code=compute_61",
                    "--use_fast_math",
                    "-Xptxas -dlcm=ca",
                    "-maxrregcount=64",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
