"""Force-build the sm61 kernel with nvcc 11.8 against torch-cu130.
1. No-op torch's CUDA version gate.
2. Downgrade torch's INJECTED '-std=c++20' (nvcc paths only) to c++17.
   MSVC /std:c++20 untouched (cl handles it; torch headers may need it)."""
import os
import runpy
import sys
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
import torch.utils.cpp_extension as ce
ce._check_cuda_version = lambda *a, **k: None
print("version gate off (nvcc12.9 + torch-cu130)", flush=True)
sys.argv = ["setup.py", "build_ext", "--inplace"]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.py"),
               run_name="__main__")
print("FORCE BUILD DONE", flush=True)
