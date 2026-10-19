import sys
import traceback
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\src\csrc")
import quillan_kernels as qk
print("QUILLAN_CPU_ONLY =", __import__("os").environ.get("QUILLAN_CPU_ONLY"), flush=True)
try:
    import torch
    from torch.utils.cpp_extension import load
    from pathlib import Path
    _CS = Path(r"C:\02_QUILLAN\03 - Training & Model\src\csrc")
    m = load(name="quillan_native_ops",
             sources=[str(_CS / "bitnet_cpu_avx2.cpp")],
             extra_cflags=["/O2", "/openmp", "/arch:AVX2"],
             extra_cuda_cflags=None,
             verbose=False)
    print("direct-with-None:", m, flush=True)
except Exception:
    traceback.print_exc()
qk._NATIVE_KERNEL_MODULE = None
try:
    m2 = qk.load_native_kernels()
    print("wrapper:", m2, flush=True)
except Exception:
    traceback.print_exc()
