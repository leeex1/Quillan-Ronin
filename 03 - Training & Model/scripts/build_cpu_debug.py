import sys
import traceback
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\src\csrc")
from pathlib import Path
_CS = Path(r"C:\02_QUILLAN\03 - Training & Model\src\csrc")
try:
    from torch.utils.cpp_extension import load
    m = load(name="quillan_native_ops",
             sources=[str(_CS / "bitnet_cpu_avx2.cpp")],
             extra_cflags=["/O2", "/openmp", "/arch:AVX2"],
             verbose=True)
    print("module:", m, flush=True)
except Exception:
    traceback.print_exc()
