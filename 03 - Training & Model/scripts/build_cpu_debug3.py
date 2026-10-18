import os
import sys
print("ENV:", os.environ.get("QUILLAN_CPU_ONLY"), flush=True)
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\src\csrc")
import torch
print("cuda avail:", torch.cuda.is_available(), flush=True)
import torch.utils.cpp_extension as ce
_orig_load = ce.load
def spy_load(*a, **k):
    print("LOAD name=", k.get("name"), flush=True)
    print("LOAD sources=", k.get("sources"), flush=True)
    return _orig_load(*a, **k)
ce.load = spy_load
import quillan_kernels as qk
print("wrapper has_cuda would be:",
      torch.cuda.is_available() and os.environ.get("QUILLAN_CPU_ONLY", "0") != "1", flush=True)
try:
    qk.load_native_kernels()
except Exception as e:
    print("wrapper raised:", type(e).__name__, str(e)[:200], flush=True)
