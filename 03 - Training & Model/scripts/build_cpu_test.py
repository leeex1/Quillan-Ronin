import sys
import time
import torch
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\src\csrc")
from quillan_kernels import load_native_kernels, native_weight_quant
m = load_native_kernels()
print("module:", m, flush=True)
if m is None:
    print("NATIVE BUILD FAILED (fallback mode)", flush=True)
    sys.exit(1)
w = torch.randn(512, 1024)
t0 = time.perf_counter()
for _ in range(5):
    q = native_weight_quant(w)
t1 = time.perf_counter()
print(f"native quant 5x512x1024: {(t1 - t0) * 1000:.0f}ms", flush=True)
print(f"ternary check unique vals: {torch.unique(q).tolist()}", flush=True)
t0 = time.perf_counter()
for _ in range(5):
    scale = 1.0 / w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    ws = w * scale
    wq = torch.round(torch.clamp(ws, -1.0, 1.0))
    _ = (ws + (wq - ws).detach()) / scale
t1 = time.perf_counter()
print(f"torch fallback 5x: {(t1 - t0) * 1000:.0f}ms", flush=True)
print("CPU QUANT OK", flush=True)
