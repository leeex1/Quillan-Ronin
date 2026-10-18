"""LOCAL PROOF BENCH - real measured numbers from THIS box. No gimmicks.
CPU quant vs fallback, prism native vs torch, gateway tok/s, system state."""
import sys
import time
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
REPO = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO / "03 - Training & Model" / "src" / "csrc"))
import torch
torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4) - 1))
from quillan_kernels import load_native_kernels, native_weight_quant

out = {"ts": datetime.now(timezone.utc).isoformat(), "box": {}, "bench": {}}
mod = load_native_kernels()
out["box"]["native_loaded"] = mod is not None
out["box"]["torch"] = torch.__version__ + "+cu" + (torch.version.cuda or "?")
out["box"]["cuda_available"] = torch.cuda.is_available()
out["box"]["cpu_threads"] = torch.get_num_threads()


def med(fn, reps=7):
    ts = sorted((lambda t0: (fn(), time.perf_counter() - t0)[1])(time.perf_counter())
                for _ in range(reps))
    return ts[len(ts) // 2] * 1000


w = torch.randn(1024, 1024)
if mod is not None:
    native_ms = med(lambda: native_weight_quant(w))
else:
    native_ms = None
def _fb():
    scale = 1.0 / w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    ws = w * scale
    wq = torch.round(torch.clamp(ws, -1.0, 1.0))
    return (ws + (wq - ws).detach()) / scale
fb_ms = med(_fb)
out["bench"]["quant_1024_native_ms"] = round(native_ms, 2) if native_ms else None
out["bench"]["quant_1024_torch_ms"] = round(fb_ms, 2)
if native_ms:
    out["bench"]["quant_speedup_x"] = round(fb_ms / native_ms, 2)

x = torch.randn(2, 64, 1024)
ws_ = torch.randn(9, 1024, 1024)
wg = torch.randn(1024, 1024)
if mod is not None:
    try:
        prism_ms = med(lambda: mod.nine_vector_prism_forward_cpu(x, ws_, wg))
    except Exception as e:
        prism_ms = f"ERR {e}"
else:
    prism_ms = None
import torch.nn.functional as F
def _pfb():
    prism = torch.einsum("bld,ned->ble", x, ws_) / 9.0
    return F.linear(prism, wg)
pfb_ms = med(_pfb)
out["bench"]["prism_native_ms"] = round(prism_ms, 2) if isinstance(prism_ms, float) else prism_ms
out["bench"]["prism_torch_ms"] = round(pfb_ms, 2)
if isinstance(prism_ms, float):
    out["bench"]["prism_speedup_x"] = round(pfb_ms / prism_ms, 2)

try:
    body = json.dumps({"model": "quillan-oni-mini-6l",
                       "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
                       "max_tokens": 30}).encode()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    comp = d["choices"][0]["message"]["content"]
    out["bench"]["gateway_s"] = round(dt, 1)
    out["bench"]["gateway_tok_s"] = round(30 / dt, 2)
    out["bench"]["gateway_sample"] = comp[:120]
except Exception as e:
    out["bench"]["gateway_err"] = str(e)[:200]

try:
    import subprocess
    smi = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    out["box"]["gpu"] = smi.stdout.strip()
except Exception as e:
    out["box"]["gpu"] = f"n/a {e}"
try:
    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = MS()
    ms.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    out["box"]["ram_free_gb"] = round(ms.ullAvailPhys / 1e9, 1)
    out["box"]["ram_total_gb"] = round(ms.ullTotalPhys / 1e9, 1)
except Exception as e:
    out["box"]["ram"] = f"n/a {e}"

p = REPO / "training_logs" / "bench_local.json"
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1), flush=True)
