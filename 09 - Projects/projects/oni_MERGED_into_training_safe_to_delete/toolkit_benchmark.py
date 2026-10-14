#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan toolkit benchmark — measured numbers only.

Probes (all executed, none simulated):
  1. CPU matmul throughput vs thread count (512 + 1024 shapes)
  2. Pinned-memory offload round-trip bandwidth (256MB staging buffer)
  3. vGPU CPU int8 GEMM path (vgpu_backend.vgpu_qgemm_cpu) throughput
  4. CUDA probe: real torch matmul on cuda:0 — expected to FAIL on sm_61
     with stock torch. A failure is recorded as cuda_usable=false; the
     vGPU split stays 0.0 (all-CPU). No simulated GPU numbers, ever.

Outputs:
  - 05_Training/training_logs/toolkit_benchmark.json (raw measurements)
  - updates configs/hybrid_compute.json (torch threads + vgpu split only)
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"C:\02_QUILLAN\00 - Meta")
sys.path.insert(0, str(ROOT / "oni"))
sys.path.insert(0, str(Path(r"C:\02_QUILLAN\02_Projects\Chip design\quillan_sm61_kernels")))

LOG = Path(r"C:\02_QUILLAN\05_Training\training_logs\toolkit_benchmark.json")
HYBRID = Path(r"C:\02_QUILLAN\00 - Meta\configs\hybrid_compute.json")


def bench_cpu_matmul(threads: int, size: int = 1024, iters: int = 10) -> float:
    """Returns GFLOP/s measured with given thread count."""
    torch.set_num_threads(threads)
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    # warmup
    for _ in range(2):
        _ = a @ b
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    dt = (time.perf_counter() - t0) / iters
    gflops = (2.0 * size ** 3 / 1e9) / dt
    return round(gflops, 2)


def bench_pinned_roundtrip(mb: int = 256) -> dict:
    """Pinned staging buffer alloc + CPU round-trip bandwidth (GB/s)."""
    n = mb * 1024 * 1024 // 4
    try:
        buf = torch.empty(n, dtype=torch.float32, pin_memory=True)
        src = torch.randn(n, dtype=torch.float32)
        t0 = time.perf_counter()
        buf.copy_(src)
        back = torch.empty_like(src)
        back.copy_(buf)
        dt = time.perf_counter() - t0
        gbs = (2 * mb / 1024) / dt
        return {"ok": True, "gb_s": round(gbs, 2), "mb": mb}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def bench_vgpu_cpu(m: int = 256, k: int = 512, n: int = 256, iters: int = 20) -> dict:
    """Throughput of the verified vGPU CPU int8 path (ms/iter)."""
    from vgpu_backend import vgpu_qgemm_cpu
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    xs = torch.rand(m) * 0.05 + 0.001
    ws = torch.rand(n) * 0.05 + 0.001
    _ = vgpu_qgemm_cpu(x, w, xs, ws)  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = vgpu_qgemm_cpu(x, w, xs, ws)
    ms = (time.perf_counter() - t0) / iters * 1000
    gops = (2.0 * m * n * k / 1e9) / (ms / 1000)
    return {"ms_per_iter": round(ms, 3), "gops": round(gops, 2),
            "shape": [m, k, n]}


def probe_cuda() -> dict:
    """Honest CUDA probe. Records success OR the exact failure reason."""
    out = {"torch_cuda_version": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        out.update({"usable": False, "reason": "torch.cuda.is_available() is False"})
        return out
    try:
        name = torch.cuda.get_device_name(0)
        out["device_name"] = name
        a = torch.randn(256, 256, device="cuda:0")
        b = torch.randn(256, 256, device="cuda:0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        out.update({"usable": True, "matmul_256_ms": round((time.perf_counter() - t0) * 1000, 3)})
    except Exception as e:
        out.update({"usable": False, "reason": str(e)[:300]})
    return out


def main():
    res = {"note": "measured on host, no simulation"}
    # 1. thread sweep, median-of-3 (host load varies wildly — see RAM hogs).
    # A single sample once claimed 315 GFLOP/s @4T and later 56 @4T.
    import statistics
    sweep = {}
    for th in (1, 2, 4):
        reps = sorted(bench_cpu_matmul(th, size=512, iters=8) for _ in range(3))
        sweep[f"threads_{th}"] = reps[1]
    res["thread_sweep_note"] = "median-of-3; re-run with a quiet box if spread is large"
    res["cpu_matmul_gflops_512"] = sweep
    best_threads = max(sweep, key=sweep.get).split("_")[1]
    res["best_threads"] = int(best_threads)
    res["cpu_matmul_gflops_1024_t4"] = bench_cpu_matmul(4, size=1024, iters=6)
    torch.set_num_threads(4)

    # 2. pinned offload
    res["pinned_roundtrip"] = bench_pinned_roundtrip()

    # 3. vgpu cpu path
    try:
        res["vgpu_cpu"] = bench_vgpu_cpu()
    except Exception as e:
        res["vgpu_cpu"] = {"ok": False, "error": str(e)[:200]}

    # 4. cuda probe
    res["cuda_probe"] = probe_cuda()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Update hybrid config ONLY from measurements
    cfg = json.loads(HYBRID.read_text(encoding="utf-8"))
    cfg["torch"]["num_threads"] = int(res["best_threads"])
    cfg["torch"]["num_interop_threads"] = min(2, int(res["best_threads"]))
    if res["cuda_probe"].get("usable"):
        cfg["cuda_status"]["sm_61_supported_by_torch"] = True
        cfg["routing"]["train_compute"] = "cuda"
        cfg["vgpu"]["split_ratio_gpu"] = 0.8
        cfg["vgpu"]["note"] = "CUDA probe passed — split set pending vGPU sweep"
    else:
        cfg["cuda_status"]["sm_61_supported_by_torch"] = False
        cfg["routing"]["train_compute"] = "cpu"
        cfg["vgpu"]["split_ratio_gpu"] = 0.0
    cfg["updated_by"] = "toolkit_benchmark.py"
    HYBRID.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(json.dumps(res, indent=2))
    print(f"[BENCH] wrote {LOG} and updated {HYBRID}")


if __name__ == "__main__":
    main()
