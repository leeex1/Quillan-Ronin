"""Sweep: CPU fp32 matmul vs full GPU int8 pipeline (quant+H2D+kernel+D2H).
Sets the dispatcher split from measured numbers. No torch CUDA kernels used."""
import sys
import time
import torch
sys.path.insert(0, r"C:\02_QUILLAN\09 - Projects\projects\Chip design\quillan_sm61_kernels")
import quillan_sm61_qgemm as Q
torch.manual_seed(1)


def bench(M, K, N, reps=20):
    x = (torch.randn(M, K) * 2).float()
    w = (torch.randn(N, K) * 2).float()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.matmul(x, w.t())
    cpu_ms = (time.perf_counter() - t0) / reps * 1000
    xs = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    ws = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    x_i8 = torch.clamp((x / xs).round(), -128, 127).to(torch.int8)
    w_i8 = torch.clamp((w / ws).round(), -128, 127).to(torch.int8)
    xg, wg, xsg, wsg = x_i8.cuda(), w_i8.cuda(), xs.cuda(), ws.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _y = Q.qgemm_forward(xg, wg, xsg, wsg)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / reps * 1000
    return cpu_ms, gpu_ms


print(f"{'M':>5} {'K':>5} {'N':>5} {'CPU_ms':>9} {'GPU_ms':>9} {'winner':>6}", flush=True)
for M, K, N in [(1, 1024, 1024), (8, 1024, 1024), (32, 1024, 1024),
                (1, 1024, 2048), (8, 1024, 2048), (32, 2048, 1024),
                (128, 1024, 1024), (128, 2048, 2048)]:
    c, g = bench(M, K, N)
    w = "GPU" if g < c else "CPU"
    print(f"{M:>5} {K:>5} {N:>5} {c:>9.3f} {g:>9.3f} {w:>6}", flush=True)
print("SWEEP DONE", flush=True)
