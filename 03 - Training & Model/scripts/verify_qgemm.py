"""Acceptance: quillan_sm61_qgemm vs torch CPU reference on the GTX 1050."""
import sys
import time
import torch
sys.path.insert(0, r"C:\02_QUILLAN\09 - Projects\projects\Chip design\quillan_sm61_kernels")
import quillan_sm61_qgemm as Q
print("module:", Q, flush=True)
print("has qgemm_forward:", hasattr(Q, "qgemm_forward"), flush=True)
torch.manual_seed(0)
M, K, N = 17, 64, 9
x = (torch.randn(M, K) * 2).to(torch.float32)
w = (torch.randn(N, K) * 2).to(torch.float32)
xs = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
ws = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
x_i8 = torch.clamp((x / xs).round(), -128, 127).to(torch.int8).cuda()
w_i8 = torch.clamp((w / ws).round(), -128, 127).to(torch.int8).cuda()
xs_c = xs.cuda()
ws_c = ws.cuda()
t0 = time.perf_counter()
y = Q.qgemm_forward(x_i8, w_i8, xs_c, ws_c)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1000
ref = torch.matmul(x, w.t())
got = y.cpu().float()
ae = (got - ref).abs().max().item()
re_ = ae / ref.abs().max().item()
print(f"qgemm {M}x{K}x{N} on sm_61: {dt:.2f}ms max_abs_err={ae:.6f} max_rel_err={re_:.6f}",
      flush=True)
print("GPU KERNEL ACCEPTED" if re_ < 0.05 else "GPU KERNEL DEVIATES", flush=True)
