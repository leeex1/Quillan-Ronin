"""QGemmSTE: BitNet-style straight-through estimator on our sm_61 DP4A kernel.
Forward: int8 GPU (proven kernel). Backward: fp32 CPU (exact grads).
Drop-in QLinear for future training scripts. Additive: touches no model file."""
import sys
import time
import torch
import torch.nn as nn
sys.path.insert(0, r"C:\02_QUILLAN\09 - Projects\projects\Chip design\quillan_sm61_kernels")
import quillan_sm61_qgemm as Q


class QGemmSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        xf, wf = x.float(), w.float()
        ctx.save_for_backward(xf, wf)
        xs = xf.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
        ws = wf.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
        xi = torch.clamp((xf / xs).round(), -128, 127).to(torch.int8).cuda()
        wi = torch.clamp((wf / ws).round(), -128, 127).to(torch.int8).cuda()
        y = Q.qgemm_forward(xi, wi, xs.cuda(), ws.cuda())
        torch.cuda.synchronize()
        return y.cpu().float()

    @staticmethod
    def backward(ctx, go):
        xf, wf = ctx.saved_tensors
        gf = go.float()
        return gf @ wf, gf.t() @ xf


class QLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_f, in_f) * 0.02)

    def forward(self, x):
        return QGemmSTE.apply(x, self.w)


if __name__ == "__main__":
    torch.manual_seed(2)
    ql = QLinear(64, 32)
    x = torch.randn(8, 64, requires_grad=True)
    t0 = time.perf_counter()
    y = ql(x)
    loss = y.sum()
    loss.backward()
    dt = (time.perf_counter() - t0) * 1000
    ref = x.detach() @ ql.w.detach().t()
    err = (y.detach() - ref).abs().max().item() / ref.abs().max().item()
    gref = torch.ones_like(y.detach()) @ ql.w.detach()
    gerr = (x.grad - gref).abs().max().item() / gref.abs().max().item()
    print(f"STE fwd+bwd 8x64x32: {dt:.1f}ms fwd_rel_err={err:.4f} grad_rel_err={gerr:.4f}",
          flush=True)
    print("STE OK" if err < 0.05 and gerr < 1e-4 else "STE DEVIATES", flush=True)
