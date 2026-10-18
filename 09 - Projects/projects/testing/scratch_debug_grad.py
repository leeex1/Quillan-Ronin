#!/usr/bin/env python3
import torch
import torch.nn.functional as F

x = torch.randn(2, 16, requires_grad=True)
h_p = x.detach().float()
h_c = x.float()

norm_p = F.normalize(h_p, dim=-1)
norm_c = F.normalize(h_c, dim=-1)
overlap = (norm_p * norm_c).sum(dim=-1)
fidelity = overlap.pow(2).clamp(0.0, 1.0)

# Unsafe:
# trace_dist = torch.sqrt(torch.clamp(1.0 - fidelity, min=0.0))

# Safe with epsilon inside sqrt:
trace_dist_safe = torch.sqrt(torch.clamp(1.0 - fidelity, min=0.0) + 1e-8) - (1e-8 ** 0.5)

iq = (1.0 * fidelity).mean() - 0.1 * trace_dist_safe.mean()
iq.backward()
print("Grad of x has NaN?:", torch.isnan(x.grad).any().item())
print("Grad norm:", x.grad.norm().item())
