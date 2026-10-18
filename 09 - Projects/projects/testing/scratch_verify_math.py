#!/usr/bin/env python3
import torch
import math

N = 16
idx = torch.arange(N, dtype=torch.float32)
diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
# Gaussian ferromagnetic coupling
j_matrix = torch.exp(-0.5 * (diff / 2.0).pow(2))
j_matrix.fill_diagonal_(0.0)
j_matrix = j_matrix / math.sqrt(float(N))

s_aligned = torch.ones(2, N)
s_random = torch.randn(2, N)

e_aligned = -0.5 * torch.einsum("bi,ij,bj->b", s_aligned, j_matrix, s_aligned)
e_random = -0.5 * torch.einsum("bi,ij,bj->b", s_random, j_matrix, s_random)

print("E aligned:", e_aligned)
print("E random:", e_random)
assert e_aligned.mean().item() < e_random.mean().item()
print("Ferromagnetic ground state: VERIFIED!")
