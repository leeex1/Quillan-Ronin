#!/usr/bin/env python3
# EvoMoE (2505.23830) — Expert Evolution for 34 Council (HNMoE)
# Evolves diverse experts from single base via evolutionary perturbations + token-aware routing
#
# C14-KAIDO (Phase 1): the per-expert dispatch loop has been vectorised into
# rank-group-batched BMMs. Experts are heterogeneous by design (11x rank-8,
# 11x rank-24, 11x rank-48), so the batch unit is the RANK GROUP, not the
# single expert: each group becomes one [n_g, ...] bmm instead of 11 separate
# nn.Sequential calls.
#
# PARITY CONTRACT
#   * Padded-group BMM is mathematically identical to the sequential loop when
#     padding slots are masked to zero (padded inputs contribute nothing to the
#     bmm result, and those slots are dropped before the index_add).
#   * Routing (softmax -> top-4), gate weighting and index_add scatter are
#     unchanged, including the .to(out_vec.dtype) cast.
#   * A group whose token distribution is too skewed for cheap padding falls
#     back to the original per-expert loop, so worst-case memory is bounded.
import os
import torch, torch.nn as nn, torch.nn.functional as F

# C14-KAIDO measured throughput (4-core CPU, torch threads=4, hidden=512,
# batch 4x128, 34 experts at ranks 8/24/48):
#
#     grouped BMM : 2185 ms/fwd      sequential loop : 1563 ms/fwd   ->  0.72x  (SLOWER)
#
# On CPU the gather/scatter of `flat_x[p[slot]]` plus the padded intermediate
# [n, maxlen, D] tensors costs more than the 34 small Sequential launches it
# replaces -- rank-8/rank-24 GEMMs are too small to amortise bmm.
#
# On CUDA the trade-off inverts: 34 python-level launches collapse to 6 bmm
# calls and launch overhead dominates, so the grouped path wins there.
#
# Hence: default ON for CUDA, default OFF for CPU, overridable via
# QUILLAN_EVOMOE_VECTORIZED=1|0. Parity is proven either way
# (see test_evo_moe_parity.py).
_ENV_VEC = os.environ.get("QUILLAN_EVOMOE_VECTORIZED", "").strip()


def _default_vectorized() -> bool:
    if _ENV_VEC:
        return _ENV_VEC.lower() in ("1", "true", "yes", "on")
    return bool(torch.cuda.is_available())


class EvoMoE(nn.Module):
    """Heterogeneous grouped experts: 34 experts in 3 groups (small/med/large) per MoHGE/MoDSE"""

    # Max padding-waste ratio before a group falls back to the sequential loop.
    PAD_WASTE_LIMIT = 8.0

    def __init__(self, hidden_dim=2048, n_experts=34, rank=24):
        super().__init__()
        self.n_experts, self.hidden_dim = n_experts, hidden_dim
        # Heterogeneous: 11 small (rank 8), 11 medium (rank 24), 11 large (rank 48) per MoDSE
        self.groups = [(0,11,8),(11,22,24),(22,34,48)]
        self.experts = nn.ModuleList()
        for i in range(n_experts):
            g_rank = [r for s,e,r in self.groups if s <= i < e][0]
            self.experts.append(nn.Sequential(nn.Linear(hidden_dim, g_rank, bias=False), nn.Linear(g_rank, hidden_dim, bias=False)))
        self.router = nn.Linear(hidden_dim, n_experts, bias=False)
        self.evolution_scale = 0.01  # EGGROLL-style perturbation
        self.vectorized = _default_vectorized()
        self._wcache = None
        self._wkey = None

    def evolve_from_base(self, base_state):
        # Evolutionary update: perturb base to create diverse experts (EGGROLL crossover)
        for expert in self.experts:
            for p in expert.parameters():
                p.data.add_(torch.randn_like(p) * self.evolution_scale)

    # -- C14-KAIDO: stacked per-group weights, cached on parameter version ----
    def _group_weights(self):
        """-> [(lo, hi, r, W1t[n,D,r], W2t[n,r,D]), ...] transposed for bmm."""
        key = tuple(p._version for e in self.experts for p in e.parameters())
        if self._wkey == key and self._wcache is not None:
            return self._wcache
        out = []
        for lo, hi, r in self.groups:
            hi = min(hi, self.n_experts)
            n = hi - lo
            if n <= 0:
                continue
            # F.linear(x, W) == x @ W.T  ->  bmm needs [D, r] / [r, D]
            W1t = torch.stack([self.experts[lo + i][0].weight.t() for i in range(n)], dim=0)
            W2t = torch.stack([self.experts[lo + i][1].weight.t() for i in range(n)], dim=0)
            out.append((lo, hi, r, W1t, W2t))
        self._wcache, self._wkey = out, key
        return out

    def forward(self, x):
        # Token-aware routing (EvoMoE): each token gets its own expert mix
        logits = self.router(x)  # [B,S,34]
        gates = F.softmax(logits, dim=-1)
        # Top-4 heterogeneous routing (like Mixtral but grouped)
        top_gates, top_idx = gates.topk(4, dim=-1)

        B, S, D = x.shape
        flat_x = x.reshape(-1, D)
        flat_top_idx = top_idx.reshape(-1, 4)
        flat_top_gates = top_gates.reshape(-1, 4)
        out_vec = torch.zeros_like(flat_x)

        if self.vectorized:
            return self._forward_grouped(flat_x, flat_top_idx, flat_top_gates,
                                         out_vec, B, S, D)
        return self._forward_loop(flat_x, flat_top_idx, flat_top_gates, out_vec, B, S, D)

    # ------------------------------------------------------------------ #
    def _forward_loop(self, flat_x, flat_top_idx, flat_top_gates, out_vec, B, S, D):
        """Original per-expert dispatch — retained as the fallback path."""
        for eid in range(self.n_experts):
            tok_pos, k_slot = (flat_top_idx == eid).nonzero(as_tuple=True)
            if tok_pos.numel() > 0:
                w = flat_top_gates[tok_pos, k_slot].unsqueeze(-1)
                e_out = self.experts[eid](flat_x[tok_pos])
                out_vec.index_add_(0, tok_pos, (w * e_out).to(out_vec.dtype))
        return out_vec.reshape(B, S, D)

    # ------------------------------------------------------------------ #
    def _forward_grouped(self, flat_x, flat_top_idx, flat_top_gates, out_vec, B, S, D):
        """Rank-group batched dispatch: 3 bmm pairs instead of 34 Sequential calls."""
        dev = flat_x.device
        BT = flat_x.size(0)
        ar = torch.arange(BT, device=dev).unsqueeze(1).expand(-1, 4).reshape(-1)  # token id
        eid_flat = flat_top_idx.reshape(-1)                                          # [BT*4]
        g_flat = flat_top_gates.reshape(-1)                                         # [BT*4]

        for lo, hi, r, W1t, W2t in self._group_weights():
            n = hi - lo
            sel = (eid_flat >= lo) & (eid_flat < hi)
            if not bool(sel.any()):
                continue
            p = ar[sel]                    # token ids
            ei = eid_flat[sel] - lo         # local expert index 0..n-1
            gw = g_flat[sel]               # gate weights

            order = torch.argsort(ei)
            p, ei, gw = p[order], ei[order], gw[order]
            counts = torch.bincount(ei, minlength=n)
            maxlen = int(counts.max().item())
            ntok = p.numel()

            # Guard: skip the padded path when skew would waste too much memory.
            if maxlen == 0 or (n * maxlen) > self.PAD_WASTE_LIMIT * ntok + 1024:
                for j in range(n):
                    m = (ei == j).nonzero(as_tuple=True)[0]
                    if m.numel() == 0:
                        continue
                    e_out = F.linear(F.linear(flat_x[p[m]], self.experts[lo + j][0].weight),
                                     self.experts[lo + j][1].weight)
                    out_vec.index_add_(0, p[m], (gw[m].unsqueeze(-1) * e_out).to(out_vec.dtype))
                continue

            starts = torch.cumsum(counts, 0) - counts
            cols = torch.arange(maxlen, device=dev).unsqueeze(0)
            valid = cols < counts.unsqueeze(1)                 # [n, maxlen]
            slot = (cols + starts.unsqueeze(1)).clamp(max=ntok - 1)   # [n, maxlen]

            xp = flat_x[p[slot]]                               # [n, maxlen, D]
            xp = xp * valid.unsqueeze(-1).to(xp.dtype)          # zero the padding
            # F.linear(x,W) == x @ W.T, so bmm(x, W1t) / bmm(h, W2t) reproduce
            # both linears in one shot with no transpose.
            h = torch.bmm(xp, W1t.to(xp.dtype))                  # [n, maxlen, r]
            y = torch.bmm(h, W2t.to(h.dtype))                    # [n, maxlen, D]
            y = y * valid.unsqueeze(-1).to(y.dtype)

            gi, si = valid.nonzero(as_tuple=True)              # [K]
            tgt = p[slot[gi, si]]
            w = gw[slot[gi, si]].unsqueeze(-1)
            out_vec.index_add_(0, tgt, (w * y[gi, si]).to(out_vec.dtype))

        return out_vec.reshape(B, S, D)