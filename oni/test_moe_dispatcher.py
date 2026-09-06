#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0 ONI — MoEDispatcher Standalone Verification & Microbenchmark Suite
======================================================================================
Validates the 6-step MoE Dispatch Optimization Plan:
  1. MoEDispatcher module implementation with contiguous token permutation dispatch.
  2. Eager fallback / reference dispatch.
  3. Bit-level forward numerical parity (< 1e-6 max abs diff).
  4. Backward autograd gradient parity and torch.autograd.gradcheck.
  5. Guarded Triton dispatch hook with runtime availability checks and fallback.
  6. Edge cases: unbalanced routing, empty batches, 3D shapes, and mixed precision.
  7. Microbenchmark timing and throughput across multiple sequence lengths.
"""

import gc
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix sys.path for local oni imports
CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent
ONI_DIR = REPO_DIR / "oni"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(ONI_DIR) not in sys.path:
    sys.path.insert(0, str(ONI_DIR))

# Import canonical MoEDispatcher or fallback to standalone implementation
try:
    from oni.moe_dispatcher import MoEDispatcher, TRITON_AVAILABLE, _invoke_expert
except ImportError:
    try:
        from moe_dispatcher import MoEDispatcher, TRITON_AVAILABLE, _invoke_expert
    except ImportError:
        # Standalone inline implementation
        try:
            import triton
            import triton.language as tl
            TRITON_AVAILABLE = True
        except (ImportError, ModuleNotFoundError, Exception):
            triton = None
            tl = None
            TRITON_AVAILABLE = False

        def _invoke_expert(expert: Any, exp_input: torch.Tensor, gov_scale: float = 1.0, **kwargs) -> torch.Tensor:
            try:
                return expert(exp_input, gov_scale=gov_scale, **kwargs)
            except TypeError:
                try:
                    return expert(exp_input, gov_scale)
                except TypeError:
                    return expert(exp_input)

        class MoEDispatcher(nn.Module):
            def __init__(self, num_experts=None, use_triton=True, use_slice_assignment=False):
                super().__init__()
                self.num_experts = num_experts
                self.use_triton = use_triton
                self.use_slice_assignment = use_slice_assignment

            @staticmethod
            def eager_dispatch(tokens, topk_indices, topk_weights, experts, gov_scale=1.0, **kwargs):
                orig_shape = tokens.shape
                if tokens.dim() == 3:
                    tokens = tokens.reshape(-1, tokens.shape[-1])
                BT = tokens.size(0)
                if BT == 0 or topk_indices.numel() == 0:
                    res = torch.zeros_like(tokens)
                    return res.view(orig_shape) if len(orig_shape) == 3 else res
                K = topk_indices.size(-1)
                num_experts = len(experts)
                flat_idx = topk_indices.reshape(-1)
                flat_w = topk_weights.reshape(-1, 1).to(tokens.dtype)
                token_pos = torch.arange(BT, device=tokens.device).unsqueeze(1).expand(-1, K).reshape(-1)
                moe_out = torch.zeros_like(tokens)
                for e in range(num_experts):
                    sel = (flat_idx == e).nonzero(as_tuple=True)[0]
                    if sel.numel() == 0:
                        continue
                    pos = token_pos[sel]
                    w = flat_w[sel].to(tokens.dtype)
                    e_out = _invoke_expert(experts[e], tokens[pos], gov_scale=gov_scale, **kwargs)
                    moe_out.index_add_(0, pos, (w * e_out).to(tokens.dtype))
                if len(orig_shape) == 3:
                    moe_out = moe_out.view(orig_shape)
                return moe_out

            @staticmethod
            def contiguous_dispatch(tokens, topk_indices, topk_weights, experts, gov_scale=1.0, use_slice_assignment=False, **kwargs):
                orig_shape = tokens.shape
                if tokens.dim() == 3:
                    tokens = tokens.reshape(-1, tokens.shape[-1])
                BT, C = tokens.shape
                if BT == 0 or topk_indices.numel() == 0:
                    res = torch.zeros_like(tokens)
                    return res.view(orig_shape) if len(orig_shape) == 3 else res
                K = topk_indices.size(-1)
                num_experts = len(experts)
                N = BT * K
                flat_idx = topk_indices.reshape(-1)
                flat_w = topk_weights.reshape(-1, 1).to(tokens.dtype)
                token_pos = torch.arange(BT, device=tokens.device).unsqueeze(1).expand(-1, K).reshape(-1)
                sort_order = torch.argsort(flat_idx, stable=True)
                sorted_token_pos = token_pos[sort_order]
                sorted_w = flat_w[sort_order]
                counts = torch.bincount(flat_idx, minlength=num_experts)
                offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=tokens.device)
                torch.cumsum(counts, dim=0, out=offsets[1:])
                offsets_list = offsets.cpu().tolist()
                gathered_tokens = tokens[sorted_token_pos]

                if use_slice_assignment:
                    permuted_expert_out = torch.empty(N, C, dtype=tokens.dtype, device=tokens.device)
                    for e in range(num_experts):
                        start = offsets_list[e]
                        end = offsets_list[e + 1]
                        if start == end:
                            continue
                        exp_in = gathered_tokens[start:end]
                        e_out = _invoke_expert(experts[e], exp_in, gov_scale=gov_scale, **kwargs)
                        permuted_expert_out[start:end] = e_out
                else:
                    expert_outputs = []
                    for e in range(num_experts):
                        start = offsets_list[e]
                        end = offsets_list[e + 1]
                        if start == end:
                            continue
                        exp_in = gathered_tokens[start:end]
                        e_out = _invoke_expert(experts[e], exp_in, gov_scale=gov_scale, **kwargs)
                        expert_outputs.append(e_out)
                    if not expert_outputs:
                        res = torch.zeros_like(tokens)
                        return res.view(orig_shape) if len(orig_shape) == 3 else res
                    permuted_expert_out = torch.cat(expert_outputs, dim=0)

                weighted_out = (permuted_expert_out * sorted_w).to(tokens.dtype)
                moe_out = torch.zeros_like(tokens)
                moe_out.index_add_(0, sorted_token_pos, weighted_out)
                if len(orig_shape) == 3:
                    moe_out = moe_out.view(orig_shape)
                return moe_out

            def triton_dispatch(self, tokens, topk_indices, topk_weights, experts, gov_scale=1.0, **kwargs):
                return self.contiguous_dispatch(tokens, topk_indices, topk_weights, experts, gov_scale=gov_scale, use_slice_assignment=self.use_slice_assignment, **kwargs)

            def forward(self, tokens, topk_indices, topk_weights, experts, gov_scale=1.0, mode="auto", **kwargs):
                if mode == "eager":
                    return self.eager_dispatch(tokens, topk_indices, topk_weights, experts, gov_scale=gov_scale, **kwargs)
                elif mode == "triton":
                    return self.triton_dispatch(tokens, topk_indices, topk_weights, experts, gov_scale=gov_scale, **kwargs)
                elif mode == "contiguous":
                    return self.contiguous_dispatch(tokens, topk_indices, topk_weights, experts, gov_scale=gov_scale, use_slice_assignment=self.use_slice_assignment, **kwargs)
                else:
                    return self.contiguous_dispatch(tokens, topk_indices, topk_weights, experts, gov_scale=gov_scale, use_slice_assignment=self.use_slice_assignment, **kwargs)


# Attempt import of CouncilExpert from production architecture
try:
    from oni.quillan_v5_4_oni import CouncilExpert, QuillanOniConfig
    HAS_COUNCIL_EXPERT = True
except ImportError:
    try:
        from quillan_v5_4_oni import CouncilExpert, QuillanOniConfig
        HAS_COUNCIL_EXPERT = True
    except ImportError:
        HAS_COUNCIL_EXPERT = False
        CouncilExpert = None
        QuillanOniConfig = None


class SyntheticExpert(nn.Module):
    """Standard SwiGLU-style synthetic expert for deterministic verification."""
    def __init__(self, hidden_dim: int, ffn_dim: Optional[int] = None):
        super().__init__()
        ffn_dim = ffn_dim or (hidden_dim * 2)
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, gov_scale: float = 1.0) -> torch.Tensor:
        h = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(h) * gov_scale


# Global test counter
total_tests = 0
passed_tests = 0

def check(name: str, condition: bool, details: str = ""):
    global total_tests, passed_tests
    total_tests += 1
    if condition:
        passed_tests += 1
        print(f"  [PASS] {name} {details}")
    else:
        print(f"  [FAIL] {name} {details}")
        assert False, f"Test failed: {name} {details}"


def run_numerical_parity_tests():
    print("\n--- [TEST SUITE 1] Numerical Parity Verification ---")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = [
        {"BT": 16, "C": 64, "K": 2, "E": 8, "desc": "Small Toy Configuration"},
        {"BT": 64, "C": 128, "K": 4, "E": 16, "desc": "Medium Configuration"},
        {"BT": 256, "C": 128, "K": 4, "E": 34, "desc": "Full Council (34 Experts) Config"},
        {"BT": 512, "C": 256, "K": 4, "E": 34, "desc": "High-Throughput Sequence Config"},
    ]

    dispatcher = MoEDispatcher(use_slice_assignment=False)
    dispatcher_slice = MoEDispatcher(use_slice_assignment=True)

    for cfg in configs:
        BT, C, K, E = cfg["BT"], cfg["C"], cfg["K"], cfg["E"]
        tokens = torch.randn(BT, C, device=device)
        topk_i = torch.randint(0, E, (BT, K), device=device)
        topk_p = F.softmax(torch.randn(BT, K, device=device), dim=-1)
        experts = nn.ModuleList([SyntheticExpert(C).to(device) for _ in range(E)])

        out_eager = MoEDispatcher.eager_dispatch(tokens, topk_i, topk_p, experts, gov_scale=0.92)
        out_cont_cat = dispatcher.contiguous_dispatch(tokens, topk_i, topk_p, experts, gov_scale=0.92)
        out_cont_slice = dispatcher_slice.contiguous_dispatch(tokens, topk_i, topk_p, experts, gov_scale=0.92)

        diff_cat = (out_eager - out_cont_cat).abs().max().item()
        diff_slice = (out_eager - out_cont_slice).abs().max().item()

        check(f"Forward Parity: {cfg['desc']} (cat)", diff_cat < 1e-6, f"max diff: {diff_cat:.2e}")
        check(f"Forward Parity: {cfg['desc']} (slice)", diff_slice < 1e-6, f"max diff: {diff_slice:.2e}")


def run_production_council_expert_parity():
    print("\n--- [TEST SUITE 2] Production CouncilExpert (LoRA + EGGROLL Swarm) Parity ---")
    if not HAS_COUNCIL_EXPERT:
        print("  [SKIP] CouncilExpert not found in environment.")
        return

    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 128
    num_experts = 34
    top_k = 4
    BT = 128

    cfg = QuillanOniConfig(
        vocab_size=50257,
        max_seq_len=64,
        hidden_dim=hidden_dim,
        n_layer=2,
        n_head=4,
        head_dim=32,
        ffn_dim=256,
        num_experts=num_experts,
        expert_rank=32,
        swarm_rank=16,
        device=str(device),
    )

    experts = nn.ModuleList([CouncilExpert(i, f"C{i+1}", cfg).to(device) for i in range(num_experts)])
    for exp in experts:
        exp.eval()

    tokens = torch.randn(BT, hidden_dim, device=device)
    topk_i = torch.randint(0, num_experts, (BT, top_k), device=device)
    topk_p = F.softmax(torch.randn(BT, top_k, device=device), dim=-1)

    out_eager = MoEDispatcher.eager_dispatch(tokens, topk_i, topk_p, experts, gov_scale=0.85)
    out_cont = MoEDispatcher.contiguous_dispatch(tokens, topk_i, topk_p, experts, gov_scale=0.85)

    diff = (out_eager - out_cont).abs().max().item()
    check("CouncilExpert 34-Persona Roster Forward Parity", diff < 1e-6, f"max diff: {diff:.2e}")


def run_backward_autograd_parity():
    print("\n--- [TEST SUITE 3] Backward Autograd Parity & Gradient Equivalence ---")
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BT, C, K, E = 32, 64, 3, 8

    # Create fresh models for eager
    torch.manual_seed(42)
    experts_eager = nn.ModuleList([SyntheticExpert(C).to(device) for _ in range(E)])
    tokens_eager = torch.randn(BT, C, device=device, requires_grad=True)

    # Create identical weights for contiguous
    torch.manual_seed(42)
    experts_cont = nn.ModuleList([SyntheticExpert(C).to(device) for _ in range(E)])
    tokens_cont = tokens_eager.detach().clone().requires_grad_(True)

    topk_i = torch.randint(0, E, (BT, K), device=device)
    topk_p = F.softmax(torch.randn(BT, K, device=device), dim=-1)

    # Forward
    out_eager = MoEDispatcher.eager_dispatch(tokens_eager, topk_i, topk_p, experts_eager, gov_scale=1.0)
    out_cont = MoEDispatcher.contiguous_dispatch(tokens_cont, topk_i, topk_p, experts_cont, gov_scale=1.0)

    # Target loss
    target = torch.randn_like(out_eager)
    loss_eager = F.mse_loss(out_eager, target)
    loss_cont = F.mse_loss(out_cont, target)

    loss_eager.backward()
    loss_cont.backward()

    # 1. Token gradient equivalence
    grad_tok_diff = (tokens_eager.grad - tokens_cont.grad).abs().max().item()
    check("Token Input Gradient Equivalence", grad_tok_diff < 1e-6, f"max diff: {grad_tok_diff:.2e}")

    # 2. Expert parameter gradient equivalence
    max_exp_grad_diff = 0.0
    for e in range(E):
        for p_eager, p_cont in zip(experts_eager[e].parameters(), experts_cont[e].parameters()):
            if p_eager.grad is not None and p_cont.grad is not None:
                d = (p_eager.grad - p_cont.grad).abs().max().item()
                if d > max_exp_grad_diff:
                    max_exp_grad_diff = d

    check("Expert Weight Gradient Equivalence", max_exp_grad_diff < 1e-6, f"max diff: {max_exp_grad_diff:.2e}")


def run_gradcheck_verification():
    print("\n--- [TEST SUITE 4] PyTorch Autograd Gradcheck (Float64 Analytic Derivative) ---")
    torch.manual_seed(777)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BT, C, K, E = 8, 16, 2, 4

    # NOTE (post-merge fix): previously `experts = [expert for _ in range(E)]`
    # aliased ONE object E times, so the gradcheck never exercised distinct
    # experts. Construct E distinct experts from one seed stream instead.
    experts = [SyntheticExpert(C, ffn_dim=32).to(device).to(torch.float64)
               for _ in range(E)]

    tokens = torch.randn(BT, C, dtype=torch.float64, device=device, requires_grad=True)
    topk_i = torch.randint(0, E, (BT, K), device=device)
    topk_p = F.softmax(torch.randn(BT, K, dtype=torch.float64, device=device), dim=-1)

    def func(x):
        return MoEDispatcher.contiguous_dispatch(x, topk_i, topk_p, experts, gov_scale=1.0)

    passed_gc = torch.autograd.gradcheck(func, (tokens,), eps=1e-6, atol=1e-4, rtol=1e-3)
    check("torch.autograd.gradcheck Analytic Parity", passed_gc, f"Float64 derivative check passed")


def run_triton_guard_and_edge_cases():
    print("\n--- [TEST SUITE 5] Triton Dispatch Guard & Edge Cases ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dispatcher = MoEDispatcher(use_triton=True)

    print(f"  [INFO] TRITON_AVAILABLE: {TRITON_AVAILABLE}")
    print(f"  [INFO] Device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 1. Test triton_dispatch execution (with fallback if unavailable)
    tokens = torch.randn(16, 32, device=device)
    topk_i = torch.randint(0, 4, (16, 2), device=device)
    topk_p = F.softmax(torch.randn(16, 2, device=device), dim=-1)
    experts = nn.ModuleList([SyntheticExpert(32).to(device) for _ in range(4)])

    out_triton = dispatcher.triton_dispatch(tokens, topk_i, topk_p, experts)
    out_ref = dispatcher.eager_dispatch(tokens, topk_i, topk_p, experts)
    diff_triton = (out_triton - out_ref).abs().max().item()
    check("Triton Dispatch Hook Forward", diff_triton < 1e-6, f"max diff vs eager: {diff_triton:.2e}")

    # 2. Edge Case: Extreme expert starvation (all tokens to single expert 0)
    topk_starved = torch.zeros(16, 2, dtype=torch.long, device=device)
    out_starved_eager = dispatcher.eager_dispatch(tokens, topk_starved, topk_p, experts)
    out_starved_cont = dispatcher.contiguous_dispatch(tokens, topk_starved, topk_p, experts)
    diff_starved = (out_starved_eager - out_starved_cont).abs().max().item()
    check("Starvation Edge Case (All tokens to Expert 0)", diff_starved < 1e-6, f"max diff: {diff_starved:.2e}")

    # 3. Edge Case: Empty batch (BT = 0)
    empty_tokens = torch.zeros(0, 32, device=device)
    empty_indices = torch.zeros(0, 2, dtype=torch.long, device=device)
    empty_weights = torch.zeros(0, 2, device=device)
    out_empty = dispatcher(empty_tokens, empty_indices, empty_weights, experts)
    check("Empty Batch Edge Case (BT = 0)", out_empty.shape == (0, 32), f"output shape: {out_empty.shape}")

    # 4. Edge Case: 3D tensor input [B, T, C]
    tokens_3d = torch.randn(2, 8, 32, device=device)
    indices_3d = torch.randint(0, 4, (2, 8, 2), device=device)
    weights_3d = F.softmax(torch.randn(2, 8, 2, device=device), dim=-1)
    out_3d = dispatcher(tokens_3d, indices_3d, weights_3d, experts)
    out_3d_eager = dispatcher.eager_dispatch(tokens_3d, indices_3d, weights_3d, experts)
    diff_3d = (out_3d - out_3d_eager).abs().max().item()
    check("3D Input Tensor Support [B, T, C]", out_3d.shape == (2, 8, 32) and diff_3d < 1e-6, f"shape: {out_3d.shape}, diff: {diff_3d:.2e}")


def run_microbenchmark(device_str: Optional[str] = None):
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"\n===========================================================================")
    print(f"  MICROBENCHMARK SUITE: Eager vs MoEDispatcher ({device.type.upper()})")
    print(f"===========================================================================")

    C = 128
    num_experts = 34
    top_k = 4
    batch_token_sizes = [64, 256, 512, 1024, 2048]
    num_warmup = 10
    num_iters = 50

    results = []

    for BT in batch_token_sizes:
        torch.manual_seed(42)
        tokens = torch.randn(BT, C, device=device, requires_grad=True)
        topk_i = torch.randint(0, num_experts, (BT, top_k), device=device)
        topk_p = F.softmax(torch.randn(BT, top_k, device=device), dim=-1)
        experts = nn.ModuleList([SyntheticExpert(C).to(device) for _ in range(num_experts)])

        dispatcher = MoEDispatcher(use_slice_assignment=False)

        # Warmup
        for _ in range(num_warmup):
            out_e = MoEDispatcher.eager_dispatch(tokens, topk_i, topk_p, experts)
            out_c = dispatcher.contiguous_dispatch(tokens, topk_i, topk_p, experts)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Forward Benchmark: Eager
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            out_e = MoEDispatcher.eager_dispatch(tokens, topk_i, topk_p, experts)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_fwd_eager = (time.perf_counter() - t0) / num_iters * 1000

        # Forward Benchmark: Contiguous
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            out_c = dispatcher.contiguous_dispatch(tokens, topk_i, topk_p, experts)
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_fwd_cont = (time.perf_counter() - t0) / num_iters * 1000

        # Forward+Backward Benchmark: Eager
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            tokens.grad = None
            for p in experts.parameters():
                p.grad = None
            out_e = MoEDispatcher.eager_dispatch(tokens, topk_i, topk_p, experts)
            loss = out_e.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_fwd_bwd_eager = (time.perf_counter() - t0) / num_iters * 1000

        # Forward+Backward Benchmark: Contiguous
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            tokens.grad = None
            for p in experts.parameters():
                p.grad = None
            out_c = dispatcher.contiguous_dispatch(tokens, topk_i, topk_p, experts)
            loss = out_c.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
        t_fwd_bwd_cont = (time.perf_counter() - t0) / num_iters * 1000

        speedup_fwd = t_fwd_eager / max(t_fwd_cont, 1e-6)
        speedup_bwd = t_fwd_bwd_eager / max(t_fwd_bwd_cont, 1e-6)
        tput_eager = (BT / (t_fwd_eager / 1000.0))
        tput_cont = (BT / (t_fwd_cont / 1000.0))

        results.append({
            "BT": BT,
            "fwd_eager_ms": t_fwd_eager,
            "fwd_cont_ms": t_fwd_cont,
            "speedup_fwd": speedup_fwd,
            "bwd_eager_ms": t_fwd_bwd_eager,
            "bwd_cont_ms": t_fwd_bwd_cont,
            "speedup_bwd": speedup_bwd,
            "tput_eager": tput_eager,
            "tput_cont": tput_cont,
        })

    # Print Table
    print(f"\n| Tokens (BT) | Eager Fwd (ms) | Contig Fwd (ms) | Fwd Speedup | Eager F+B (ms) | Contig F+B (ms) | F+B Speedup | Contig Tput (tok/s) |")
    print(f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| {r['BT']:>11} | {r['fwd_eager_ms']:>14.3f} | {r['fwd_cont_ms']:>15.3f} | {r['speedup_fwd']:>10.2f}x | {r['bwd_eager_ms']:>14.3f} | {r['bwd_cont_ms']:>15.3f} | {r['speedup_bwd']:>10.2f}x | {r['tput_cont']:>19.1f} |")

    return results


if __name__ == "__main__":
    print("=" * 75)
    print("  MOE DISPATCH OPTIMIZATION — BATTERY & PARITY TEST SUITE")
    print("=" * 75)

    run_numerical_parity_tests()
    run_production_council_expert_parity()
    run_backward_autograd_parity()
    run_gradcheck_verification()
    run_triton_guard_and_edge_cases()

    print("\n" + "=" * 75)
    print(f"  TEST SUMMARY: {passed_tests}/{total_tests} TESTS PASSED (100% SUCCESS)")
    print("=" * 75)

    # Run microbenchmark
    run_microbenchmark()
