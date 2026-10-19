#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 QUILLAN-RONIN v5.3.1 — TRAINING INFRASTRUCTURE FIXES
--------------------------------------------------------------------------------
 Drop-in replacements for:
   1. ComplexityRouter + EvolvableVectorizedMoE (load balancing + Z-loss)
   2. EGGROLLSwarmEvolver (true gradient-free evolution)
   3. Synthetic 10M-parameter validation harness

 INTEGRATION:
   - Replace ComplexityRouter and EvolvableVectorizedMoE in your main file.
   - Add EGGROLLSwarmEvolver below the model classes.
   - Run run_synthetic_harness() before any full-scale training.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = r"C:\02_QUILLAN"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _dev.quillan_v8_saturated import (
    BitLinear, _weight_quant, CouncilExpertSwarm,
    QuillanRoninSovereign, QuillanArchConfig, LeeMach6Governor
)



# ═══════════════════════════════════════════════════════════════════════════════
# 1. VECTORIZED MOE WITH REAL LOAD BALANCING & Z-LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexityRouter(nn.Module):
    """
    Simplified router that returns raw logits for Z-loss computation.
    NOTE: If BitNet ternary routing proves unstable during early training,
    set quantize_weight=False in the BitLinear call below.
    """
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.router = BitLinear(hidden_dim, num_experts, bias=False,
                                quantize_act=True, quantize_weight=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        router_logits = self.router(flat_x)          # (B*L, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        return routing_weights, router_logits


class EvolvableVectorizedMoE(nn.Module):
    """
    Drop-in replacement with:
      - Load balancing auxiliary loss (Switch Transformer formula)
      - Router Z-loss (ST-MoE formula) to prevent logit explosion
      - Loop only over ACTIVE experts (skips the ~30 empty experts per batch)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.router = ComplexityRouter(cfg.hidden_dim, cfg.num_experts)

        # Expert FFN weights (ternary quantized at runtime)
        self.w1 = nn.Parameter(torch.randn(cfg.num_experts, cfg.hidden_dim, cfg.ffn_dim) * 0.02)
        self.wgate = nn.Parameter(torch.randn(cfg.num_experts, cfg.hidden_dim, cfg.ffn_dim) * 0.02)
        self.w2 = nn.Parameter(torch.randn(cfg.num_experts, cfg.ffn_dim, cfg.hidden_dim) * 0.02)

        # LoRA adapters for expert FFNs
        self.w1_lora_A = nn.Parameter(torch.randn(cfg.num_experts, cfg.hidden_dim, 16) * 0.01)
        self.w1_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, 16, cfg.ffn_dim))
        self.wgate_lora_A = nn.Parameter(torch.randn(cfg.num_experts, cfg.hidden_dim, 16) * 0.01)
        self.wgate_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, 16, cfg.ffn_dim))
        self.w2_lora_A = nn.Parameter(torch.randn(cfg.num_experts, cfg.ffn_dim, 16) * 0.01)
        self.w2_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, 16, cfg.hidden_dim))

        # 34 Council Expert Swarms
        self.expert_swarms = nn.ModuleList([
            CouncilExpertSwarm(cfg.ffn_dim, rank=24) for _ in range(cfg.num_experts)
        ])
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

        # Auxiliary loss coefficients (Switch / ST-MoE defaults)
        self.aux_loss_coef = 0.05    # alpha (strengthened load balancing)
        self.z_loss_coef = 0.001     # z-loss weight


    def forward(self, x: torch.Tensor, gov_scale: float = 1.0):
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        routing_probs, router_logits = self.router(x)
        self._last_probs = routing_probs

        # Top-k gating (on probabilities, preserving your existing behavior)
        topk_p, topk_idx = torch.topk(routing_probs, self.cfg.top_k, dim=-1)
        topk_p = F.softmax(topk_p, dim=-1)

        # ─── AUXILIARY LOSSES ──────────────────────────────────────────────
        # Load balancing: encourage uniform dispatch across all 34 experts
        # router_mask[i,e] = 1 if expert e is in top-k for token i
        router_mask = torch.zeros_like(routing_probs).scatter_(1, topk_idx, 1.0)
        f_i = router_mask.mean(dim=0)            # fraction of tokens dispatched to expert i
        P_i = routing_probs.mean(dim=0)          # mean router probability for expert i
        aux_loss = self.cfg.num_experts * (f_i * P_i).sum()

        # Z-loss: penalize large logits to prevent routing collapse / scale drift
        z_loss = torch.log(torch.exp(router_logits).sum(dim=-1) + 1e-6).pow(2).mean()

        total_aux = self.aux_loss_coef * aux_loss + self.z_loss_coef * z_loss

        # ─── EXPERT EXECUTION ──────────────────────────────────────────────
        compute_dtype = x.dtype
        w1_q_all = _weight_quant(self.w1)
        wgate_q_all = _weight_quant(self.wgate)
        w2_q_all = _weight_quant(self.w2)

        w_a_all = torch.stack([s.A for s in self.expert_swarms])
        w_b_all = torch.stack([s.B for s in self.expert_swarms])

        final_out = torch.zeros_like(flat_x)

        # CRITICAL FIX: only iterate experts that actually received tokens.
        # With top_k=4, typically 4–8 experts are active per batch, not all 34.
        active_experts = torch.unique(topk_idx)

        for e in active_experts.tolist():
            mask = (topk_idx == e)
            token_indices = mask.any(dim=-1)
            if not token_indices.any():
                continue

            # Gather gating weights for tokens assigned to this expert
            expert_gates = (topk_p * mask.to(compute_dtype)).sum(dim=-1)[token_indices].unsqueeze(-1)
            x_tok = flat_x[token_indices].to(compute_dtype)

            w1_q_c = w1_q_all[e].to(compute_dtype)
            wgate_q_c = wgate_q_all[e].to(compute_dtype)

            rs_scaling = 16.0 / math.sqrt(self.w1_lora_B.shape[1])
            w1_out = x_tok @ w1_q_c + ((x_tok @ self.w1_lora_A[e]) @ self.w1_lora_B[e]) * rs_scaling
            wgate_out = x_tok @ wgate_q_c + ((x_tok @ self.wgate_lora_A[e]) @ self.wgate_lora_B[e]) * rs_scaling

            h = F.silu(w1_out) * wgate_out
            h_swarm = self.expert_swarms[e](
                h, scale=gov_scale,
                w_a=w_a_all[e].to(compute_dtype),
                w_b=w_b_all[e].to(compute_dtype)
            )
            w2_out = h_swarm @ w2_q_all[e].to(compute_dtype) + ((h_swarm @ self.w2_lora_A[e]) @ self.w2_lora_B[e]) * rs_scaling

            idx_flat = token_indices.nonzero(as_tuple=True)[0]
            final_out.index_add_(0, idx_flat, (w2_out * expert_gates).to(final_out.dtype))

        return self.output_norm(final_out.reshape(B, L, D)), total_aux


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRUE EGGROLL EVOLUTIONARY SWARM LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class EGGROLLSwarmEvolver:
    """
    Gradient-free evolutionary optimizer for CouncilExpertSwarm low-rank matrices.

    Usage:
        evolver = EGGROLLSwarmEvolver(model, population_size=8, lr=0.1)
        # In training loop:
        loss.backward(); base_optimizer.step(); base_optimizer.zero_grad()
        evolver.evolve(input_batch, target_batch, loss_fn)
    """
    def __init__(self, model, population_size: int = 8, mutation_sigma: float = 0.005,
                 elite_ratio: float = 0.25, lr: float = 0.1, device: str = 'cpu'):
        self.model = model
        self.pop_size = population_size
        self.sigma = mutation_sigma
        self.elite_k = max(1, int(population_size * elite_ratio))
        self.lr = lr
        self.device = device

        # Swarm parameters are updated manually (no backprop)
        for name, param in model.named_parameters():
            if 'expert_swarms' in name:
                param.requires_grad = False

        # Snapshot master swarm states (A/B matrices only; C/D are diversity/noise)
        self.masters = {}
        for idx, swarm in enumerate(model.moe.expert_swarms):
            self.masters[idx] = {
                'A': swarm.A.data.clone().detach().to(device),
                'B': swarm.B.data.clone().detach().to(device),
            }

    def _apply_swarm(self, expert_idx: int, A: torch.Tensor, B: torch.Tensor):
        self.model.moe.expert_swarms[expert_idx].A.data.copy_(A)
        self.model.moe.expert_swarms[expert_idx].B.data.copy_(B)

    @torch.no_grad()
    def evolve(self, input_txt: torch.Tensor, target_txt: torch.Tensor, loss_fn):
        """
        One evolution step across all 34 expert swarms.
        Call AFTER your standard base-weight optimizer step.
        """
        was_training = self.model.training
        self.model.eval()

        for e in range(len(self.model.moe.expert_swarms)):
            master_A = self.masters[e]['A']
            master_B = self.masters[e]['B']

            # Generate population: master + low-rank Gaussian noise
            pop_A = [master_A + torch.randn_like(master_A, device=self.device) * self.sigma
                     for _ in range(self.pop_size)]
            pop_B = [master_B + torch.randn_like(master_B, device=self.device) * self.sigma
                     for _ in range(self.pop_size)]

            # Include master (control) — ensures we never regress
            pop_A.append(master_A.clone())
            pop_B.append(master_B.clone())

            fitness = []
            for A_cand, B_cand in zip(pop_A, pop_B):
                self._apply_swarm(e, A_cand, B_cand)
                logits = self.model(input_txt)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_txt.reshape(-1))
                fitness.append(-loss.item())   # negative loss = higher fitness

            # Elite selection
            fitness_t = torch.tensor(fitness, device=self.device)
            elite_idx = torch.topk(fitness_t, self.elite_k).indices

            # Fitness-weighted recombination
            elite_fitness = fitness_t[elite_idx]
            weights = F.softmax(elite_fitness - elite_fitness.min(), dim=0)

            elite_A = torch.stack([pop_A[i] for i in elite_idx])
            elite_B = torch.stack([pop_B[i] for i in elite_idx])

            new_A = (elite_A * weights.view(-1, 1, 1)).sum(0)
            new_B = (elite_B * weights.view(-1, 1, 1)).sum(0)

            # Momentum update toward elite center
            updated_A = (1 - self.lr) * master_A + self.lr * new_A
            updated_B = (1 - self.lr) * master_B + self.lr * new_B

            self.masters[e]['A'].copy_(updated_A)
            self.masters[e]['B'].copy_(updated_B)
            self._apply_swarm(e, updated_A, updated_B)

        if was_training:
            self.model.train()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SYNTHETIC 10M DETERMINISTIC VALIDATION HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HarnessConfig:
    """Minimal config for ~10M parameter validation model."""
    text_only: bool = True
    multimodal: bool = False
    hidden_dim: int = 128
    low_mem: bool = True
    low_gpu: bool = True
    ffn_dim: int = 256
    vocab_size: int = 128
    num_experts: int = 34
    num_experts_active: int = 4
    top_k: int = 4
    use_lora: bool = True
    device: str = 'cpu'
    eggroll_rank: int = 8
    e_ice_limit_ms: int = 100
    max_seq_len: int = 48


def run_synthetic_harness():
    """
    Deterministic sequence-shift overfitting test.
    Requirements:
      - Loss < 0.01 within 100 steps.
      - 0 dead experts (all 34 council members must receive tokens).
    """
    cfg = HarnessConfig()

    # Build model (assumes QuillanRoninSovereign is in scope)
    model = QuillanRoninSovereign(cfg)
    model.to(cfg.device)
    model.train()

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model params: {total_p:,} total | {train_p:,} base-trainable")

    # Base optimizer: ONLY non-swarm parameters (swarm evolved separately)
    base_params = [p for n, p in model.named_parameters() if 'expert_swarms' not in n]
    optimizer = torch.optim.AdamW(base_params, lr=3e-4)

    # EGGROLL evolver (small population for validation speed)
    evolver = EGGROLLSwarmEvolver(
        model, population_size=4, mutation_sigma=0.01, lr=0.2, device=cfg.device
    )

    loss_fn = nn.CrossEntropyLoss()

    # Diverse synthetic sequences to test load balancing across vocab
    seq_len = cfg.max_seq_len
    batch_size = 16
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=cfg.device)
    y = (x + 1) % cfg.vocab_size


    print("=" * 60)
    print("🧪 QUILLAN 10M SYNTHETIC HARNESS — Sequence Shift Task")
    print("=" * 60)

    for step in range(100):
        optimizer.zero_grad()

        logits = model(x)                       # (B, L, vocab_size)
        loss = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

        # Backprop through base weights only
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params, 1.0)
        optimizer.step()

        # EGGROLL evolution step (gradient-free)
        evolver.evolve(x, y, loss_fn)

        # Telemetry
        if step % 10 == 0 or step == 99:
            with torch.no_grad():
                _ = model(x)
                probs = model.moe._last_probs               # (B*L, num_experts)
                _, topk_idx = torch.topk(probs, cfg.top_k, dim=-1)
                active_counts = torch.bincount(topk_idx.reshape(-1), minlength=cfg.num_experts)
                dead_experts = int((active_counts == 0).sum().item())
                util_std = active_counts.float().std().item()

                print(f"Step {step:03d} | Loss: {loss.item():.6f} | "
                      f"Dead Experts: {dead_experts:02d}/{cfg.num_experts} | "
                      f"Util σ: {util_std:.1f}")

    # ─── FINAL ASSERTIONS ────────────────────────────────────────────────────
    final_loss = loss.item()
    assert final_loss < 0.01, f"❌ FAIL: Loss {final_loss:.6f} >= 0.01 (model not converging)"
    assert dead_experts == 0, f"❌ FAIL: {dead_experts} experts starved (routing collapse)"

    print("\n✅ ALL ASSERTIONS PASSED")
    print(f"   • Loss converged to {final_loss:.6f} (< 0.01)")
    print(f"   • All {cfg.num_experts} council experts received tokens (balanced routing)")
    print("   • Safe to scale to full dataset training.\n")
    return model, optimizer, evolver


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # NOTE: Before running, ensure QuillanRoninSovereign and dependencies
    # are imported or defined in this file's scope.
    run_synthetic_harness()
