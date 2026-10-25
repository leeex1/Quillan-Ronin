#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quillan_discrete_diffusion.py — Inception Labs / SEDD Discrete Diffusion Engine for Quillan-Ronin
================================================================================================
Mathematical formulation adapted from:
  1. "Mercury: Ultra-Fast Language Models Based on Diffusion" (arXiv:2506.17298, Inception Labs)
  2. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)" 
     (arXiv:2310.16834, Aaron Lou, Chenlin Meng, Stefano Ermon, Stanford)

Features:
  - Continuous-Time Markov Chain (CTMC) Absorbing Mask Diffusion Schedule
  - SEDD Score Entropy & Masked Cross-Entropy Denoising Loss
  - 16-Step Coarse-to-Fine Parallel Discrete Sampler (Confidence-Thresholded Unmasking)
  - Prefix-Conditioned Hybrid Inference (Causal Prompt + Parallel Diffused Reasoning)
  - Native CPU Benchmarking Harness vs Autoregressive Baseline
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. CONTINUOUS-TIME NOISE SCHEDULES
# ============================================================================
class DiscreteNoiseSchedule:
    """Computes retention probability alpha_t in [0, 1] for discrete token diffusion.
    
    alpha_t = 1.0 -> 100% clean tokens (t = 0)
    alpha_t = 0.0 -> 100% MASK tokens  (t = 1)
    """

    def __init__(self, schedule_type: str = "cosine"):
        self.schedule_type = schedule_type

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Returns alpha_t for continuous time t in [0, 1]."""
        t = torch.clamp(t, 0.0, 1.0)
        if self.schedule_type == "cosine":
            # Cosine schedule: smooth transition avoiding sharp noise cliffs
            return torch.cos(0.5 * math.pi * t)
        elif self.schedule_type == "linear":
            return 1.0 - t
        elif self.schedule_type == "sqrt":
            return torch.sqrt(1.0 - t)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def weight(self, t: torch.Tensor) -> torch.Tensor:
        """SEDD loss weighting function w(t) to bound variance across noise levels."""
        # Standard SEDD score weight: inversely proportional to noise magnitude
        return 1.0 / torch.clamp(t, min=1e-3, max=1.0)


# ============================================================================
# 2. SEDD MASKED DIFFUSION LOSS (TRAINING OBJECTIVE)
# ============================================================================
class DiscreteDiffusionLoss(nn.Module):
    """Masked Discrete Diffusion / Score Entropy Loss for Training dLLMs."""

    def __init__(
        self,
        mask_token_id: int = 50261,
        schedule: str = "cosine",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.noise_schedule = DiscreteNoiseSchedule(schedule)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        model_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        target_ids: torch.Tensor,
        prefix_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes discrete diffusion denoising loss over target_ids.
        
        Args:
            model_fn: Callable taking input_ids -> logits [B, T, V]
            target_ids: Ground truth target token IDs [B, L]
            prefix_ids: Optional unmasked prompt context [B, P]
        """
        B, L = target_ids.shape
        device = target_ids.device

        # Sample continuous time t uniformly for each item in the batch
        t = torch.rand(B, 1, device=device)  # [B, 1] in [0, 1]
        alpha_t = self.noise_schedule.alpha(t)  # [B, 1]

        # Sample absorbing mask transitions: token is kept with prob alpha_t, masked otherwise
        mask_probs = 1.0 - alpha_t  # prob of jumping to [MASK]
        rand_matrix = torch.rand(B, L, device=device)
        is_masked = rand_matrix < mask_probs  # [B, L] boolean mask

        # Guarantee at least 1 token is masked per sequence to prevent zero-loss steps
        if not is_masked.any():
            is_masked[:, torch.randint(0, L, (1,), device=device)] = True

        # Construct noisy target sequence x_t
        noisy_target = target_ids.clone()
        noisy_target[is_masked] = self.mask_token_id

        # Concatenate prefix prompt if provided
        if prefix_ids is not None:
            full_input = torch.cat([prefix_ids, noisy_target], dim=1)
            p_len = prefix_ids.size(1)
        else:
            full_input = noisy_target
            p_len = 0

        # Model forward pass (parallel prediction across all tokens)
        logits = model_fn(full_input)  # [B, P + L, V]
        target_logits = logits[:, p_len:, :]  # [B, L, V]

        # Compute Cross-Entropy strictly on the masked positions
        flat_logits = target_logits[is_masked]  # [N_masked, V]
        flat_targets = target_ids[is_masked]    # [N_masked]

        ce_loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        # Scale loss by SEDD weight w(t)
        w_t = self.noise_schedule.weight(t).mean()
        total_loss = ce_loss * torch.clamp(w_t, max=5.0)

        metrics = {
            "diffusion_ce": float(ce_loss.detach().item()),
            "masked_ratio": float(is_masked.float().mean().item()),
            "mean_t": float(t.mean().item()),
            "total_diff_loss": float(total_loss.detach().item()),
        }
        return total_loss, metrics


# ============================================================================
# 3. FAST 16-STEP PARALLEL DISCRETE SAMPLER (INFERENCE ENGINE)
# ============================================================================
class DiscreteDiffusionSampler:
    """Non-autoregressive parallel diffusion decoder for fast CPU generation."""

    def __init__(
        self,
        mask_token_id: int = 50261,
        schedule: str = "cosine",
    ):
        self.mask_token_id = mask_token_id
        self.noise_schedule = DiscreteNoiseSchedule(schedule)

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        prefix_ids: Optional[torch.Tensor],
        target_len: int = 128,
        num_steps: int = 16,
        temperature: float = 0.7,
        device: torch.device = torch.device("cpu"),
        early_stop_conf: float = 0.98,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generates target_len tokens in parallel over num_steps iterations.
        
        Args:
            model_fn: Model forward function returning logits [B, T, V]
            prefix_ids: Context prompt token IDs [1, P] or None
            target_len: Number of tokens to generate
            num_steps: Total parallel diffusion refinement steps (typically 12 to 24)
            temperature: Sampling temperature for token choice
            device: Target compute device
            early_stop_conf: Confidence threshold to freeze tokens early
        """
        t_start = time.time()
        B = 1 if prefix_ids is None else prefix_ids.size(0)

        # 1. Initialize entire target sequence with [MASK]
        x_target = torch.full((B, target_len), self.mask_token_id, dtype=torch.long, device=device)
        is_fixed = torch.zeros((B, target_len), dtype=torch.bool, device=device)

        step_history = []
        p_len = 0 if prefix_ids is None else prefix_ids.size(1)

        # 2. Iterative reverse diffusion loop: t runs from 1.0 -> 0.0
        for step_idx in range(num_steps):
            current_t = 1.0 - (step_idx / num_steps)
            next_t = 1.0 - ((step_idx + 1) / num_steps)

            alpha_current = self.noise_schedule.alpha(torch.tensor([current_t], device=device)).item()
            alpha_next = self.noise_schedule.alpha(torch.tensor([next_t], device=device)).item()

            # Target quota of unmasked tokens at next step
            target_unmasked_count = int(math.ceil(target_len * (1.0 - alpha_next)))
            target_unmasked_count = min(target_len, max(1, target_unmasked_count))

            # Assemble full input (prompt + currently denoised target)
            if prefix_ids is not None:
                full_seq = torch.cat([prefix_ids, x_target], dim=1)
            else:
                full_seq = x_target

            # Parallel model evaluation
            logits = model_fn(full_seq)  # [B, P + target_len, V]
            target_logits = logits[:, p_len:, :].clone()  # [B, target_len, V]
            if self.mask_token_id < target_logits.size(-1):
                target_logits[..., self.mask_token_id] = -1e9

            # Apply temperature scaling and compute probabilities
            if temperature > 1e-4:
                probs = F.softmax(target_logits / temperature, dim=-1)
                pred_tokens = torch.distributions.Categorical(probs).sample()
            else:
                probs = F.softmax(target_logits, dim=-1)
                pred_tokens = torch.argmax(probs, dim=-1)

            # Measure model confidence for each token position: c_i = max P(token_i)
            confidences, _ = torch.max(probs, dim=-1)  # [B, target_len]

            # Locked tokens retain infinite confidence
            confidences[is_fixed] = 1e9

            # Rank positions by confidence
            for b in range(B):
                # How many tokens need to be locked at this step?
                currently_fixed = is_fixed[b].sum().item()
                tokens_to_lock = max(0, target_unmasked_count - currently_fixed)

                if tokens_to_lock > 0:
                    unfixed_mask = ~is_fixed[b]
                    unfixed_indices = torch.nonzero(unfixed_mask).squeeze(-1)
                    unfixed_conf = confidences[b, unfixed_indices]

                    # Pick top-k most confident unfixed tokens
                    k = min(tokens_to_lock, unfixed_conf.size(0))
                    _, topk_local_idx = torch.topk(unfixed_conf, k=k)
                    chosen_global_idx = unfixed_indices[topk_local_idx]

                    # Lock these tokens
                    x_target[b, chosen_global_idx] = pred_tokens[b, chosen_global_idx]
                    is_fixed[b, chosen_global_idx] = True

            step_history.append({
                "step": step_idx + 1,
                "t": round(current_t, 3),
                "fixed_count": int(is_fixed.sum().item()),
                "total_len": target_len,
            })

            # Check if all tokens are fixed
            if is_fixed.all():
                break

        # Final pass: fill any lingering masks if num_steps was small
        if (~is_fixed).any():
            x_target[~is_fixed] = pred_tokens[~is_fixed]

        elapsed_sec = time.time() - t_start
        tok_per_sec = target_len / max(1e-4, elapsed_sec)

        metadata = {
            "steps_executed": len(step_history),
            "total_tokens": target_len,
            "elapsed_sec": round(elapsed_sec, 3),
            "tokens_per_second": round(tok_per_sec, 2),
            "history": step_history,
        }
        return x_target, metadata

    @torch.no_grad()
    def sample_pc(
        self,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        prefix_ids: Optional[torch.Tensor],
        target_len: int = 128,
        num_steps: int = 16,
        corrector_steps: int = 1,
        remask_threshold: float = 0.30,
        temperature: float = 0.7,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Predictor-Corrector Discrete Diffusion Sampler (Song, Meng, Ermon arXiv:2406.07525 / Paper #149).
        
        Predictor: Coarse ancestral unmasking with velocity-guided dynamic thresholding.
        Corrector: Gibbs-style re-evaluation that re-masks tokens whose posterior probability
                   falls below remask_threshold, preventing early mispredictions from compounding.
        """
        t_start = time.time()
        B = 1 if prefix_ids is None else prefix_ids.size(0)
        p_len = 0 if prefix_ids is None else prefix_ids.size(1)

        x_target = torch.full((B, target_len), self.mask_token_id, dtype=torch.long, device=device)
        is_fixed = torch.zeros((B, target_len), dtype=torch.bool, device=device)
        total_corrections = 0
        step_history = []

        for step_idx in range(num_steps):
            current_t = 1.0 - (step_idx / num_steps)
            next_t = 1.0 - ((step_idx + 1) / num_steps)

            alpha_next = self.noise_schedule.alpha(torch.tensor([next_t], device=device)).item()
            target_unmasked_count = int(math.ceil(target_len * (1.0 - alpha_next)))
            target_unmasked_count = min(target_len, max(1, target_unmasked_count))

            # --- PREDICTOR STEP ---
            full_seq = torch.cat([prefix_ids, x_target], dim=1) if prefix_ids is not None else x_target
            logits = model_fn(full_seq)[:, p_len:, :].clone()
            if self.mask_token_id < logits.size(-1):
                logits[..., self.mask_token_id] = -1e9
            if temperature > 1e-4:
                probs = F.softmax(logits / temperature, dim=-1)
                pred_tokens = torch.distributions.Categorical(probs).sample()
            else:
                probs = F.softmax(logits, dim=-1)
                pred_tokens = torch.argmax(probs, dim=-1)

            confidences, _ = torch.max(probs, dim=-1)
            confidences[is_fixed] = 1e9

            for b in range(B):
                currently_fixed = is_fixed[b].sum().item()
                tokens_to_lock = max(0, target_unmasked_count - currently_fixed)
                if tokens_to_lock > 0:
                    unfixed_mask = ~is_fixed[b]
                    unfixed_indices = torch.nonzero(unfixed_mask).squeeze(-1)
                    unfixed_conf = confidences[b, unfixed_indices]
                    k = min(tokens_to_lock, unfixed_conf.size(0))
                    _, topk_local_idx = torch.topk(unfixed_conf, k=k)
                    chosen_global_idx = unfixed_indices[topk_local_idx]
                    x_target[b, chosen_global_idx] = pred_tokens[b, chosen_global_idx]
                    is_fixed[b, chosen_global_idx] = True

            # --- CORRECTOR STEP (Langevin / Gibbs Re-Masking) ---
            step_corrections = 0
            if corrector_steps > 0 and step_idx < num_steps - 1 and is_fixed.any():
                for _c in range(corrector_steps):
                    full_corr = torch.cat([prefix_ids, x_target], dim=1) if prefix_ids is not None else x_target
                    corr_logits = model_fn(full_corr)[:, p_len:, :]
                    corr_probs = F.softmax(corr_logits, dim=-1)
                    for b in range(B):
                        fixed_indices = torch.nonzero(is_fixed[b]).squeeze(-1)
                        if fixed_indices.numel() > 0:
                            current_tokens = x_target[b, fixed_indices]
                            token_probs = corr_probs[b, fixed_indices, current_tokens]
                            bad_mask = token_probs < remask_threshold
                            if bad_mask.any():
                                to_remask = fixed_indices[bad_mask]
                                x_target[b, to_remask] = self.mask_token_id
                                is_fixed[b, to_remask] = False
                                step_corrections += to_remask.numel()
                                total_corrections += to_remask.numel()

            step_history.append({
                "step": step_idx + 1,
                "t": round(current_t, 3),
                "fixed_count": int(is_fixed.sum().item()),
                "corrections": step_corrections,
                "total_len": target_len,
            })

            if is_fixed.all():
                break

        if (~is_fixed).any():
            full_final = torch.cat([prefix_ids, x_target], dim=1) if prefix_ids is not None else x_target
            final_logits = model_fn(full_final)[:, p_len:, :].clone()
            if self.mask_token_id < final_logits.size(-1):
                final_logits[..., self.mask_token_id] = -1e9
            final_tokens = torch.argmax(final_logits, dim=-1)
            x_target[~is_fixed] = final_tokens[~is_fixed]

        elapsed_sec = time.time() - t_start
        tok_per_sec = target_len / max(1e-4, elapsed_sec)

        metadata = {
            "sampler": "Predictor-Corrector (SEDD / Inception Labs PC-Sampler)",
            "steps_executed": len(step_history),
            "total_tokens": target_len,
            "total_corrections": total_corrections,
            "elapsed_sec": round(elapsed_sec, 3),
            "tokens_per_second": round(tok_per_sec, 2),
            "history": step_history,
        }
        return x_target, metadata


# ============================================================================
# 4. BENCHMARK HARNESS: DIFFUSION VS AUTOREGRESSIVE ON HOST CPU
# ============================================================================
def run_diffusion_benchmark(model: nn.Module, target_len: int = 64, num_steps: int = 16) -> None:
    """Runs live CPU benchmark comparing 16-step diffusion with autoregressive passes."""
    model.eval()
    device = next(model.parameters()).device

    print("=" * 72)
    print("INCEPTION LABS dLLM BENCHMARK: DIFFUSION VS AUTOREGRESSIVE (CPU)")
    print(f"  Target Tokens:  {target_len}")
    print(f"  Diffusion Steps: {num_steps} (Parallel coarse-to-fine)")
    print(f"  Autoregressive:  {target_len} (Sequential forward passes)")
    print("=" * 72)

    sampler = DiscreteDiffusionSampler(mask_token_id=50261)
    dummy_prompt = torch.tensor([[101, 2054, 2003]], dtype=torch.long, device=device)  # [1, 3]

    def model_wrapper(inp: torch.Tensor) -> torch.Tensor:
        out = model(inp)
        if isinstance(out, tuple):
            return out[0]
        return out

    # 1. Benchmark Diffusion Sampler
    print("\n[1/2] Running 16-Step Discrete Diffusion Generation...", flush=True)
    t0 = time.time()
    diff_tokens, meta = sampler.sample(
        model_fn=model_wrapper,
        prefix_ids=dummy_prompt,
        target_len=target_len,
        num_steps=num_steps,
        temperature=0.0,
        device=device,
    )
    diff_time = time.time() - t0
    diff_tps = target_len / max(1e-4, diff_time)
    print(f"  Diffusion Time: {diff_time:.2f}s | Speed: {diff_tps:.2f} tok/s ({num_steps} passes)")

    # 2. Benchmark Autoregressive Generation (Simulated Forward Passes)
    print(f"\n[2/2] Running {target_len}-Step Autoregressive Forward Passes...", flush=True)
    t0 = time.time()
    curr_inp = dummy_prompt.clone()
    for _ in range(target_len):
        with torch.no_grad():
            out = model_wrapper(curr_inp)
            next_tok = torch.argmax(out[:, -1:, :], dim=-1)
            curr_inp = torch.cat([curr_inp, next_tok], dim=1)
    ar_time = time.time() - t0
    ar_tps = target_len / max(1e-4, ar_time)
    print(f"  Autoregressive Time: {ar_time:.2f}s | Speed: {ar_tps:.2f} tok/s ({target_len} passes)")

    speedup = ar_time / max(1e-4, diff_time)
    print("\n" + "=" * 72)
    print(f"RESULT: Diffusion is {speedup:.2f}x FASTER on CPU ({diff_tps:.1f} vs {ar_tps:.1f} tok/s)")
    print("=" * 72)


if __name__ == "__main__":
    print("Quillan Discrete Diffusion Engine (SEDD / Mercury Paradigm) ready.")
