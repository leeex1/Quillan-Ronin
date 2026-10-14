r"""
DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
(Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo - DeepSeek-AI, arXiv:2402.03300v3).

Mathematical Foundations & Core Algorithmic Invariants:
--------------------------------------------------------
1. Group Relative Policy Optimization (GRPO) Objective (Eq. 3):
   PPO maintains a value/critic model V_\psi of comparable size to the policy \pi_\theta,
   consuming ~50% extra memory and complicating token-level baseline calibration.
   GRPO eliminates the critic entirely by sampling a group of G candidate outputs
   {o_1, ..., o_G} for each question q from the old policy \pi_{\theta_{old}}, and
   optimizes:
       J_GRPO(\theta) = E_{q, {o_i}_{i=1}^G} [ (1 / G) * sum_{i=1}^G (1 / |o_i|) * sum_{t=1}^{|o_i|}
           ( min( r_{i,t}(\theta) \hat{A}_{i,t}, clip(r_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{i,t} )
             - \beta D_{KL}(\pi_\theta || \pi_{ref}) ) ]
   where r_{i,t}(\theta) = \pi_\theta(o_{i,t}) / \pi_{\theta_{old}}(o_{i,t}).

2. Group-Relative Advantage Estimation:
   (a) Outcome Supervision (Section 4.1.2):
       Rewards r = {r_1, ..., r_G} are evaluated at sequence completion.
       Normalized across group:
           \hat{A}_{i,t} = \tilde{r}_i = (r_i - mean(r)) / (std(r) + \epsilon_{adv})
       Assigned uniformly to all tokens t in response o_i.

   (b) Process Supervision (Section 4.1.3):
       Rewards given at reasoning step boundaries index(1), ..., index(K_i).
       Normalized: \tilde{r}_i^{index(j)} = (r_i^{index(j)} - mean(R)) / (std(R) + \epsilon_{adv}).
       Token advantage is cumulative future return from active step forward:
           \hat{A}_{i,t} = sum_{index(j) >= t} \tilde{r}_i^{index(j)}

3. Schulman Unbiased, Non-Negative KL Divergence Estimator (Eq. 4):
       D_{KL}(\pi_\theta || \pi_{ref}) = (\pi_{ref} / \pi_\theta) - log(\pi_{ref} / \pi_\theta) - 1
   Letting u = \pi_{ref} / \pi_\theta, f(u) = u - log(u) - 1 >= 0 for all u > 0,
   with equality if and only if u = 1.
   Guaranteed non-negative, low variance, and unbiased estimator.

4. Iterative Policy & Reference Model Advancement (Algorithm 1):
   Outer loop updates reference anchor \pi_{ref} <- \pi_\theta, preventing policy drift
   while enabling multi-stage mathematical discovery across iterations.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    """Hyperparameters for Group Relative Policy Optimization."""
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    group_size: int = 4
    adv_eps: float = 1e-8
    supervision_mode: str = "outcome"  # "outcome" or "process"
    use_sample_level_reduction: bool = True  # Canonical DeepSeekMath sample-level (1/G * 1/|o_i|)


def schulman_kl_divergence(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    r"""
    Computes Schulman (2020) unbiased, non-negative KL divergence estimator (Eq. 4):
        D_KL(\pi_\theta || \pi_{ref}) = u - log(u) - 1
    where u = \pi_{ref} / \pi_\theta = exp(ref_logprobs - logprobs).
    Guaranteed >= 0 for all values.
    """
    # log_ratio = ref_logprobs - logprobs = log(u)
    log_ratio = ref_logprobs - logprobs
    # u = exp(log_ratio)
    u = torch.exp(log_ratio)
    # D_KL = u - log(u) - 1 = exp(ref - logp) - (ref - logp) - 1
    kl = u - log_ratio - 1.0
    return torch.clamp(kl, min=0.0)


def compute_outcome_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Outcome Supervision Group-Relative Advantage (Section 4.1.2):
        \tilde{r}_i = (r_i - mean(r)) / (std(r) + eps)
    Shape: (G,) or (B, G) -> same shape returned.
    """
    if rewards.dim() == 1:
        mean_r = rewards.mean()
        std_r = rewards.std(unbiased=False)
        return (rewards - mean_r) / (std_r + eps)
    elif rewards.dim() == 2:
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True, unbiased=False)
        return (rewards - mean_r) / (std_r + eps)
    else:
        raise ValueError(f"Unsupported rewards dimension: {rewards.dim()}")


def compute_process_advantages(
    step_rewards: List[List[float]],
    step_end_indices: List[List[int]],
    seq_lengths: List[int],
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""
    Process Supervision Group-Relative Advantage (Section 4.1.3):
        Normalizes all step rewards across the group, then assigns each token
        the cumulative sum of future normalized step rewards:
            \hat{A}_{i,t} = sum_{index(j) >= t} \tilde{r}_i^{index(j)}
    Returns tensor of shape (G, max_seq_len).
    """
    g = len(step_rewards)
    max_len = max(seq_lengths)
    all_rewards = [r for sublist in step_rewards for r in sublist]
    if len(all_rewards) == 0:
        return torch.zeros(g, max_len)

    tensor_all = torch.tensor(all_rewards, dtype=torch.float32)
    mean_r = tensor_all.mean().item()
    std_r = tensor_all.std(unbiased=False).item() + eps

    advantages = torch.zeros(g, max_len, dtype=torch.float32)

    for i in range(g):
        steps = step_rewards[i]
        ends = step_end_indices[i]
        t_len = seq_lengths[i]

        norm_steps = [(r - mean_r) / std_r for r in steps]

        for t in range(t_len):
            # Sum normalized rewards for all steps completing at or after token t
            adv_t = sum(norm_steps[j] for j, end_idx in enumerate(ends) if end_idx >= t)
            advantages[i, t] = adv_t

    return advantages


class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization Loss Module (Eq. 3 & Eq. 4).
    Computes clipped surrogate objective with direct Schulman KL divergence regularization.
    """
    def __init__(self, config: Optional[GRPOConfig] = None):
        super().__init__()
        self.config = config or GRPOConfig()

    def forward(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logprobs: Current policy log-probs, shape (B, T) or (G, T)
            old_logprobs: Rollout policy log-probs, shape (B, T) or (G, T)
            ref_logprobs: Reference policy log-probs, shape (B, T) or (G, T)
            advantages: Token advantages, shape (B, T) or (B,) or (G,)
            mask: Binary token mask, shape (B, T)

        Returns:
            Dict containing loss, surr1, surr2, kl_div, and clip_ratio.
        """
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)  # broadcast to (B, 1)

        if mask is None:
            mask = torch.ones_like(logprobs)

        # Ratio r_{i,t} = \pi_\theta / \pi_{\theta_{old}} = exp(logprobs - old_logprobs)
        ratio = torch.exp(logprobs - old_logprobs)

        # Symmetric clipping [1 - \epsilon, 1 + \epsilon]
        clip_low = 1.0 - self.config.clip_eps
        clip_high = 1.0 + self.config.clip_eps
        clipped_ratio = torch.clamp(ratio, min=clip_low, max=clip_high)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        surrogate_loss = torch.min(surr1, surr2)

        # Schulman KL divergence (Eq. 4)
        kl = schulman_kl_divergence(logprobs, ref_logprobs)

        # Net per-token objective: surrogate - \beta * D_KL
        # We maximize J_GRPO, so gradient descent minimizes -J_GRPO
        token_objective = surrogate_loss - self.config.kl_beta * kl
        token_loss = -token_objective

        if self.config.use_sample_level_reduction:
            # Canonical GRPO: (1/G) * sum (1/|o_i|) * sum token_loss
            seq_lens = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            sample_loss = (token_loss * mask).sum(dim=1, keepdim=True) / seq_lens
            loss = sample_loss.mean()
        else:
            total_tokens = mask.sum().clamp(min=1.0)
            loss = (token_loss * mask).sum() / total_tokens

        with torch.no_grad():
            clipped = ((ratio < clip_low) | (ratio > clip_high)) & (mask > 0)
            valid_tokens = mask.sum().clamp(min=1.0)
            clip_ratio = clipped.float().sum() / valid_tokens
            mean_kl = (kl * mask).sum() / valid_tokens

        return {
            "loss": loss,
            "surrogate": (surrogate_loss * mask).sum() / valid_tokens,
            "kl_div": mean_kl,
            "clip_ratio": clip_ratio,
        }


class IterativeGRPOTrainer:
    """
    Iterative Group Relative Policy Optimization Framework (Algorithm 1).
    Manages policy updates, reference policy anchoring, and group advantage updates.
    """
    def __init__(
        self,
        policy_model: nn.Module,
        config: Optional[GRPOConfig] = None,
        lr: float = 1e-4,
    ):
        self.policy = policy_model
        self.config = config or GRPOConfig()
        self.loss_fn = GRPOLoss(self.config)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

        # Snapshot initial reference policy weights
        self.reference_weights = {
            name: param.detach().clone() for name, param in self.policy.named_parameters()
        }
        self.iteration = 0
        self.step = 0

    def sync_reference_model(self):
        r"""Advances reference model: \pi_{ref} <- \pi_\theta (Algorithm 1, line 3)."""
        self.iteration += 1
        for name, param in self.policy.named_parameters():
            self.reference_weights[name].copy_(param.detach())

    def train_step(
        self,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        old_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inner_iterations: int = 1,
    ) -> Dict[str, float]:
        r"""
        Executes inner GRPO optimization loop (\mu iterations, Algorithm 1, lines 10-11).
        """
        total_loss = 0.0
        total_kl = 0.0
        total_clip = 0.0

        for _ in range(inner_iterations):
            self.step += 1
            self.optimizer.zero_grad()

            logits = self.policy(input_tokens)
            logprobs = F.log_softmax(logits, dim=-1)
            # Gather logprobs for target tokens
            gathered_logp = logprobs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)

            res = self.loss_fn(
                logprobs=gathered_logp,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
                advantages=advantages,
                mask=mask,
            )

            loss = res["loss"]
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_kl += res["kl_div"].item()
            total_clip += res["clip_ratio"].item()

        return {
            "loss": total_loss / inner_iterations,
            "kl_div": total_kl / inner_iterations,
            "clip_ratio": total_clip / inner_iterations,
            "iteration": self.iteration,
            "step": self.step,
        }
