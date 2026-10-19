r"""
DAPO: An Open-Source LLM Reinforcement Learning System at Scale
(Hao Zhou, Mingxuan Wang, et al. - ByteDance Seed, Tsinghua AIR, SIA-Lab, HKU, arXiv:2503.14476v2).

Mathematical Foundations & Core Algorithmic Invariants:
--------------------------------------------------------
1. Decoupled Clip-Higher Strategy (Eq. 10):
   Standard PPO / GRPO clips the importance sampling ratio symmetrically:
       clip(r_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon)
   When \epsilon = 0.2 and \hat{A}_{i,t} > 0, an exploration token with low prior
   \pi_{\theta_{old}} = 0.01 is bounded to \le 0.012, severely choking exploration and
   triggering entropy collapse.
   DAPO decouples the upper and lower clipping bounds:
       clip(r_{i,t}(\theta), 1 - \epsilon_{low}, 1 + \epsilon_{high})
   where \epsilon_{high} > \epsilon_{low} (e.g. \epsilon_{low} = 0.2, \epsilon_{high} = 0.6 to 0.8).
   This raises the ceiling on low-probability exploration tokens while keeping \epsilon_{low}
   tight to prevent probability collapse to zero.

2. Dynamic Sampling with Contrastive Group Filtering (Eq. 11, Algorithm 1):
   In group relative policy optimization, if all G candidate rollouts for a prompt are
   correct (acc = 1) or all incorrect (acc = 0), reward variance is zero:
       std({R_i}_{i=1}^G) = 0  ==>  \hat{A}_{i,t} = 0
   All-zero advantages yield zero policy gradients, diluting batch gradients and inflating
   noise. DAPO enforces the strict non-triviality condition:
       0 < |{o_i | is_equivalent(a, o_i)}| < G
   Dynamic over-sampling loops until the batch is packed with active groups possessing non-zero variance.

3. Token-Level Policy Gradient Loss (Eq. 12):
   GRPO averages loss within each sample then averages across samples:
       J_GRPO = (1/G) * sum_{i=1}^G (1/|o_i|) * sum_{t=1}^{|o_i|} L_{i,t}
   This penalizes tokens in long chains of thought by 1/|o_i|, diluting reasoning gradients and
   failing to suppress repetitive gibberish.
   DAPO computes token-level reduction:
       J_DAPO = (1 / sum_{i=1}^G |o_i|) * sum_{i=1}^G sum_{t=1}^{|o_i|} min(r_{i,t} \hat{A}_i, clip(...) \hat{A}_i)
   Every token carries uniform gradient weight regardless of sample length, maintaining stable
   entropy and eliminating unhealthy response elongation.

4. Soft Overlong Reward Shaping (Eq. 13):
   Truncated samples suffer harsh punitive -1 rewards that confuse valid reasoning.
   DAPO applies a continuous linear ramp-down buffer [L_max - L_cache, L_max]:
       R_length(y) = 0                                          if |y| <= L_max - L_cache
       R_length(y) = ((L_max - L_cache) - |y|) / L_cache        if L_max - L_cache < |y| <= L_max
       R_length(y) = -1                                         if |y| > L_max
   R_total = R_accuracy(y, a) + R_length(y)

5. Reference KL Divergence Removal:
   Eliminates \beta D_{KL}(\pi_\theta || \pi_{ref}) to enable unbounded reasoning exploration.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DAPOConfig:
    """Hyperparameters for DAPO algorithm."""
    eps_low: float = 0.2
    eps_high: float = 0.6
    group_size: int = 8
    max_length: int = 4096
    cache_length: int = 512
    adv_eps: float = 1e-8
    use_token_level_loss: bool = True
    mask_overlong_truncated: bool = True


def soft_overlong_punishment(
    response_length: Union[int, torch.Tensor],
    max_length: int = 4096,
    cache_length: int = 512,
) -> Union[float, torch.Tensor]:
    """
    Computes Soft Overlong Punishment R_length(y) according to Equation 13.

    R_length(y) =
        0,                                                  |y| <= L_max - L_cache
        ((L_max - L_cache) - |y|) / L_cache,               L_max - L_cache < |y| <= L_max
        -1,                                                |y| > L_max
    """
    threshold = max_length - cache_length
    if isinstance(response_length, torch.Tensor):
        penalty = torch.zeros_like(response_length, dtype=torch.float32)
        in_buffer = (response_length > threshold) & (response_length <= max_length)
        penalty[in_buffer] = (threshold - response_length[in_buffer].float()) / float(cache_length)
        penalty[response_length > max_length] = -1.0
        return penalty
    else:
        if response_length <= threshold:
            return 0.0
        elif response_length <= max_length:
            return float((threshold - response_length) / cache_length)
        else:
            return -1.0


def compute_rule_reward(
    is_correct: Union[bool, torch.Tensor],
    response_length: Union[int, torch.Tensor],
    max_length: int = 4096,
    cache_length: int = 512,
) -> Union[float, torch.Tensor]:
    """
    Combines rule-based accuracy outcome reward with soft overlong shaping.
    Correct = +1.0, Incorrect = -1.0, plus R_length(y).
    """
    penalty = soft_overlong_punishment(response_length, max_length, cache_length)
    if isinstance(is_correct, torch.Tensor):
        base_reward = torch.where(is_correct, 1.0, -1.0).float()
        return base_reward + penalty
    else:
        base_reward = 1.0 if is_correct else -1.0
        return base_reward + penalty


def is_valid_dapo_group(group_correctness: List[bool]) -> bool:
    """
    Checks DAPO dynamic sampling non-triviality constraint (Eq. 11):
        0 < |{o_i | is_equivalent(a, o_i)}| < G
    Returns True if the group contains both correct and incorrect outputs.
    """
    n_correct = sum(1 for c in group_correctness if c)
    g = len(group_correctness)
    return 0 < n_correct < g



def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes group-relative normalized advantage (Eq. 9):
        A_hat_i = (R_i - mean({R}_G)) / (std({R}_G) + eps)
    Expects shape (G,) or (B, G).
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


class DAPOLoss(nn.Module):
    """
    Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) Loss.
    Implements Token-level Policy Gradient Loss with Decoupled Clipping [1 - eps_low, 1 + eps_high].
    """
    def __init__(self, config: Optional[DAPOConfig] = None):
        super().__init__()
        self.config = config or DAPOConfig()

    def forward(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates DAPO objective (Eq. 10 & 12).

        Args:
            logprobs: Current policy log-probabilities, shape (B, T) or (G, T)
            old_logprobs: Rollout policy log-probabilities, shape (B, T) or (G, T)
            advantages: Group relative advantages, shape (B,) or (G,) or (B, T)
            mask: Binary token validity mask (1 for valid token, 0 for pad/truncated), shape (B, T)

        Returns:
            Dict containing:
                - loss: Scalar loss for gradient descent (-J_DAPO)
                - surrogate1: Mean unclipped objective
                - surrogate2: Mean clipped objective
                - clip_ratio: Fraction of tokens clipped
                - up_clip_ratio: Fraction of exploration tokens up-clipped
                - entropy: Approximate token entropy proxy
        """
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)  # Broadcast to (B, 1)

        # Ratio r_{i,t}(\theta) = \pi_\theta / \pi_{\theta_{old}} = exp(logprobs - old_logprobs)
        ratio = torch.exp(logprobs - old_logprobs)

        # Decoupled clipping: [1 - eps_low, 1 + eps_high]
        lower_bound = 1.0 - self.config.eps_low
        upper_bound = 1.0 + self.config.eps_high
        clipped_ratio = torch.clamp(ratio, min=lower_bound, max=upper_bound)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages

        # Min of surr1 and surr2 (standard pessimistic policy optimization)
        token_loss = -torch.min(surr1, surr2)

        if mask is None:
            mask = torch.ones_like(logprobs)

        if self.config.use_token_level_loss:
            # DAPO Token-Level reduction (Eq. 12):
            # Normalization by total valid token count across all sequences in group
            total_tokens = mask.sum().clamp(min=1.0)
            loss = (token_loss * mask).sum() / total_tokens
        else:
            # Baseline Sample-Level reduction (GRPO Eq. 5):
            # Normalization per sequence then across batch
            seq_lens = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            seq_loss = (token_loss * mask).sum(dim=1, keepdim=True) / seq_lens
            loss = seq_loss.mean()

        with torch.no_grad():
            up_clipped = ((ratio > upper_bound) & (mask > 0)).float()
            low_clipped = ((ratio < lower_bound) & (mask > 0)).float()
            total_valid = mask.sum().clamp(min=1.0)
            up_clip_ratio = up_clipped.sum() / total_valid
            clip_ratio = (up_clipped + low_clipped).sum() / total_valid

        return {
            "loss": loss,
            "surrogate1": (surr1 * mask).sum() / mask.sum().clamp(min=1.0),
            "surrogate2": (surr2 * mask).sum() / mask.sum().clamp(min=1.0),
            "clip_ratio": clip_ratio,
            "up_clip_ratio": up_clip_ratio,
        }


class DAPODynamicSampler:
    """
    Dynamic Sampling Buffer Manager (Algorithm 1 & Eq. 11).
    Collects prompts and groups, discarding trivial homogeneous batches (acc=0 or acc=1)
    until target effective batch size is attained.
    """
    def __init__(self, target_batch_size: int, group_size: int = 8):
        self.target_batch_size = target_batch_size
        self.group_size = group_size
        self.buffer: List[Dict] = []
        self.rejected_all_correct: int = 0
        self.rejected_all_incorrect: int = 0

    def submit_group(
        self,
        prompt_id: str,
        group_correctness: List[bool],
        responses: List[str],
        rewards: torch.Tensor,
        token_lengths: List[int],
    ) -> bool:
        """
        Evaluates group validity. If 0 < acc < 1, retains in buffer.
        """
        assert len(group_correctness) == self.group_size
        n_correct = sum(group_correctness)

        if n_correct == self.group_size:
            self.rejected_all_correct += 1
            return False
        if n_correct == 0:
            self.rejected_all_incorrect += 1
            return False

        # Non-trivial group: calculate relative advantages
        adv = compute_group_advantages(rewards)
        entry = {
            "prompt_id": prompt_id,
            "correctness": group_correctness,
            "responses": responses,
            "rewards": rewards,
            "advantages": adv,
            "token_lengths": token_lengths,
        }
        self.buffer.append(entry)
        return True

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.target_batch_size

    def pop_batch(self) -> List[Dict]:
        if not self.is_ready():
            raise RuntimeError(f"Buffer underflow: {len(self.buffer)} / {self.target_batch_size}")
        batch = self.buffer[:self.target_batch_size]
        self.buffer = self.buffer[self.target_batch_size:]
        return batch

    def statistics(self) -> Dict[str, Union[int, float]]:
        total_eval = len(self.buffer) + self.rejected_all_correct + self.rejected_all_incorrect
        eff_rate = len(self.buffer) / max(1, total_eval)
        return {
            "buffered_groups": len(self.buffer),
            "rejected_all_correct": self.rejected_all_correct,
            "rejected_all_incorrect": self.rejected_all_incorrect,
            "effective_gradient_rate": eff_rate,
        }
