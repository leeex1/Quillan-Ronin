"""
Mixture of Diverse Size Experts (MoDSE) Engine
=============================================
Formal Paper Reference:
"Mixture of Diverse Size Experts" (Manxi Sun, Wei Liu, Jian Luan, Pengzhi Gao, Bin Wang;
Xiaomi AI Lab, Beijing; Sept 2024 / arXiv:2409.12210)

Core Architectural & Mathematical Invariants:
1. Diverse Size Experts (Section 3.1, Eqs. 6-8):
   Assigns experts a spectrum of diverse parameter sizes by varying hidden dimensions h_i.
   Tokens of varying difficulty can select experts with appropriate parameter capacity.
   Grouped into n = N / 2 complementary pairs (i1_k, i2_k) satisfying:
       h_{i1}^k + h_{i2}^k = 2 * h_base
   guaranteeing that the total parameter size of the MoDSE model exactly matches
   that of a vanilla MoE model with uniform expert sizes.
2. Expert-Pair Hardware Allocation Strategy (Section 3.2):
   Places each complementary pair (i1_k, i2_k) on the same GPU node / accelerator.
   Since h_{i1} + h_{i2} = 2 * h_base, every device holds an identical number of parameters,
   eliminating GPU memory and computation imbalances.
3. Switch-Transformers Style Auxiliary Load Balancing Loss (Section 3.2, Eq. 9):
   L_a = alpha * N * sum_{i=1}^N f_i * P_i
   where f_i is the token routing fraction and P_i is the mean router probability.
4. Top-K Gating with Normalized Softmax:
   y_hat = sum_{i=1}^N G_hat_i(x) * E_hat_i(x)
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("MoDSE")


@dataclass
class MoDSEConfig:
    """Hyperparameters for Mixture of Diverse Size Experts Architecture."""
    dim: int = 128                                          # Hidden dimension d_model (default 2048/4096)
    n_layers: int = 2                                       # Number of transformer layers
    head_dim: int = 32                                      # Dimension per attention head
    base_hidden_dim: int = 256                              # Baseline uniform hidden dimension h
    # Complementary expert size pairs [(h_large, h_small), ...] where h_large + h_small = 2 * base_hidden_dim
    # In full scale: [(6912, 768), (6144, 1536), (4608, 3072), (3840, 3840)] -> sum = 7680 = 2 * 3840
    expert_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(384, 128), (320, 192), (288, 224), (256, 256)]
    )
    top_k: int = 2                                          # Number of active experts per token
    n_heads: int = 4                                        # Query attention heads
    n_kv_heads: int = 1                                     # KV attention heads (GQA)
    vocab_size: int = 32000                                 # Vocabulary size
    context_len: int = 2048                                 # Context length
    rope_theta: float = 10000.0                             # RoPE base
    rms_norm_eps: float = 1e-5                              # RMSNorm epsilon
    aux_loss_coef: float = 0.01                             # Auxiliary load balance multiplier alpha

    @property
    def num_experts(self) -> int:
        return len(self.expert_pairs) * 2

    @property
    def expert_hidden_dims(self) -> List[int]:
        dims: List[int] = []
        for pair in self.expert_pairs:
            dims.extend([pair[0], pair[1]])
        return dims

    @property
    def num_queries_per_kv(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        return self.n_heads // self.n_kv_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to tensor of shape (batch, heads, seq, head_dim)."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class RoPEEmbedding(nn.Module):
    """Precomputes Rotary Positional Embedding sinusoids."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


class MoDSEAttention(nn.Module):
    """Grouped-Query Attention for MoDSE Transformer."""

    def __init__(self, config: MoDSEConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_queries_per_kv

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=config.context_len, theta=config.rope_theta)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(seq_len, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        else:
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class DiverseSwiGLUExpert(nn.Module):
    """
    SwiGLU Expert with configurable hidden dimension h_i:
    E_i(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    where W_gate, W_up in R^{dim x h_i} and W_down in R^{h_i x dim}.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    @property
    def parameter_count(self) -> int:
        return 3 * self.dim * self.hidden_dim


class MoDSERouter(nn.Module):
    """
    MoDSE Gating Router with Top-K selection and Switch-Transformers Load Balancing Loss.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int = 2, aux_loss_coef: float = 0.01):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Flattened tokens (num_tokens, dim)
        Returns:
            routing_weights: (num_tokens, top_k) normalized softmax weights summing to 1.0
            selected_experts: (num_tokens, top_k) expert indices in {0, ..., num_experts - 1}
            aux_loss: Scalar Switch-Transformers load balancing loss (Eq. 9)
        """
        num_tokens = x.shape[0]
        logits = self.gate(x)  # (num_tokens, num_experts)

        # Top-K selection
        top_k_logits, selected_experts = torch.topk(logits, self.top_k, dim=-1)
        routing_weights = F.softmax(top_k_logits, dim=-1)

        # Auxiliary load balancing loss (Section 3.2, Eq. 9):
        # L_a = alpha * N * sum_{i=1}^N f_i * P_i
        router_probs = F.softmax(logits, dim=-1)  # (num_tokens, num_experts)
        P_i = router_probs.mean(dim=0)            # (num_experts,)

        mask_selected = torch.zeros(num_tokens, self.num_experts, device=x.device)
        mask_selected.scatter_(1, selected_experts, 1.0)
        f_i = mask_selected.mean(dim=0)           # (num_experts,)

        aux_loss = self.aux_loss_coef * float(self.num_experts) * torch.sum(f_i * P_i)

        return routing_weights, selected_experts, aux_loss


class MoDSEBlock(nn.Module):
    """
    Mixture of Diverse Size Experts (MoDSE) Layer:
    Executes N experts with diverse hidden dimensions {h_1, ..., h_N}.
    Provides expert-pair allocation mapping to ensure equal GPU hardware loading.
    """

    def __init__(self, config: MoDSEConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.expert_pairs = config.expert_pairs
        self.hidden_dims = config.expert_hidden_dims

        # Enforce Section 3.1 Eq. 8: h_{i1}^k + h_{i2}^k = 2 * base_hidden_dim
        for pair_idx, (h1, h2) in enumerate(self.expert_pairs):
            pair_sum = h1 + h2
            expected_sum = 2 * config.base_hidden_dim
            assert pair_sum == expected_sum, (
                f"Pair {pair_idx} sum {pair_sum} does not equal 2 * base_hidden_dim ({expected_sum})"
            )

        self.router = MoDSERouter(
            dim=config.dim,
            num_experts=self.num_experts,
            top_k=config.top_k,
            aux_loss_coef=config.aux_loss_coef
        )

        # Construct diverse size experts
        self.experts = nn.ModuleList([
            DiverseSwiGLUExpert(config.dim, h_i)
            for h_i in self.hidden_dims
        ])

    def get_device_allocation_map(self) -> Dict[int, Dict[str, Any]]:
        """
        Computes the expert-pair hardware allocation map (Section 3.2).
        Pairs (2k, 2k+1) are co-located on Device k.
        Returns device mapping and verifies that each device contains identical total parameters.
        """
        device_map: Dict[int, Dict[str, Any]] = {}
        for pair_idx, (h1, h2) in enumerate(self.expert_pairs):
            e1_idx = 2 * pair_idx
            e2_idx = 2 * pair_idx + 1
            e1_params = self.experts[e1_idx].parameter_count
            e2_params = self.experts[e2_idx].parameter_count
            total_pair_params = e1_params + e2_params

            device_map[pair_idx] = {
                "experts": (e1_idx, e2_idx),
                "hidden_dims": (h1, h2),
                "device_params": total_pair_params,
                "balanced_with_baseline": (h1 + h2 == 2 * self.config.base_hidden_dim)
            }
        return device_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized token dispatch to diverse size experts.
        Args:
            x: (batch_size, seq_len, dim)
        Returns:
            out: (batch_size, seq_len, dim)
            aux_loss: Scalar load balancing loss
        """
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Route tokens
        routing_weights, selected_experts, aux_loss = self.router(x_flat)

        out_flat = torch.zeros_like(x_flat)

        # Dispatch tokens to each diverse expert
        for expert_idx in range(self.num_experts):
            token_indices, k_rank = torch.where(selected_experts == expert_idx)
            if token_indices.numel() == 0:
                continue

            expert_tokens = x_flat[token_indices]
            expert_out = self.experts[expert_idx](expert_tokens)
            weights = routing_weights[token_indices, k_rank].unsqueeze(-1)
            out_flat.index_add_(0, token_indices, weights * expert_out)

        out = out_flat.view(batch_size, seq_len, dim)
        return out, aux_loss


class MoDSEDecoderLayer(nn.Module):
    """Transformer Decoder Layer with MoDSE diverse-size FFN."""

    def __init__(self, config: MoDSEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = MoDSEAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.moe = MoDSEBlock(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x + self.attn(self.attn_norm(x), mask=mask)
        moe_out, aux_loss = self.moe(self.ffn_norm(h))
        out = h + moe_out
        return out, aux_loss


class MoDSEModel(nn.Module):
    """Full MoDSE Transformer Language Model Backbone."""

    def __init__(self, config: MoDSEConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([
            MoDSEDecoderLayer(config) for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            assert input_ids is not None, "Must provide input_ids or inputs_embeds"
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        total_aux_loss = torch.tensor(0.0, device=h.device)
        for layer in self.layers:
            h, layer_aux = layer(h, mask=mask)
            total_aux_loss = total_aux_loss + layer_aux

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, total_aux_loss


def verify_modse_invariants() -> Dict[str, Any]:
    """
    Verifies all mathematical and architectural invariants of MoDSE (Paper 67):
    1. Diverse Expert Size Pairs: for all pairs, h1 + h2 = 2 * h_base.
    2. Total Parameter Conservation: sum of all expert parameters equals vanilla MoE.
    3. Hardware Memory Balancing: each pair holds strictly identical parameter counts.
    4. Heterogeneous Dispatch: tokens successfully process through varying dimension experts.
    5. Full MoDSEModel forward, causal LM loss, and gradient backpropagation.
    """
    results: Dict[str, Any] = {}
    cfg = MoDSEConfig(
        dim=64,
        n_layers=2,
        head_dim=16,
        base_hidden_dim=128,
        expert_pairs=[(192, 64), (160, 96), (144, 112), (128, 128)],  # sum = 256 for all
        top_k=2,
        n_heads=4,
        n_kv_heads=1,
        vocab_size=100,
        aux_loss_coef=0.01
    )

    batch_size = 2
    seq_len = 8

    # 1. Verify Complementary Pair Sum Invariant
    for pair_idx, (h1, h2) in enumerate(cfg.expert_pairs):
        assert h1 + h2 == 2 * cfg.base_hidden_dim, \
            f"Pair {pair_idx} does not satisfy h1 + h2 == 2 * h_base"
    results["complementary_pair_sum_verified"] = True

    # 2. Verify Total Parameter Conservation Invariant
    block = MoDSEBlock(cfg)
    total_modse_expert_params = sum(e.parameter_count for e in block.experts)
    expected_vanilla_params = cfg.num_experts * (3 * cfg.dim * cfg.base_hidden_dim)
    assert total_modse_expert_params == expected_vanilla_params, \
        f"Parameter count mismatch: {total_modse_expert_params} != {expected_vanilla_params}"
    results["parameter_conservation_verified"] = True

    # 3. Verify Hardware Device Memory Balancing Invariant
    dev_map = block.get_device_allocation_map()
    expected_device_params = 2 * (3 * cfg.dim * cfg.base_hidden_dim)
    for dev_id, info in dev_map.items():
        assert info["device_params"] == expected_device_params, \
            f"Device {dev_id} has unbalanced parameters: {info['device_params']} != {expected_device_params}"
        assert info["balanced_with_baseline"] is True
    results["device_memory_balancing_verified"] = True

    # 4. Forward Pass & Heterogeneous Token Dispatch
    x = torch.randn(batch_size, seq_len, cfg.dim, requires_grad=True)
    out, aux_loss = block(x)
    assert out.shape == (batch_size, seq_len, cfg.dim)
    assert aux_loss.item() > 0.0

    (out.sum() + aux_loss).backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert block.router.gate.weight.grad is not None
    assert any(e.w_gate.weight.grad is not None for e in block.experts)
    results["heterogeneous_dispatch_grad_verified"] = True

    # 5. Full MoDSEModel End-to-End Verification
    model = MoDSEModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits, total_aux = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert total_aux.item() > 0.0

    target = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    lm_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1)) + total_aux
    lm_loss.backward()

    assert model.embed_tokens.weight.grad is not None
    assert model.layers[0].moe.router.gate.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    results["full_model_training_verified"] = True

    results["all_invariants_passed"] = True
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing MoDSE (Mixture of Diverse Size Experts) Invariants...")
    res = verify_modse_invariants()
    print("Verification Results:", res)
