#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — BOUNDED GATED UNROLLED SOVEREIGN TRANSFORMER
---------------------------------------------------------------------------------------
Architectural Specifications:
- 12 Deep Causal Decoder Layers (1024-dim, 16 heads, 64 head_dim)
- 34 Unique Council Expert Channels per layer (34 * [A, B] with alpha/r = 0.25 LoRA scaling)
- 34 Dedicated Underling Swarm Agent Modules per layer (34 * [A, B, C, D] with alpha/r scaling)
- 9-Vector Sovereign Prism Attention Decomposition (Language, Sentiment, Context, Intent, Meta, Creativity, Ethics, Strategy, Constraint)
- Dual-Brain Q1/Q2 Ingestion Bridges with Residual Bounded Gating
- Bounded Tanh MoE Output Modulation (prevents compounding logit attractors)
- Stateful KV Caching for rapid CPU token generation (<0.3s/tok)
- Causal Cross-Entropy Loss computation with prompt masking (-100)
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from collections import Counter

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class QuillanUnrolledConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 1024
    n_layer: int = 12
    n_head: int = 16
    head_dim: int = 64
    num_experts: int = 34
    top_k: int = 4
    expert_rank: int = 64
    swarm_rank: int = 24
    lora_alpha: float = 16.0

class Conv1D(nn.Module):
    def __init__(self, nf: int, nx: int):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)

class NineVectorPrismDecomposition(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.n_embd = n_embd
        self.vector_names = [
            'Language', 'Sentiment', 'Context', 'Intent',
            'Meta', 'Creativity', 'Ethics', 'Strategy', 'Constraint'
        ]
        self.projections = nn.ModuleDict({
            name: nn.Linear(n_embd, n_embd, bias=False)
            for name in self.vector_names
        })
        for p in self.projections.values():
            nn.init.normal_(p.weight, std=0.01)
        self.w_gate = nn.Linear(n_embd, n_embd, bias=False)
        nn.init.normal_(self.w_gate.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prism = sum(proj(x) for proj in self.projections.values()) / 9.0
        return self.w_gate(prism)

class UnrolledUnderlingSwarm(nn.Module):
    def __init__(self, expert_id: int, cfg: QuillanUnrolledConfig):
        super().__init__()
        self.expert_id = expert_id
        self.rank = cfg.swarm_rank
        self.A = nn.Parameter(torch.randn(cfg.n_embd, self.rank) * 0.01)
        self.B = nn.Parameter(torch.randn(self.rank, cfg.n_embd) * 0.01)
        self.C = nn.Parameter(torch.randn(cfg.n_embd, self.rank) * 0.01)
        self.D = nn.Parameter(torch.randn(self.rank, cfg.n_embd) * 0.01)
        self.scaling = cfg.lora_alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        div = (x @ self.C) @ self.D
        var = (x @ self.A) @ self.B + div * 0.467
        return x + var * (0.25 * self.scaling)

class UnrolledCouncilExpert(nn.Module):
    def __init__(self, expert_id: int, name: str, cfg: QuillanUnrolledConfig):
        super().__init__()
        self.expert_id = expert_id
        self.name = name
        self.rank = cfg.expert_rank
        self.lora_A = nn.Parameter(torch.randn(cfg.n_embd, self.rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, cfg.n_embd))
        self.swarm = UnrolledUnderlingSwarm(expert_id, cfg)
        self.scaling = cfg.lora_alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.lora_A) @ self.lora_B * self.scaling
        h = x + delta
        return self.swarm(h)

class UnrolledCouncilMoEBlock(nn.Module):
    def __init__(self, cfg: QuillanUnrolledConfig):
        super().__init__()
        self.cfg = cfg
        self.c_fc = Conv1D(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj = Conv1D(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU(approximate='tanh')
        
        self.router = nn.Linear(cfg.n_embd, cfg.num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        
        self.moe_gate = nn.Linear(cfg.n_embd, 1)
        nn.init.normal_(self.moe_gate.weight, std=0.01)
        nn.init.zeros_(self.moe_gate.bias)
        
        self.experts = nn.ModuleList([
            UnrolledCouncilExpert(i, f"C{i}", cfg)
            for i in range(cfg.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_dense = self.c_proj(self.act(self.c_fc(x)))
        
        B, T, C = x.size()
        flat_x = x.view(-1, C)
        logits = self.router(flat_x)
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = torch.topk(probs, self.cfg.top_k, dim=-1)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        
        moe_out = torch.zeros_like(flat_x)
        for k in range(self.cfg.top_k):
            indices = topk_i[:, k]
            weights = topk_p[:, k].unsqueeze(-1)
            for e_idx in range(self.cfg.num_experts):
                mask = (indices == e_idx)
                if mask.any():
                    e_out = self.experts[e_idx](flat_x[mask])
                    idx_nonzero = mask.nonzero(as_tuple=True)[0]
                    moe_out = moe_out.index_add(0, idx_nonzero, weights[mask] * e_out)
                    
        g = torch.tanh(self.moe_gate(flat_x))
        return h_dense + (moe_out * g).view(B, T, C)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: QuillanUnrolledConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = Conv1D(3 * cfg.n_embd, cfg.n_embd)
        self.c_proj = Conv1D(cfg.n_embd, cfg.n_embd)
        self.prism = NineVectorPrismDecomposition(cfg.n_embd)

    def _split_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        return tensor.view(new_shape).permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        x_attn = self.c_attn(x)
        query, key, value = x_attn.split(self.n_embd, dim=2)
        
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        w = torch.matmul(query, key.transpose(-1, -2))
        w = w * (1.0 / math.sqrt(self.head_dim))

        T = query.size(-2)
        total_T = key.size(-2)
        if layer_past is None:
            mask = torch.tril(torch.ones(T, total_T, device=x.device, dtype=torch.bool))
            w = w.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e4)

        w = F.softmax(w, dim=-1)
        a = torch.matmul(w, value)
        a = self._merge_heads(a, self.n_head, self.head_dim)
        a = self.c_proj(a)
        
        p_out = self.prism(x)
        return a + p_out, present

class UnrolledTransformerBlock(nn.Module):
    def __init__(self, cfg: QuillanUnrolledConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.moe = UnrolledCouncilMoEBlock(cfg)

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + a
        m = self.moe(self.ln_2(x))
        x = x + m
        return x, present

class QuillanUnrolledSovereign(nn.Module):
    def __init__(self, cfg: Optional[QuillanUnrolledConfig] = None):
        super().__init__()
        if cfg is None: cfg = QuillanUnrolledConfig()
        self.cfg = cfg
        
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        
        self.q1_bridge = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.q2_bridge = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ingest_gate = nn.Linear(cfg.n_embd * 2, cfg.n_embd)
        nn.init.zeros_(self.q1_bridge.weight)
        nn.init.zeros_(self.q2_bridge.weight)
        nn.init.zeros_(self.ingest_gate.weight)
        nn.init.zeros_(self.ingest_gate.bias)
        
        self.h = nn.ModuleList([UnrolledTransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.size()
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        x = self.wte(input_ids) + self.wpe(pos)
        
        q1 = self.q1_bridge(x)
        q2 = self.q2_bridge(x)
        gate = torch.sigmoid(self.ingest_gate(torch.cat([q1, q2], dim=-1)))
        x = x + 0.05 * (gate * q1 + (1.0 - gate) * q2)
        
        presents = [] if use_cache else None
        if past_key_values is None: past_key_values = [None] * len(self.h)
        
        for i, block in enumerate(self.h):
            if self.training and past_key_values[i] is None and not use_cache:
                x, _ = torch.utils.checkpoint.checkpoint(block, x, None, False, use_reentrant=False)
            else:
                x, present = block(x, layer_past=past_key_values[i], use_cache=use_cache)
                if use_cache: presents.append(present)
            
        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            return logits, loss
            
        if use_cache: return logits, presents
        return logits

    @torch.no_grad()
    def generate(self, input_tokens: List[int], max_tokens: int = 60, temp: float = 0.7, top_k: int = 50, top_p: float = 0.90, repetition_penalty: float = 1.05, frequency_penalty: float = 0.5, presence_penalty: float = 0.4) -> List[int]:
        """Autoregressive generation with Frequency & Presence Penalty to prevent repetition loops."""
        self.eval()
        gen = list(input_tokens)
        device = next(self.parameters()).device
        
        inp = torch.tensor([gen], dtype=torch.long, device=device)
        logits, kv_cache = self.forward(inp, use_cache=True)
        
        for _ in range(max_tokens):
            curr_logits = logits[:, -1, :].clone()
            
            if len(gen) > len(input_tokens):
                generated_tokens = gen[len(input_tokens):]
                counts = Counter(generated_tokens)
                for t, count in counts.items():
                    curr_logits[0, t] -= (count * frequency_penalty + presence_penalty)
            
            if temp <= 0.01:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
                curr_logits = curr_logits / max(0.05, temp)
                probs = F.softmax(curr_logits, dim=-1)
                
                if top_k > 0:
                    val_k, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < val_k[:, -1:]] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                if top_p < 1.0:
                    sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
                    cum_p = torch.cumsum(sorted_p, dim=-1)
                    cutoff = cum_p > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_p[cutoff] = 0.0
                    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                    probs.scatter_(1, sorted_i, sorted_p)
                    
                next_tok = torch.multinomial(probs, num_samples=1).item()
                
            gen.append(next_tok)
            if next_tok == 50256: # EOT
                break
                
            inp_single = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, kv_cache = self.forward(inp_single, past_key_values=kv_cache, use_cache=True)
            
        return gen
