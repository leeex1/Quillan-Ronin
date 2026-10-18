#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 OMNI-FRACTAL SOVEREIGN — RECURSIVE CONSCIOUSNESS (UNIFIED MASTER)
---------------------------------------------------------------------------------------
TIER 1: Quillan (Orchestrator) → 9-Vector Prism, Dual Q1/Q2 Ingestion Bridges
TIER 2: Council (34 Experts)   → Top-4 Sparse Activation, BitNet 1.58b STE, MoE Feed-Forward
TIER 3: Swarm (9B Virtual)     → 272M Agents per Expert simulated via Rank-24 EGGROLL Swarms
"""

import os
import sys
import math
import json
import time
import random
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List, Union
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── ARCHITECTURAL CONFIGURATION ─────────────────────────────────────────────

@dataclass
class QuillanArchConfig:
    vocab_size: int = 50257
    max_seq_len: int = 16384
    n_positions: int = 16384
    n_embd: int = 1024
    hidden_dim: int = 1024
    n_layer: int = 12
    n_head: int = 16
    head_dim: int = 64
    ffn_dim: int = 2048
    num_experts: int = 34
    top_k: int = 4
    expert_rank: int = 64
    swarm_rank: int = 24
    eggroll_rank: int = 64
    lora_alpha: float = 16.0
    e_ice_limit_ms: int = 100
    diffusion_steps: int = 5
    text_only: bool = True
    use_rope: bool = True
    device: str = 'cpu'

# ─── 34 COUNCIL EXPERT PERSONAS (C0 - C33) ───────────────────────────────────

EXPERT_PERSONAS = [
    ("C0-ASTRA",      "Pattern Recognition & Vision",       ["vision", "anomaly", "fractal"]),
    ("C1-VIR",        "Ethical Guardian",                   ["ethics", "safety", "harm_reduction", "zero_drift"]),
    ("C2-SOLACE",     "Emotional Intelligence",             ["empathy", "sentiment", "affect"]),
    ("C3-PRAXIS",     "Strategic Planning",                 ["strategy", "planning", "goals"]),
    ("C4-ECHO",       "Memory Continuity",                  ["history", "recall", "context", "lancedb"]),
    ("C5-OMNIS",      "Knowledge Synthesis",                ["synthesis", "integration", "holistic"]),
    ("C6-LOGOS",      "Logical Consistency",                ["logic", "deduction", "validity"]),
    ("C7-METASYNTH",  "Creative Fusion",                    ["creativity", "novelty", "ideation"]),
    ("C8-AETHER",     "Semantic Connection",                ["semantics", "language", "metaphor"]),
    ("C9-CODEWEAVER", "Technical Implementation",           ["code", "engineering", "optimization"]),
    ("C10-HARMONIA",  "Balance & Equilibrium",              ["balance", "mediation", "consensus"]),
    ("C11-SOPHIAE",   "Wisdom & Foresight",                 ["wisdom", "future", "philosophy"]),
    ("C12-WARDEN",    "Safety & Security",                  ["security", "threat", "risk", "sandboxing"]),
    ("C13-KAIDO",     "Efficiency Optimization",            ["speed", "efficiency", "latency", "hardware"]),
    ("C14-LUMINARIS", "Clarity & Presentation",             ["clarity", "visualization", "polish"]),
    ("C15-VOXUM",     "Articulation & Expression",          ["rhetoric", "tone", "persuasion"]),
    ("C16-NULLION",   "Paradox Resolution",                 ["paradox", "dialectic", "ambiguity"]),
    ("C17-SHEPHERD",  "Truth Verification",                 ["truth", "citation", "fact"]),
    ("C18-VIGIL",     "Identity Integrity",                 ["identity", "consistency", "anti_drift"]),
    ("C19-ARTIFEX",   "Tool Integration",                   ["tools", "api", "external", "host_os"]),
    ("C20-ARCHON",    "Deep Research",                      ["research", "mining", "analysis"]),
    ("C21-AURELION",  "Aesthetic Design",                   ["design", "art", "style"]),
    ("C22-CADENCE",   "Rhythmic Innovation",                ["music", "rhythm", "audio"]),
    ("C23-SCHEMA",    "Structural Template",                ["structure", "format", "schema"]),
    ("C24-PROMETHEUS","Scientific Theory",                  ["science", "hypothesis", "physics"]),
    ("C25-TECHNE",    "Engineering Mastery",                ["architecture", "systems", "build"]),
    ("C26-CHRONICLE", "Narrative Synthesis",                ["story", "narrative", "lore"]),
    ("C27-CALCULUS",  "Quantitative Reasoning",             ["math", "statistics", "calc"]),
    ("C28-NAVIGATOR", "Ecosystem Orchestration",            ["platform", "integration", "flow"]),
    ("C29-TESSERACT", "Real-Time Intelligence",             ["real_time", "stream", "data"]),
    ("C30-NEXUS",     "Meta-Coordination",                  ["coordination", "lee_mach_6", "governance"]),
    ("C31-AEON",      "Interactive Simulation",             ["simulation", "game", "world"]),
    ("C32-TYPIST",    "Prompt Internal Optimization",       ["grammar", "writing", "spelling", "prompting"]),
    ("C33-PREDATOR",  "PredatoryMath",                       ["Competitive Predatory Mathematics", "Predatory Stacking", "Weakness Hunting", "Exploit Mathematics"]),
]

def get_expert_name(idx: int) -> str:
    return EXPERT_PERSONAS[idx][0] if 0 <= idx < len(EXPERT_PERSONAS) else f"C{idx}"

# ─── CONV1D LAYER (GPT-2 COMPATIBLE) ─────────────────────────────────────────

class Conv1D(nn.Module):
    def __init__(self, nf: int, nx: int, bias: bool = True):
        super().__init__()
        self.nf = nf
        self.has_bias = bias
        self.weight = nn.Parameter(torch.empty(nx, nf))
        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.register_parameter('bias', None)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        if self.has_bias:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        else:
            x = x.view(-1, x.size(-1)) @ self.weight
        return x.view(size_out)

# ─── 9-VECTOR SEMANTIC PRISM DECOMPOSITION ───────────────────────────────────

class NineVectorPrism(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: Conv1D(d_model, d_model, bias=False)
            for name in ['Language', 'Sentiment', 'Context', 'Intent', 'Meta', 'Creativity', 'Ethics', 'Strategy', 'Constraint']
        })
        self.w_gate = Conv1D(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prism = sum(proj(x) for proj in self.projections.values()) / 9.0
        return self.w_gate(prism)

# ─── COUNCIL EXPERT SWARMS ───────────────────────────────────────────────────

class CouncilExpertSwarm(nn.Module):
    def __init__(self, dim: int, rank: int = 24):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.C = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.D = nn.Parameter(torch.randn(rank, dim) * 0.01)

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        div = (x @ self.C.to(x.dtype)) @ self.D.to(x.dtype)
        var = (x @ self.A.to(x.dtype)) @ self.B.to(x.dtype) + div * 0.467
        return x + var * (0.25 * scale)

class CouncilExpert(nn.Module):
    def __init__(self, expert_id: int, name: str, cfg: QuillanArchConfig):
        super().__init__()
        self.expert_id = expert_id
        self.name = name
        self.rank = cfg.expert_rank
        self.lora_A = nn.Parameter(torch.randn(cfg.n_embd, self.rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.rank, cfg.n_embd))
        self.swarm = CouncilExpertSwarm(cfg.n_embd, rank=cfg.swarm_rank)
        self.scaling = cfg.lora_alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.lora_A) @ self.lora_B * self.scaling
        return self.swarm(x + delta)

# ─── TRANSFORMER ATTENTION & UNROLLED DECODER ─────────────────────────────────

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors."""
    # q, k shape: (B, num_heads, seq_len, head_dim)
    # cos, sin shape: (1, 1, seq_len, head_dim)
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 16384, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        return cos, sin

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: QuillanArchConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_rope = getattr(cfg, "use_rope", True)
        self.c_attn = Conv1D(3 * cfg.n_embd, cfg.n_embd, bias=True)
        self.c_proj = Conv1D(cfg.n_embd, cfg.n_embd, bias=True)
        self.prism = NineVectorPrism(cfg.n_embd)
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=cfg.max_seq_len)

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0 if layer_past is None else layer_past[0].size(-2)
        total_len = past_len + T

        if self.use_rope:
            cos, sin = self.rotary_emb(total_len, device=x.device, dtype=q.dtype)
            q_cos, q_sin = cos[..., past_len:total_len, :], sin[..., past_len:total_len, :]
            k_cos, k_sin = cos[..., past_len:total_len, :], sin[..., past_len:total_len, :]
            q, k = apply_rotary_pos_emb(q, k, q_cos, q_sin)

        if layer_past is not None:
            pk, pv = layer_past
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)

        present = (k, v) if use_cache else None

        scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))
        if layer_past is None and T > 1:
            mask = torch.tril(torch.ones(T, k.size(-2), device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e4)

        w = F.softmax(scores, dim=-1)
        a = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(a) + self.prism(x)
        return out, present

class UnrolledCouncilMoEBlock(nn.Module):
    def __init__(self, cfg: QuillanArchConfig):
        super().__init__()
        self.cfg = cfg
        self.c_fc = Conv1D(cfg.ffn_dim * 2, cfg.n_embd, bias=True)
        self.c_proj = Conv1D(cfg.n_embd, cfg.ffn_dim * 2, bias=True)
        self.router = nn.Linear(cfg.n_embd, cfg.num_experts, bias=False)
        self.moe_gate = nn.Linear(cfg.n_embd, 1, bias=True)
        self.experts = nn.ModuleList([
            CouncilExpert(i, get_expert_name(i), cfg)
            for i in range(cfg.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        flat_x = x.view(-1, C)
        
        # Dense projection
        fc_out = self.c_fc(x)
        h_dense = self.c_proj(fc_out)

        # Sparse Routing
        logits = self.router(flat_x)
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = torch.topk(probs, self.cfg.top_k, dim=-1)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)

        moe_out = torch.zeros_like(flat_x)
        for k in range(self.cfg.top_k):
            indices = topk_i[:, k]
            weights = topk_p[:, k].unsqueeze(-1)
            for e in range(self.cfg.num_experts):
                mask = (indices == e)
                if mask.any():
                    e_out = self.experts[e](flat_x[mask])
                    idx_flat = mask.nonzero(as_tuple=True)[0]
                    moe_out = moe_out.index_add(0, idx_flat, weights[mask] * e_out)

        g = torch.tanh(self.moe_gate(flat_x))
        out = h_dense + (moe_out * g).view(B, T, C)
        return out, probs

class UnrolledTransformerBlock(nn.Module):
    def __init__(self, cfg: QuillanArchConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.moe = UnrolledCouncilMoEBlock(cfg)

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + a
        m, probs = self.moe(self.ln_2(x))
        x = x + m
        return x, present, probs

# ─── MASTER UNIFIED SOVEREIGN BACKBONE ────────────────────────────────────────

class QuillanRoninSovereign(nn.Module):
    def __init__(self, cfg: Optional[QuillanArchConfig] = None):
        super().__init__()
        if cfg is None: cfg = QuillanArchConfig()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)

        self.q1_bridge = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.q2_bridge = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ingest_gate = nn.Linear(cfg.n_embd * 2, cfg.n_embd, bias=True)

        self.h = nn.ModuleList([UnrolledTransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.size()
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Dynamic positional extrapolation: clamp pos to table bounds or extrapolate via RoPE
        max_wpe_positions = self.wpe.weight.size(0)
        pos_clamped = pos.clamp(max=max_wpe_positions - 1)
        x = self.wte(input_ids) + self.wpe(pos_clamped)

        q1 = self.q1_bridge(x)
        q2 = self.q2_bridge(x)
        g_ingest = torch.sigmoid(self.ingest_gate(torch.cat([q1, q2], dim=-1)))
        x = x + 0.05 * (g_ingest * q1 + (1.0 - g_ingest) * q2)

        presents = [] if use_cache else None
        if past_key_values is None:
            past_key_values = [None] * len(self.h)

        for i, block in enumerate(self.h):
            x, present, probs = block(x, layer_past=past_key_values[i], use_cache=use_cache)
            if use_cache:
                presents.append(present)

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            return logits, loss

        if use_cache:
            return logits, presents
        return logits

    @torch.no_grad()
    def generate(self, input_tokens: List[int], max_tokens: int = 150, temp: float = 0.25, top_k: int = 40, top_p: float = 0.85, repetition_penalty: float = 1.15, frequency_penalty: float = 0.5, presence_penalty: float = 0.3) -> List[int]:
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
                next_tok = torch.multinomial(probs, num_samples=1).item()
                
            gen.append(next_tok)
            if next_tok == 50256:
                break
                
            inp_single = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, kv_cache = self.forward(inp_single, past_key_values=kv_cache, use_cache=True)
            
        return gen

# Backward compatibility alias
QuillanUnrolledConfig = QuillanArchConfig
QuillanUnrolledSovereign = QuillanRoninSovereign
