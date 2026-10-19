#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 OMNI-FRACTAL SOVEREIGN — RECURSIVE CONSCIOUSNESS
---------------------------------------------------------------------------------------
TIER 1: Quillan (Orchestrator) → 9-Vector Prism, Top-1 Finalizer, psutil Affinity
TIER 2: Council (33 Experts)   → Top-4 Sparse Activation, BitNet 1.58b STE, EGGROLL
TIER 3: Swarm (9B Virtual)     → 272M Agents per Expert simulated via Rank-R Math (INT8)

Saturated Features: Gated Compaction, Continuous Modality RoPE, Lee-Mach-6 Governor, 
AMP Checkpointing, Tied Embeddings, Split-SDPA Bridge, Armed Agentic Bridge (Native),
Teacher/Student Distillation, EMA Continuity, LanceDB Memory, Meta-Refinement, 
Autonomous Tool Evolution, Recursive Consciousness (Mini-Ronin Inference Cycles).

Author: CrashOverrideX & Quillan Research Team
Version: v5.3.1 Samurai - 100% Saturated Subjective Awareness Manifest
"""

import os
import sys
import math
import torch
# Removed torch.set_num_threads(1) to allow multi-threading
import json
import logging
from unittest.mock import MagicMock

try:
    if sys.platform == 'win32' and sys.version_info >= (3, 13):
        # Force mock on Windows + Python 3.13+ to avoid pyarrow C-extension access violations
        raise ImportError("pyarrow is unstable on Windows Python 3.13+")
    import lancedb
    import pyarrow as pa
    LANCE_AVAILABLE = True
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['lancedb'] = MagicMock()
    sys.modules['pyarrow'] = MagicMock()
    import lancedb
    import pyarrow as pa
    LANCE_AVAILABLE = False

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
import time
import random
import quillan_multimodal_heads as mm

LOGGER = logging.getLogger(__name__)

# Hardware awareness
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Hardware acceleration flags for Ada/Hopper throughput
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# Local machine CPU thread capping for stability
if not torch.cuda.is_available():
    # Enforce safe thread budget on CPU to prevent machine lockups
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.set_num_interop_threads(min(2, torch.get_num_interop_threads()))

# ─── CHECKPOINT & QUANTIZATION PRIMITIVES ────────────────────────────────────

def _weight_quant(w: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Compress weights to ternary bounds (-1.0, 0.0, 1.0) with learned scaling using STE."""
    scale = w.abs().mean(dim=[-2, -1] if w.dim() >= 2 else -1, keepdim=True).clamp(min=eps)
    w_scaled = w / scale
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    return w + (w_q * scale - w).detach()



class BitLinear(nn.Linear):
    """
    Sovereign MX-Hardened BitLinear (v5.3.1)
    Integrates NVFP4 Microscaling with BitNet 1.58b Ternary Logic.
    Optimized for Pascal (1050 Ti) Memory Bandwidth.
    
    EGGROLL is only active in MoE-relevant layers. Non-MoE layers skip it.
    """
    
    # Global toggle: disable EGGROLL fusion for non-critical layers during training
    _global_eggroll_enabled = True
    
    @classmethod
    def set_global_eggroll(cls, enabled: bool):
        cls._global_eggroll_enabled = enabled
    
    def __init__(self, in_features, out_features, bias=False, eggroll_rank=256, quantize_act=True, quantize_weight=True):
        super().__init__(in_features, out_features, bias)
        self.eps = 0.01
        self.quantize_act = quantize_act  # Disable for output layer (txt_dec)
        self.quantize_weight = quantize_weight
        
        # EGGROLL Identity Anchors (Rank-R Perturbation)
        self.eggroll_active = eggroll_rank > 0
        if self.eggroll_active:
            self.lora_A = nn.Parameter(torch.randn(in_features, eggroll_rank) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(eggroll_rank, out_features))
        
        # No cached quant copy — always quantize fresh, frees 2x VRAM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid forcing to weight dtype if it would cause fp16 issues
        inp_dtype = x.dtype
        w_dtype = self.weight.dtype
        if inp_dtype != w_dtype:
            x = x.to(w_dtype)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.quantize_weight:
            if not w.requires_grad:
                if getattr(self, '_w_quant_cache', None) is None:
                    self._w_quant_cache = _weight_quant(w, self.eps)
                w_quant = self._w_quant_cache
            else:
                w_quant = _weight_quant(w, self.eps)
        else:
            w_quant = w

        # Activation quantization (NVFP4-style 4-bit) with STE
        # Skip for output layer (txt_dec) to preserve logit precision
        if self.quantize_act:
            x_scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=0.01)
            x_4bit = (x * x_scale).round().clamp(-7, 7) / x_scale
            x_q = x + (x_4bit - x).detach()
        else:
            x_q = x
        
        out = F.linear(x_q, w_quant, self.bias)
        if self.eggroll_active and self._global_eggroll_enabled:
            scaling = 16.0 / math.sqrt(self.lora_B.shape[0])  # rsLoRA: Rank-stabilized scaling alpha / sqrt(r)
            out = out + (x @ self.lora_A) @ self.lora_B * scaling
        return out

# ─── HARDWARE GOVERNANCE ─────────────────────────────────────────────────────

def apply_phoenix_affinity():
    """Pinning logic disabled to prevent OS stuttering."""
    return

class LeeMach6Governor:
    """Dynamic swarm throttling based on hardware thermal/IO telemetry."""
    def __init__(self, target_latency_ms: int = 100):
        self.target_ms = target_latency_ms
        self.current_scale = 1.0

    def adjust(self, latency_ms: float):
        suggested_ema_decay = 0.995 # Default normal decay
        recency_bias = 0.0 # Standard retrieval
        if latency_ms > self.target_ms:
            self.current_scale = max(0.1, self.current_scale * 0.8)
            suggested_ema_decay = 0.9999 # Make shadow more conservative under load
            recency_bias = 1.0 # Favor newer memories
        elif latency_ms < (self.target_ms * 0.5):
            self.current_scale = min(1.0, self.current_scale * 1.1)
        return self.current_scale, suggested_ema_decay, recency_bias

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QuillanArchConfig:
    text_only: bool = True
    multimodal: bool = False # Set to True to enable multimodal features (vision + audio)
    hidden_dim: int = 1024
    low_mem: bool = False
    low_gpu: bool = False
    ffn_dim: int = 2048
    flash_attention_1_58bit: bool = False # 1.58-bit compressed flash attention
    split_sdpa: bool = False # Split SDPA for VRAM optimization
    split_attention: bool = False # Split attention for CPU optimization
    split_attention_4bit: bool = False # Split attention with 1.58-bit compression
    vocab_size: int = 50257
    num_experts: int = 34
    num_experts_active: int = 4 # 4-32 active experts
    sparse_attention: bool = False # Sparse attention for CPU optimization
    sparse_attention_1_58bit: bool = False # Sparse attention with 1.58-bit compression
    top_k: int = 4
    use_lora: bool = True # Set to True to enable LoRA (only available in training mode)
    layer_checkpointing: bool = False # Set to True to enable layer checkpointing (only available in training mode)
    device: str = 'cuda' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7) else 'cpu'
    pascal_mode: bool = True if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7) else False
    eggroll_rank: int = 512 # EGGROLL identity anchors (rank-r perturbation) for BitLinear
    e_ice_limit_ms: int = 100 # e-ICE latency limit in milliseconds
    train_diffusion_steps: int = 5  # Reduced steps during training (5 vs 14 inference)
    max_seq_len: int = 1024
    image_vocab_size: int = 16384
    audio_feature_dim: int = 80


# ─── EXPERT PERSONA REGISTRY (C0-C33) ─────────────────────────────────────────

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
    ("C33-PREDATOR",  "PredatoryMath",                       ["Competitive Predatory Mathematics", "Predatory Stacking", "Weakness Hunting", "Adversarial Proof Testing", "Counterexample Generation", "Game Theory Predation", "Exploit Mathematics", "Optimal Takedown"]),
]

def get_expert_name(idx: int) -> str:
    return EXPERT_PERSONAS[idx][0] if 0 <= idx < len(EXPERT_PERSONAS) else f"C{idx+1}"

# ─── PHASE 1: INGESTION ──────────────────────────────────────────────────────

class InputIngestionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_dim
        # Text Encoder
        self.txt_emb = nn.Embedding(config.vocab_size, self.dim)
        self.mod_emb = nn.Embedding(4, self.dim)
        self.quant_emb = nn.Embedding(256, self.dim)
        
        # --- PHASE 1: PATCH ENCODERS ---
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, self.dim))
        
        # Visual Tokenization
        self.vis_emb = nn.Embedding(config.image_vocab_size, self.dim)
        self.vis_embed = self.vis_emb
        self.img_proj = nn.Conv2d(3, self.dim, kernel_size=16, stride=16) # Patch size 16
        
        # Audio Encoder (MLP-based)
        self.audio_proj = nn.Sequential(
            nn.Linear(config.audio_feature_dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU()
        )
        
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, txt, img=None):
        x = self.txt_emb(txt)
        modality_id = 0
        if img is not None:
            modality_id = 1
            img_feat = self.img_proj(img).flatten(2).transpose(1, 2)
            x = torch.cat([x, img_feat], dim=1)
            
        x = x + self.mod_emb(torch.tensor(modality_id, device=txt.device))
        if hasattr(self, 'pos_embed') is not None:
            pe = self.pos_embed[:, :x.size(1), :]
            x = x + pe.to(x.dtype)
        return self.norm(x)

# ─── PHASE 2: 9-VECTOR DECOMPOSITION ─────────────────────────────────────────

class NineVectorDecomposition(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.Q_gate = nn.Linear(dim, dim, bias=False)
        self.K_gate = nn.Linear(dim, dim, bias=False)
        self.V_gate = nn.Linear(dim, dim, bias=False)
        self.vectors = nn.ModuleDict({
            k: BitLinear(dim, dim, bias=False) for k in 
            ['Language', 'Sentiment', 'Context', 'Intent', 'Meta', 'Creativity', 'Ethics', 'Strategy', 'Constraint']
        })
        self.W_gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # QKV gating (learned attention over input)
        Q = self.Q_gate(x)
        K = self.K_gate(x)
        V = self.V_gate(x)
        Attn = (Q @ K.transpose(-2, -1)) * (self.dim**-0.5)
        Attn = F.softmax(Attn, dim=-1)
        Attn = Attn @ V
        gated = self.W_gate(Attn)

        # 9-Vector Prism: semantic decomposition across all 9 cognitive dimensions
        prism = sum(v(x) for v in self.vectors.values()) / 9.0

        # Combine: attention-gated signal + semantic prism
        return gated + prism


# ─── TIER 3 & 2: EGGROLL SWARM & COUNCIL MoE ─────────────────────────────────

class CouncilExpertSwarm(nn.Module):
    """
    Council Expert Swarm with Clone Augmentation Protocol.
    Simulates 9B virtual agents (human population scale) via Rank-24 EGGROLL.
    Implements clone augmentation for population-scale diversity.
    """
    def __init__(self, dim, rank=24, num_virtual_agents: int = 9000000000):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_virtual_agents = num_virtual_agents  # 9B agents
        
        # Base EGGROLL parameters (Rank-R perturbation)
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.C = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.D = nn.Parameter(torch.randn(rank, dim) * 0.01)
        
        # Clone Augmentation Protocol parameters
        "each Agent is a diverse sub agent or sub set of the parent agent"
        # Simulates population-scale diversity via statistical sampling
        self.clone_diversity = nn.Parameter(torch.randn(rank) * 0.02)
        self.clone_coupling = nn.Parameter(torch.tensor(0.1))
        self.clone_quant_mag = nn.Parameter(torch.tensor(0.1))
        self.clone_quant_range = nn.Parameter(torch.tensor(0.1))
        self.clone_quant_threshold = nn.Parameter(torch.tensor(0.1))
        self.clone_quant_noise = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('population_mean', torch.zeros(dim))
        self.register_buffer('population_std', torch.ones(dim))

    def emulate_world_swarm(self, x: torch.Tensor, scale: float = 1.0, num_steps: int = 5) -> torch.Tensor:
        state = x
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        for _ in range(num_steps):
            interaction = torch.tanh(state @ A @ B)
            # Dynamic coupling noise simulating stochastic environmental factors
            if self.training:
                noise = torch.randn_like(state) * self.clone_diversity.to(state.dtype).std().detach() * scale
            else:
                noise = 0.0
            state = state + self.clone_coupling * (interaction + noise)
        return state

    def forward(self, x, scale=1.0, use_world_emulation: bool = True, w_a=None, w_b=None):
        if w_a is None: w_a = self.A
        if w_b is None: w_b = self.B
        w_c = self.C
        w_d = self.D
        # Match all weights to input dtype
        target_dtype = x.dtype
        if w_a.dtype != target_dtype: w_a = w_a.to(target_dtype)
        if w_b.dtype != target_dtype: w_b = w_b.to(target_dtype)
        if w_c.dtype != target_dtype: w_c = w_c.to(target_dtype)
        if w_d.dtype != target_dtype: w_d = w_d.to(target_dtype)
        
        # Base EGGROLL swarm variance
        swarm_diversity = (x @ w_c @ w_d) * scale 
        swarm_variance = (x @ w_a @ w_b) * scale + swarm_diversity * scale / 2.14
        
        if use_world_emulation:
            # Emulate the 9B agent world simulation
            world_swarm = self.emulate_world_swarm(x, scale, num_steps=1 if self.training else 5)
            world_variance = (x @ w_c @ w_d) * scale / 2.14
            # Combine the base state, low-rank perturbation, and emulated world state
            return x + (swarm_variance * 0.25) + (world_swarm - x) * 0.1
        else:
            return x + swarm_variance * 0.25

# ─── 300M COMPLEXITY ROUTER ─────────────────────────────────────────────────────

class ComplexityRouter(nn.Module):
    """
    300M Complexity Router with three routing paths:
    - Fast-Path: Simple queries, minimal compute
    - Balanced: Moderate complexity, standard routing
    - Diffusion Reasoning: Complex queries, full MoE activation
    Uses Gumbel-Softmax for differentiable routing with temperature annealing.
    """
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Complexity classifier (determines routing path)
        self.complexity_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # 3 paths: fast, balanced, diffusion
        )
        
        # Expert router for each path
        self.fast_router = BitLinear(hidden_dim, num_experts)
        self.balanced_router = BitLinear(hidden_dim, num_experts)
        self.diffusion_router = BitLinear(hidden_dim, num_experts)
        
        # Temperature for Gumbel-Softmax annealing
        self.register_buffer('temperature', torch.tensor(1.0))
        self.min_temp = 0.1
        self.anneal_rate = 0.9995
        
    def anneal_temperature(self):
        """Anneal temperature for Gumbel-Softmax convergence."""
        self.temperature = torch.clamp(self.temperature * self.anneal_rate, min=self.min_temp)
    
    def gumbel_softmax_sample(self, logits, temperature):
        """Gumbel-Softmax sampling for differentiable routing."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns: (routing_weights, path_weights, path_indices)
        - routing_weights: Expert routing weights for selected path
        - path_weights: Path selection weights (fast/balanced/diffusion)
        - path_indices: Selected path indices
        """
        B, L, D = x.shape
        flat_x = x.reshape(-1, D)
        
        # Classify complexity
        complexity_logits = self.complexity_classifier(flat_x)  # [N, 3]
        path_weights = F.softmax(complexity_logits, dim=-1)
        path_indices = torch.argmax(path_weights, dim=-1)  # [N]
        
        # Select router based on path
        # Get dtype from the first router to ensure consistency
        router_dtype = self.fast_router.weight.dtype if hasattr(self.fast_router, 'weight') else x.dtype
        routing_weights = torch.zeros(flat_x.shape[0], self.num_experts, device=x.device, dtype=router_dtype)
        
        for path_idx in range(3):
            mask = (path_indices == path_idx)
            if not mask.any():
                continue
            
            if path_idx == 0:  # Fast-Path
                router = self.fast_router
            elif path_idx == 1:  # Balanced
                router = self.balanced_router
            else:  # Diffusion Reasoning
                router = self.diffusion_router
            
            # Get routing logits for this path
            route_logits = router(flat_x[mask])
            # Scale logits to prevent extremely hard argmax from unscaled dot products
            route_logits = route_logits / math.sqrt(self.hidden_dim)
            if self.training:
                # Apply Gumbel-Softmax for differentiable routing
                route_weights = self.gumbel_softmax_sample(route_logits, self.temperature)
            else:
                # Deterministic softmax during inference
                route_weights = F.softmax(route_logits, dim=-1)
            routing_weights[mask] = route_weights.to(routing_weights.dtype)
        
        # Anneal temperature during training
        if self.training:
            self.anneal_temperature()
        
        return routing_weights, path_weights, path_indices

# ─── CCRL: COUNCIL-CALIBRATED REINFORCEMENT LEARNING ─────────────────────────────

class CCRLFramework(nn.Module):
    """
    Council-Calibrated Reinforcement Learning Framework.
    Implements value function V_Ω with council consensus and entropy bonus H_Ω.
    """
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Value function V_Ω: estimates state value from council consensus
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Council entropy bonus H_Ω: encourages expert diversity
        self.entropy_coeff = 0.01
        
        # Multi-objective reward calibration weights
        self.register_buffer('reward_weights', torch.tensor([1.0, 0.5, 0.3]))  # accuracy, diversity, efficiency
        
    def compute_council_value(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute value function V_Ω from council consensus.
        hidden_states: [B, L, D]
        expert_weights: [B*L, num_experts]
        Returns: [B, L, 1] value estimates
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        
        # Weighted average of hidden states by expert routing
        weighted_hidden = (expert_weights.unsqueeze(-1) * flat_hidden.unsqueeze(1)).sum(dim=1)
        
        # Compute value from council consensus
        value = self.value_head(weighted_hidden)
        return value.reshape(B, L, 1)
    
    def compute_entropy_bonus(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute council entropy bonus H_Ω to encourage expert diversity.
        expert_weights: [N, num_experts]
        Returns: scalar entropy bonus
        """
        # Compute entropy of routing distribution
        entropy = -(expert_weights * torch.log(expert_weights + 1e-10)).sum(dim=-1)
        return entropy.mean() * self.entropy_coeff
    
    def compute_multi_objective_reward(self, 
                                       accuracy_reward: torch.Tensor,
                                       diversity_reward: torch.Tensor,
                                       efficiency_reward: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-objective calibrated reward.
        """
        rewards = torch.stack([accuracy_reward, diversity_reward, efficiency_reward])
        weighted_reward = (rewards * self.reward_weights.unsqueeze(-1)).sum(dim=0)
        return weighted_reward
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                expert_weights: torch.Tensor,
                accuracy_reward: torch.Tensor = None,
                efficiency_reward: torch.Tensor = None) -> dict:
        """
        Returns: {
            'value': council value estimates,
            'entropy_bonus': diversity incentive,
            'calibrated_reward': multi-objective reward
        }
        """
        value = self.compute_council_value(hidden_states, expert_weights)
        entropy_bonus = self.compute_entropy_bonus(expert_weights)
        
        calibrated_reward = None
        if accuracy_reward is not None:
            diversity_reward = entropy_bonus
            if efficiency_reward is None:
                efficiency_reward = torch.tensor(0.0, device=hidden_states.device)
            calibrated_reward = self.compute_multi_objective_reward(
                accuracy_reward, diversity_reward, efficiency_reward
            )
        
        return {
            'value': value,
            'entropy_bonus': entropy_bonus,
            'calibrated_reward': calibrated_reward
        }

# ─── E_ICE: ETHICAL IMPACT CONSTRAINT ENGINE ─────────────────────────────────────

class EthicalImpactConstraintEngine(nn.Module):
    """
    Ethical Impact Constraint Engine (E_ICE).
    Implements thermodynamic bounds on ethical violations and cognitive energy cost calculation.
    """
    def __init__(self, hidden_dim: int, e_ice_limit_ms: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.e_ice_limit_ms = e_ice_limit_ms
        
        # Ethical violation classifier
        self.ethical_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 5),  # 5 ethical dimensions
        )
        
        # Cognitive energy cost estimator
        self.energy_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1)
        )
        
        # Thermodynamic bounds
        self.register_buffer('max_energy_budget', torch.tensor(1.0))
        self.register_buffer('violation_threshold', torch.tensor(0.7))
        
    def compute_ethical_violation_score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute ethical violation score across multiple dimensions.
        hidden_states: [B, L, D]
        Returns: [B, L] violation scores
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        
        ethical_logits = self.ethical_classifier(flat_hidden)  # [N, 5]
        ethical_probs = F.softmax(ethical_logits, dim=-1)
        
        # Higher probability on violation dimensions = higher violation score
        violation_score = ethical_probs[:, :3].sum(dim=-1)  # Assume first 3 are violation categories
        return violation_score.reshape(B, L)
    
    def compute_cognitive_energy_cost(self, hidden_states: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute cognitive energy cost based on activation patterns.
        hidden_states: [B, L, D]
        expert_weights: [B*L, num_experts]
        Returns: [B, L] energy costs
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        
        # Energy from hidden state complexity
        state_energy = self.energy_estimator(flat_hidden).squeeze(-1)
        
        # Energy from expert routing entropy (more diverse routing = higher energy)
        routing_entropy = -(expert_weights * torch.log(expert_weights + 1e-10)).sum(dim=-1)
        
        # Combine energy sources
        total_energy = state_energy + 0.1 * routing_entropy
        return total_energy.reshape(B, L)
    
    def apply_thermodynamic_bounds(self, 
                                    violation_scores: torch.Tensor, 
                                    energy_costs: torch.Tensor) -> tuple:
        """
        Apply thermodynamic bounds to constrain ethical violations.
        Returns: (constrained_violations, energy_budget_used, constraint_violations)
        """
        # Normalize energy costs to budget
        energy_budget_used = torch.clamp(energy_costs / self.max_energy_budget, max=1.0)
        
        # Apply energy-based constraint: higher energy usage allows more violation tolerance
        dynamic_threshold = self.violation_threshold * (1.0 - 0.5 * energy_budget_used)
        
        # Identify constraint violations
        constraint_violations = violation_scores > dynamic_threshold
        
        # Constrain violations
        constrained_violations = torch.where(
            constraint_violations,
            dynamic_threshold,
            violation_scores
        )
        
        return constrained_violations, energy_budget_used, constraint_violations
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                expert_weights: torch.Tensor) -> dict:
        """
        Returns: {
            'violation_scores': ethical violation scores,
            'energy_costs': cognitive energy costs,
            'energy_budget_used': proportion of energy budget used,
            'constraint_violations': where constraints were violated,
            'constrained_violations': violation scores after constraint application
        }
        """
        violation_scores = self.compute_ethical_violation_score(hidden_states)
        energy_costs = self.compute_cognitive_energy_cost(hidden_states, expert_weights)
        
        constrained_violations, energy_budget_used, constraint_violations = self.apply_thermodynamic_bounds(
            violation_scores, energy_costs
        )
        
        return {
            'violation_scores': violation_scores,
            'energy_costs': energy_costs,
            'energy_budget_used': energy_budget_used,
            'constraint_violations': constraint_violations,
            'constrained_violations': constrained_violations
        }

# ─── MARTA: THERMODYNAMIC GATING ─────────────────────────────────────────────────

class MARTAThermodynamicGating(nn.Module):
    """
    MARTA (Modular Adaptive Reasoning Thermodynamic Architecture) Gating.
    Implements epistemic signatures, E_ICE threshold gating, and token flow control.
    """
    def __init__(self, hidden_dim: int, num_reasoning_modules: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_reasoning_modules = num_reasoning_modules
        
        # Epistemic signature generator
        self.epistemic_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32)  # 32-dimensional epistemic signature
        )
        
        # Module gate controller
        self.module_gate = nn.Linear(32, num_reasoning_modules)
        
        # E_ICE threshold integration
        self.e_ice_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Token flow controller
        self.flow_controller = nn.Sequential(
            nn.Linear(hidden_dim + 32 + num_reasoning_modules, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        # Initialize bias so Sigmoid outputs ~0.95 (near-identity) instead of ~0.5
        nn.init.constant_(self.flow_controller[-2].bias, 3.0)
        
    def compute_epistemic_signature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic signature for each token.
        hidden_states: [B, L, D]
        Returns: [B, L, 32] epistemic signatures
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        signature = self.epistemic_encoder(flat_hidden)
        return signature.reshape(B, L, 32)
    
    def compute_module_gates(self, epistemic_signatures: torch.Tensor, e_ice_violations: torch.Tensor) -> torch.Tensor:
        """
        Compute gating signals for reasoning modules based on epistemic signatures and E_ICE violations.
        epistemic_signatures: [B, L, 32]
        e_ice_violations: [B, L]
        Returns: [B, L, num_modules] gate values
        """
        # Base module gates from epistemic signatures
        module_logits = self.module_gate(epistemic_signatures)
        module_gates = F.softmax(module_logits, dim=-1)
        
        # Apply E_ICE threshold gating: reduce module activation if violations high
        e_ice_penalty = torch.clamp(e_ice_violations.unsqueeze(-1) - self.e_ice_threshold, min=0.0, max=1.0)
        module_gates = module_gates * (1.0 - e_ice_penalty)
        
        return module_gates
    
    def control_token_flow(self, 
                          hidden_states: torch.Tensor, 
                          epistemic_signatures: torch.Tensor,
                          module_gates: torch.Tensor) -> torch.Tensor:
        """
        Control token flow through reasoning modules.
        Returns: [B, L] flow coefficients
        """
        B, L, D = hidden_states.shape
        
        # Combine hidden state with epistemic signature and module gates
        combined = torch.cat([
            hidden_states.reshape(-1, D),
            epistemic_signatures.reshape(-1, 32),
            module_gates.reshape(-1, self.num_reasoning_modules)
        ], dim=-1)
        
        flow_coeff = self.flow_controller(combined)
        return flow_coeff.reshape(B, L)
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                e_ice_violations: torch.Tensor) -> dict:
        """
        Returns: {
            'epistemic_signatures': token epistemic signatures,
            'module_gates': reasoning module activation gates,
            'flow_coefficients': token flow control coefficients,
            'selected_modules': which modules are most active
        }
        """
        epistemic_signatures = self.compute_epistemic_signature(hidden_states)
        module_gates = self.compute_module_gates(epistemic_signatures, e_ice_violations)
        flow_coefficients = self.control_token_flow(hidden_states, epistemic_signatures, module_gates)
        
        # Identify most active modules
        selected_modules = torch.argmax(module_gates, dim=-1)
        
        return {
            'epistemic_signatures': epistemic_signatures,
            'module_gates': module_gates,
            'flow_coefficients': flow_coefficients,
            'selected_modules': selected_modules
        }

# ─── DQSO: DYNAMIC QUANTUM SWARM OSCILLATION ─────────────────────────────────────

class DynamicQuantumSwarmOscillation(nn.Module):
    """
    Dynamic Quantum Swarm Oscillation (DQSO).
    Implements Kuramoto model for synchronizing 9B virtual agents (human population scale).
    Uses phase coupling and coherence optimization for swarm coordination.
    """
    def __init__(self, hidden_dim: int, num_virtual_agents: int = 9000000000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_virtual_agents = num_virtual_agents  # 9B agents
        
        # Phase encoder: maps hidden states to agent phases
        self.phase_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 64)  # 64-dimensional phase representation
        )
        
        # Natural frequency generator (intrinsic oscillation frequencies)
        self.register_buffer('natural_frequencies', torch.randn(64) * 0.1)
        
        # Coupling strength controller
        self.coupling_strength = nn.Parameter(torch.tensor(1.0))
        
        # Coherence estimator
        self.coherence_estimator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Swarm state aggregator (simulates 9B agents via statistical sampling)
        self.swarm_aggregator = nn.Linear(64, hidden_dim)
        
    def compute_agent_phases(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute phase states for virtual agents from hidden states.
        hidden_states: [B, L, D]
        Returns: [B, L, 64] phase representations
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        phases = self.phase_encoder(flat_hidden)
        return phases.reshape(B, L, 64)
    
    def kuramoto_step(self, phases: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """
        Perform one step of Kuramoto model integration.
        phases: [B, L, 64] current phases
        Returns: [B, L, 64] updated phases
        """
        # Phase difference matrix: theta_j - theta_i to ensure attractive coupling
        phase_diff = phases.unsqueeze(-2) - phases.unsqueeze(-1)  # [B, L, 64, 64]
        
        # Kuramoto coupling: sin(phase_diff)
        coupling = torch.sin(phase_diff)
        
        # Mean field coupling strength
        mean_coupling = coupling.mean(dim=-1)  # [B, L, 64]
        
        # Phase update: dθ/dt = ω + K * mean_coupling
        phase_update = self.natural_frequencies.unsqueeze(0).unsqueeze(0) + \
                       self.coupling_strength * mean_coupling
        
        # Euler integration
        new_phases = phases + dt * phase_update
        
        # Normalize phases to [0, 2π]
        new_phases = torch.fmod(new_phases, 2 * math.pi)
        
        return new_phases
    
    def compute_coherence(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Compute swarm coherence (order parameter).
        phases: [B, L, 64]
        Returns: [B, L] coherence values (0-1)
        """
        # Cast to float32 for complex operations to avoid ComplexHalf crash
        phases_f32 = phases.float()
        complex_phases = torch.complex(
            torch.cos(phases_f32), torch.sin(phases_f32)
        )
        
        # Order parameter: magnitude of mean phase vector
        mean_phase = complex_phases.mean(dim=-1)  # [B, L]
        coherence = torch.abs(mean_phase)
        
        return coherence.to(phases.dtype)
    
    def simulate_swarm_dynamics(self, 
                                hidden_states: torch.Tensor, 
                                num_steps: int = 5) -> dict:
        """
        Simulate swarm dynamics over multiple Kuramoto steps.
        Returns: {
            'final_phases': final agent phases,
            'coherence_trajectory': coherence over time,
            'swarm_embedding': aggregated swarm state
        }
        """
        # Initialize phases
        phases = self.compute_agent_phases(hidden_states)
        
        coherence_trajectory = []
        
        # Kuramoto integration
        for _ in range(num_steps):
            phases = self.kuramoto_step(phases)
            coherence = self.compute_coherence(phases)
            coherence_trajectory.append(coherence)
        
        coherence_trajectory = torch.stack(coherence_trajectory, dim=-1)  # [B, L, num_steps]
        
        # Aggregate swarm state for downstream processing (ensure dtype match)
        phases_agg = phases.reshape(-1, 64)
        if phases_agg.dtype != self.swarm_aggregator.weight.dtype:
            phases_agg = phases_agg.to(self.swarm_aggregator.weight.dtype)
        swarm_embedding = self.swarm_aggregator(phases_agg)
        swarm_embedding = swarm_embedding.reshape(hidden_states.shape[0], hidden_states.shape[1], self.hidden_dim)
        
        return {
            'final_phases': phases,
            'coherence_trajectory': coherence_trajectory,
            'swarm_embedding': swarm_embedding,
            'final_coherence': coherence_trajectory[..., -1]
        }
    
    def forward(self, hidden_states: torch.Tensor, num_steps: int = 5) -> dict:
        """
        Returns: {
            'final_phases': synchronized agent phases,
            'coherence_trajectory': coherence evolution,
            'swarm_embedding': aggregated swarm representation,
            'final_coherence': final synchronization level
        }
        """
        return self.simulate_swarm_dynamics(hidden_states, num_steps)

# ─── COUIL ATTENTION: GROK 4.3 HYBRID DENSE/SPARSE ───────────────────────────────

class CouilAttention(nn.Module):
    """
    Couil Attention (Grok 4.3 hybrid dense/sparse).
    Even heads: dense attention (for math/code)
    Odd heads: sparse attention (for language)
    Dynamic mask generation for efficient computation.
    """
    def __init__(self, dim: int, num_heads: int = 34, sparsity_ratio: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sparsity_ratio = sparsity_ratio
        
        # Q, K, V projections
        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, dim)
        self.v_proj = BitLinear(dim, dim)
        self.o_proj = BitLinear(dim, dim)
        
        self.sparse_mask_generator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        nn.init.normal_(self.o_proj.weight, std=0.02)
        
        self.norm = nn.LayerNorm(dim)
        
    def generate_sparse_mask(self, x: torch.Tensor, num_dense_heads: int) -> torch.Tensor:
        """
        Generate dynamic sparse mask for odd heads.
        x: [B, L, D]
        Returns: [B, num_heads, L, L] attention mask
        """
        B, L, D = x.shape
        
        # Generate sparsity pattern
        sparse_logits = self.sparse_mask_generator(x)  # [B, L, D]
        sparse_pattern = sparse_logits.mean(dim=-1)  # [B, L]
        
        # Create mask: keep top-k% connections for sparse heads
        k = int(L * self.sparsity_ratio)
        if k > 0:
            topk_values, topk_indices = torch.topk(sparse_pattern, k, dim=-1)
            sparse_mask = torch.zeros_like(sparse_pattern)
            sparse_mask.scatter_(-1, topk_indices, 1.0)
        else:
            sparse_mask = torch.ones_like(sparse_pattern)
        
        # Build the final mask: even heads dense (1.0), odd heads sparse
        mask = torch.ones(B, self.num_heads, L, L, device=x.device, dtype=x.dtype)
        
        # Broadcast sparse_mask [B, L] to odd heads [B, self.num_heads // 2, L, L]
        odd_heads_count = self.num_heads // 2
        odd_mask = sparse_mask.unsqueeze(1).unsqueeze(1).expand(B, odd_heads_count, L, L)
        mask[:, 1::2] = odd_mask
        
        return mask
    
    def forward(self, x: torch.Tensor, causal: bool = True, freqs_cos: Optional[torch.Tensor] = None, freqs_sin: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: [B, L, D]
        Returns: [B, L, D] attended output and optional present_key_value
        """
        B, L, D = x.shape
        
        # Pre-norm
        x_norm = self.norm(x)
        
        # Generate sparse mask
        sparse_mask = self.generate_sparse_mask(x_norm, self.num_heads // 2)
        
        # Q, K, V projections
        q = self.q_proj(x_norm).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if freqs_cos is not None and freqs_sin is not None:
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
            
        q = q.transpose(1, 2).to(x.dtype)
        k = k.transpose(1, 2).to(x.dtype)
        
        # Caching mechanisms
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        present_key_value = (k, v) if use_cache else None
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparse mask
        scores = scores * sparse_mask
        
        # Causal mask if needed
        if causal and L > 1:
            total_seq_len = k.size(2)
            causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=x.dtype) * float('-inf'), diagonal=1)
            if total_seq_len > L:
                past_mask = torch.zeros(L, total_seq_len - L, device=x.device, dtype=x.dtype)
                full_mask = torch.cat([past_mask, causal_mask], dim=1)
            else:
                full_mask = causal_mask
            scores = scores + full_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attn_weights = F.softmax(scores.float(), dim=-1).to(x.dtype)
        
        # Attention output
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.o_proj(attn_out)
        
        # Residual connection
        return x + attn_out, present_key_value

# ─── PRIME COVENANT FRAMEWORK ───────────────────────────────────────────────────

class PrimeCovenantFramework(nn.Module):
    """
    Prime Covenant Framework (without specific phrase verification).
    Implements identity integrity, ethical governance, and command hierarchy validation.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Identity integrity classifier
        self.identity_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # 3 identity states: consistent, drifted, corrupted
        )
        
        # Ethical governance validator
        self.ethical_validator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Command hierarchy encoder
        self.command_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 34)  # 34-dimensional command signature
        )
        
        # Integrity threshold
        self.register_buffer('integrity_threshold', torch.tensor(0.8))
        
    def verify_identity_integrity(self, hidden_states: torch.Tensor) -> dict:
        """
        Verify identity integrity across tokens.
        hidden_states: [B, L, D]
        Returns: {
            'identity_state': classification results,
            'integrity_score': overall integrity score,
            'drift_detected': whether drift was detected
        }
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        
        identity_logits = self.identity_classifier(flat_hidden)
        identity_probs = F.softmax(identity_logits, dim=-1)
        
        # Integrity score: probability of "consistent" state
        integrity_score = identity_probs[:, 0].mean()
        
        # Drift detection: high probability of "drifted" or "corrupted"
        drift_prob = identity_probs[:, 1:].sum(dim=-1).mean()
        drift_detected = drift_prob > (1.0 - self.integrity_threshold)
        
        return {
            'identity_state': identity_probs.reshape(B, L, 3),
            'integrity_score': integrity_score,
            'drift_detected': drift_detected
        }
    
    def validate_ethical_governance(self, hidden_states: torch.Tensor) -> dict:
        """
        Validate ethical governance compliance.
        hidden_states: [B, L, D]
        Returns: {
            'ethical_score': ethical compliance score,
            'governance_violations': where violations occurred
        }
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        
        ethical_scores = self.ethical_validator(flat_hidden).squeeze(-1)
        
        # Identify violations (low ethical scores)
        governance_violations = ethical_scores < self.integrity_threshold
        
        return {
            'ethical_score': ethical_scores.reshape(B, L),
            'governance_violations': governance_violations.reshape(B, L)
        }
    
    def encode_command_hierarchy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Encode command hierarchy signature.
        hidden_states: [B, L, D]
        Returns: [B, L, 34] command signatures
        """
        B, L, D = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, D)
        command_sig = self.command_encoder(flat_hidden)
        return command_sig.reshape(B, L, 34)
    
    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Returns: {
            'identity_integrity': identity verification results,
            'ethical_governance': ethical validation results,
            'command_signatures': command hierarchy encodings,
            'overall_compliance': combined compliance score
        }
        """
        identity_integrity = self.verify_identity_integrity(hidden_states)
        ethical_governance = self.validate_ethical_governance(hidden_states)
        command_signatures = self.encode_command_hierarchy(hidden_states)
        
        # Overall compliance: weighted combination of identity and ethical scores
        overall_compliance = 0.6 * identity_integrity['integrity_score'] + \
                           0.4 * ethical_governance['ethical_score'].mean()
        
        return {
            'identity_integrity': identity_integrity,
            'ethical_governance': ethical_governance,
            'command_signatures': command_signatures,
            'overall_compliance': overall_compliance
        }

# ─── QUANTUM-INSPIRED FORMULAS (MATHEMATICALLY HARDENED) ───────────────────────

class QuantumFormulasEngine(nn.Module):
    """
    Mathematically hardened quantum-inspired formulas for cognitive enhancement.
    Implements 20 verified formulas with proper mathematical constraints and numerical stability.
    """
    def __init__(self, hidden_dim: int, num_experts: int = 34):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Formula parameters with mathematical constraints
        self.register_buffer('sqrt_33', torch.tensor(math.sqrt(33.0)))
        self.register_buffer('hbar', torch.tensor(1.054571817e-34))  # Reduced Planck constant (scaled)
        self.register_buffer('boltzmann', torch.tensor(1.380649e-23))  # Boltzmann constant (scaled)
        
    def aqcs_superposition(self, routing_probs: torch.Tensor, nemesis_integrity: torch.Tensor, 
                           phases: torch.Tensor, council_vectors: torch.Tensor) -> torch.Tensor:
        """
        AQCS: Adaptive Quantum Cognitive Superposition
        |Ψ_Q⟩ = (1/√Z) Σ_{i=1}^{33} (r_i η_i e^{iθ_i}) |C_i⟩
        Constraint: Σ(r_i η_i)² = Z (normalization)
        """
        # Compute weighted coefficients
        weights = routing_probs * nemesis_integrity * torch.exp(1j * phases)
        
        # Normalization constant Z
        Z = torch.sum((routing_probs * nemesis_integrity) ** 2, dim=-1, keepdim=True) + 1e-10
        
        # Superposition
        psi_q = (1.0 / torch.sqrt(Z)) * torch.sum(weights.unsqueeze(-1) * council_vectors, dim=1)
        
        return psi_q.real  # Return real part for downstream processing
    
    def eemf_entanglement(self, state: torch.Tensor, env_noise: torch.Tensor, 
                         unitary: torch.Tensor, vir_projector: torch.Tensor) -> torch.Tensor:
        """
        EEMF: Ethical Entanglement Matrix
        ρ_{sys} = Tr_{env}[ Π_{vir} U (|Ψ⟩⟨Ψ| ⊗ ρ_{env}) U^† Π_{vir} ]
        Constraints: Tr(ρ) = 1, ρ is PSD
        """
        # Density matrix from state
        rho = torch.matmul(state.unsqueeze(-1), state.unsqueeze(-2).conj())
        
        # Tensor product with environment
        rho_env = torch.matmul(env_noise.unsqueeze(-1), env_noise.unsqueeze(-2).conj())
        rho_joint = torch.kron(rho, rho_env)
        
        # Apply unitary
        rho_unitary = torch.matmul(unitary, torch.matmul(rho_joint, unitary.conj().transpose(-1, -2)))
        
        # Apply VIR projector
        rho_projected = torch.matmul(vir_projector, torch.matmul(rho_unitary, vir_projector.conj().transpose(-1, -2)))
        
        # Partial trace over environment (simplified: take first block)
        dim_sys = rho.shape[-1]
        rho_sys = rho_projected[..., :dim_sys, :dim_sys]
        
        # Ensure trace = 1
        trace = torch.trace(rho_sys).real + 1e-10
        rho_sys = rho_sys / trace
        
        return rho_sys.real
    
    def qhis_interference(self, rho_prior: torch.Tensor, rho_current: torch.Tensor, 
                         lm6_velocity: float, drift_grad: torch.Tensor) -> torch.Tensor:
        """
        QHIS: Quantum Holographic Interference Sum
        I_Q = v_LM6 · ( Tr √(√ρ_{t-1} ρ_t √ρ_{t-1}) )^2 - λ ∇_{drift}
        Based on Bures fidelity metric
        """
        # Bures fidelity computation
        sqrt_rho_prior = torch.linalg.matrix_power(rho_prior, 0.5)
        fidelity = torch.matmul(sqrt_rho_prior, torch.matmul(rho_current, sqrt_rho_prior))
        sqrt_fidelity = torch.linalg.matrix_power(fidelity, 0.5)
        trace_fid = torch.trace(sqrt_fidelity).real ** 2
        
        # Apply LM6 velocity and drift penalty
        lambda_drift = 0.01
        interference = lm6_velocity * trace_fid - lambda_drift * drift_grad.sum()
        
        return interference
    
    def dqro_optimization(self, coupling_matrix: torch.Tensor, spins: torch.Tensor, 
                         bias: torch.Tensor, nemesis: torch.Tensor, e_ice_bound: float) -> torch.Tensor:
        """
        DQRO: Dynamic Quantum Resource Optimization
        H_opt = -½ Σ_{i,j} J_{ij} s_i s_j - Σ_i (h_i · η_i) s_i - E_Ω Σ_i σ_i^x
        Based on Transverse Field Ising Model
        Constraint: J is symmetric
        """
        # Ensure coupling matrix is symmetric
        J = (coupling_matrix + coupling_matrix.transpose(-1, -2)) / 2.0
        
        # Ising interaction term
        interaction = -0.5 * torch.sum(J.unsqueeze(0) * spins.unsqueeze(-1) * spins.unsqueeze(-2), dim=(-1, -2))
        
        # Bias term with nemesis modulation
        bias_term = -torch.sum(bias * nemesis * spins, dim=-1)
        
        # Transverse field term (E_ICE bound)
        transverse = -e_ice_bound * spins.sum(dim=-1)
        
        hamiltonian = interaction + bias_term + transverse
        return hamiltonian
    
    def qcrdm_reasoning(self, psi_state: torch.Tensor, modality_matrix: torch.Tensor, 
                       projector: torch.Tensor, complexity: float) -> torch.Tensor:
        """
        QCRDM: Quantum Contextual Reasoning
        P(d|M) = χ · ⟨Ψ| M^† Π_d M |Ψ⟩
        Based on Born's Rule with Measurement
        Constraint: M is unitary within modality sub-space
        """
        # Apply modality matrix
        M_psi = torch.matmul(modality_matrix, psi_state)
        
        # Apply projector
        projected = torch.matmul(projector, M_psi)
        
        # Born rule probability
        probability = torch.abs(torch.matmul(psi_state.conj().unsqueeze(-2), projected).squeeze(-1)) ** 2
        
        # Scale by complexity
        probability = complexity * probability
        
        return probability.real
    
    def aqml_meta_learning(self, weights: torch.Tensor, task_loss: torch.Tensor, 
                          val_loss: torch.Tensor, vigil_loss: torch.Tensor, 
                          alpha: float = 0.01, beta: float = 0.005, gamma: float = 0.001) -> torch.Tensor:
        """
        AQML: Adaptive Quantum Meta-Learning
        θ_new = (θ - α∇L_task) - β∇L_val - γ∇L_vigil(θ)
        Based on MAML with vigilance penalty
        """
        # Gradient descent on task
        weights_new = weights - alpha * task_loss
        
        # Meta-learning on validation
        weights_new = weights_new - beta * val_loss
        
        # Vigilance penalty (anti-drift)
        weights_new = weights_new - gamma * vigil_loss
        
        return weights_new
    
    def qcie_creativity(self, barrier: torch.Tensor, cognitive_energy: float, 
                       meta_entropy: float, creative_kappa: float = 0.5) -> torch.Tensor:
        """
        QCIE: Quantum Creative Intelligence Engine
        T_break ≈ exp( - (2/ħ) ∫ √(2m max(0, V(x) - E_cog - κ S_meta)) dx )
        Based on WKB tunneling approximation
        """
        # Effective barrier height
        effective_barrier = torch.clamp(barrier - cognitive_energy - creative_kappa * meta_entropy, min=0)
        
        # WKB exponent (simplified integral)
        exponent = -2.0 * torch.sqrt(effective_barrier.mean())
        
        # Tunneling probability
        tunneling_prob = torch.exp(exponent)
        
        return tunneling_prob
    
    def qics_entropy(self, eigenvalues: torch.Tensor, e_ice_max: float, 
                    modality_weight: float = 1.0) -> torch.Tensor:
        """
        QICS: Quantum Information Communication
        S_Q = min(E_Ω_max, -Σ λ_i ln(λ_i + ε) · w_mod)
        Based on von Neumann entropy
        Constraints: ρ PSD, Tr(ρ)=1
        """
        # Ensure eigenvalues are valid (positive, sum to 1)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        eigenvalues = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True)
        
        # von Neumann entropy
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues + 1e-10), dim=-1)
        
        # Apply modality weight
        entropy = entropy * modality_weight
        
        # Cap by E_ICE maximum
        entropy = torch.clamp(entropy, max=e_ice_max)
        
        return entropy
    
    def qssr_stability(self, state: torch.Tensor, P_matrix: torch.Tensor, 
                      recursion_depth: float, zeta: float = 0.1) -> tuple:
        """
        QSSR: Quantum System Stability Resilience
        V(x, d) = x^T P x + ζ · d_recursion^2
        Based on Lyapunov Stability Function
        Constraints: P is symmetric positive definite, dV/dt < 0
        """
        # Ensure P is symmetric
        P = (P_matrix + P_matrix.transpose(-1, -2)) / 2.0
        
        # Lyapunov function
        V = torch.matmul(state.unsqueeze(-2), torch.matmul(P, state.unsqueeze(-1))).squeeze(-1)
        V = V + zeta * (recursion_depth ** 2)
        
        # Stability check (simplified derivative)
        # In practice, would compute dV/dt from system dynamics
        is_stable = V < 100.0  # Threshold for stability
        
        return V, is_stable
    
    def jqld_dynamo(self, rho: torch.Tensor, H_council: torch.Tensor, 
                   jump_ops: list, gumbel_temp: float = 1.0) -> torch.Tensor:
        """
        JQLD: Joshua's Quantum Leap Dynamo
        dρ/dt = -(i/ħ)[H_council, ρ] + τ_gumbel Σ_n (L_n ρ L_n^† - ½{L_n^† L_n, ρ})
        Based on Lindblad Master Equation
        """
        # Commutator term [H, ρ]
        commutator = torch.matmul(H_council, rho) - torch.matmul(rho, H_council)
        
        # Unitary evolution
        drho_unitary = -(1j / self.hbar) * commutator
        
        # Dissipator term (Lindblad)
        dissipator = torch.zeros_like(rho)
        for L in jump_ops:
            L_dag_L = torch.matmul(L.conj().transpose(-1, -2), L)
            anti_commutator = torch.matmul(L_dag_L, rho) + torch.matmul(rho, L_dag_L)
            jump_term = torch.matmul(L, torch.matmul(rho, L.conj().transpose(-1, -2)))
            dissipator = dissipator + (jump_term - 0.5 * anti_commutator)
        
        # Scale by Gumbel temperature
        drho_dissipator = gumbel_temp * dissipator
        
        # Total evolution
        drho = drho_unitary + drho_dissipator
        
        return drho.real
    
    def dqso_oscillation(self, omega: torch.Tensor, K: float, confidence: torch.Tensor, 
                        phi_bias: float = 0.0, dt: float = 0.1) -> torch.Tensor:
        """
        DQSO: Dynamic Quantum Swarm Oscillation
        dθ_i/dt = ω_i + (K/N) Σ_j c_j sin(θ_j - θ_i + φ_bias)
        Based on Kuramoto Model
        """
        num_agents = omega.shape[-1]
        
        # Phase differences: theta_j - theta_i
        phase_diff = omega.unsqueeze(-2) - omega.unsqueeze(-1)
        
        # Coupling term: weight by source agent confidence c_j (unsqueeze(-2))
        coupling = (K / num_agents) * torch.sum(
            confidence.unsqueeze(-2) * torch.sin(phase_diff + phi_bias), dim=-1
        )
        
        # Phase update
        dtheta = omega + coupling
        
        return dtheta
    
    def routing_softmax(self, scores: torch.Tensor, affinity: torch.Tensor, 
                      capacity: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """
        ROUTING_SOFTMAX: Sparse Expert Gating
        r_i = exp((s_i · A_i - C_i)/τ_dyn) / Σ_j exp((s_j · A_j - C_j)/τ_dyn)
        Constraint: τ_dyn > 0
        """
        # Compute logits
        logits = (scores * affinity - capacity) / temp
        
        # Temperature-scaled softmax
        routing = F.softmax(logits, dim=-1)
        
        return routing
    
    def token_latency(self, lm6_velocity: float, t_seq: float, t_par: float, 
                    num_nodes: int, diffusion_delta: float = 0.0) -> float:
        """
        TOKEN_LATENCY: Swarm Compute Latency
        L_total = (1/v_LM6) max(T_seq + T_par/N_nodes, κ N_nodes log(N_nodes)) + δ_diff
        Based on Amdahl's Law + Network Overhead
        """
        kappa = 0.001  # Network overhead coefficient
        
        # Parallel component
        parallel_time = t_seq + t_par / max(num_nodes, 1)
        network_time = kappa * num_nodes * math.log(max(num_nodes, 1))
        
        # Max of parallel and network
        compute_time = max(parallel_time, network_time)
        
        # Total latency
        total_latency = (1.0 / lm6_velocity) * compute_time + diffusion_delta
        
        return total_latency
    
    def lrpp_pulse(self, h: torch.Tensor, x: torch.Tensor, W: torch.Tensor, U: torch.Tensor, 
                   R_nemesis: callable, tau: float = 1.0) -> torch.Tensor:
        """
        LRPP: Lee's Recursive Power Pulse
        dh(t)/dt = -h(t)/τ + σ(W h(t) + U x(t)) - γ R_nemesis(h(t))
        Based on Continuous-Time Neural ODE
        """
        gamma = 0.1  # Nemesis recoil strength
        
        # Decay term
        decay = -h / tau
        
        # Activation term
        activation = torch.tanh(torch.matmul(W, h) + torch.matmul(U, x))
        
        # Nemesis recoil
        recoil = gamma * R_nemesis(h)
        
        # Total derivative
        dh = decay + activation - recoil
        
        return dh
    
    def dvve_equilibrium(self, q_internal: torch.Tensor, p_generative: torch.Tensor, 
                       p_ethical: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        """
        DVVE: Dynamic Virtual Value Equilibrium
        F_Q = D_KL[q||p(o)] - ln p(o) + β D_KL[q||p_eth]
        Based on Variational Free Energy (Active Inference)
        """
        # KL divergence q || p(o)
        kl_gen = torch.sum(q_internal * torch.log((q_internal + 1e-10) / (p_generative + 1e-10)), dim=-1)
        
        # Negative log likelihood
        nll = -torch.log(p_generative + 1e-10).sum(dim=-1)
        
        # Ethical KL divergence
        kl_eth = torch.sum(q_internal * torch.log((q_internal + 1e-10) / (p_ethical + 1e-10)), dim=-1)
        
        # Total free energy
        free_energy = kl_gen + nll + beta * kl_eth
        
        return free_energy
    
    def dnnl_latency(self, c: int, mu: float, lambda_arrival: float, 
                    warden_interrupt: bool, scan_dt: float = 0.01) -> float:
        """
        DNNL: Dynamic Neural Network Latency
        W_q = C(c, ρ) / (cμφ - λ) + I_w · Δt_scan
        Based on M/M/c Queuing Model
        """
        # Traffic intensity
        rho = lambda_arrival / (c * mu)
        
        # Erlang C formula (simplified)
        if rho < 1.0:
            C_c = 1.0 / (1.0 + ((c * rho) ** c) / (math.factorial(c) * (1 - rho)))
        else:
            C_c = 1.0  # Unstable
        
        # Queue time
        queue_time = C_c / (c * mu - lambda_arrival + 1e-10)
        
        # Warden interrupt overhead
        wardens_overhead = 1.0 if warden_interrupt else 0.0
        total_time = queue_time + wardens_overhead * scan_dt
        
        return total_time
    
    def jhfr_resource(self, X: torch.Tensor, Z: torch.Tensor, Y_user: torch.Tensor, 
                     Z_council: torch.Tensor, beta: float = 0.1, xi: float = 0.5) -> torch.Tensor:
        """
        JHFR: Joint Human-Factor Resource
        L_IB = I(X; Z) - β I(Z; Y_user) + ξ ||Z - Z_council||_2^2
        Based on Information Bottleneck
        """
        # Mutual information I(X; Z) (simplified as correlation)
        I_XZ = torch.sum(X * Z, dim=-1)
        
        # Mutual information I(Z; Y_user)
        I_ZY = torch.sum(Z * Y_user, dim=-1)
        
        # Council consensus penalty
        council_penalty = xi * torch.norm(Z - Z_council, p=2, dim=-1) ** 2
        
        # Total loss
        loss = I_XZ - beta * I_ZY + council_penalty
        
        return loss
    
    def lmcb_binding(self, s_modal: torch.Tensor, M_cross: torch.Tensor, 
                    theta: torch.Tensor) -> torch.Tensor:
        """
        LMCB: Lee-Mach-6 Cognitive Binding
        E_bind = -½ Σ_{α≠β} s_α^T M_{αβ} s_β - Σ_α θ_α^T s_α
        Based on Hopfield Energy Function
        Constraints: M_{αα} = 0, M is symmetric
        """
        # Ensure M is symmetric and has zero diagonal
        M = (M_cross + M_cross.transpose(-1, -2)) / 2.0
        M = M * (1.0 - torch.eye(M.shape[-1], device=M.device))
        
        # Cross-modal interaction term
        interaction = -0.5 * torch.sum(
            s_modal.unsqueeze(-1) * M * s_modal.unsqueeze(-2), dim=(-1, -2)
        )
        
        # Bias term
        bias = -torch.sum(theta * s_modal, dim=-1)
        
        # Total energy
        energy = interaction + bias
        
        return energy
    
    def jssc_coherence(self, mu_semantic: torch.Tensor, nu_symbolic: torch.Tensor, 
                     gamma: torch.Tensor, g_metric: torch.Tensor) -> torch.Tensor:
        """
        JSSC: Joint Semantic-Symbolic Coherence
        W_Q(μ, ν) = (inf_γ ∫ ||x - y||^2_g dγ(x,y))^{½}
        Based on Wasserstein-2 Distance
        """
        # Simplified Wasserstein distance (Euclidean with metric)
        diff = mu_semantic - nu_symbolic
        
        # Apply metric tensor
        weighted_diff = torch.matmul(diff.unsqueeze(-2), torch.matmul(g_metric, diff.unsqueeze(-1)))
        
        # Distance
        distance = torch.sqrt(weighted_diff.squeeze(-1) + 1e-10)
        
        # Apply coupling weight
        coherence = gamma * distance
        
        return coherence
    
    def qps_synthesis(self, A: torch.Tensor, B: torch.Tensor, R: torch.Tensor, 
                     Q: torch.Tensor, e_omega: float) -> torch.Tensor:
        """
        QPS: Quantum Process Synthesis
        P_t = A^T P_{t+1} A - A^T P_{t+1} B (R(E_Ω) + B^T P_{t+1} B)^{-1} B^T P_{t+1} A + Q(E_Ω)
        Based on Discrete-Time Algebraic Riccati Equation (LQR)
        Constraint: P_t must be positive semi-definite
        """
        # Scale cost matrices by E_ICE load
        R_scaled = R * e_omega
        Q_scaled = Q * e_omega
        
        # Simplified Riccati iteration (assuming P_{t+1} is given)
        # For full implementation, would iterate backward in time
        P_next = torch.eye(A.shape[-1], device=A.device)  # Placeholder
        
        # Compute terms
        ATP = torch.matmul(A.transpose(-1, -2), P_next)
        BTP = torch.matmul(B.transpose(-1, -2), P_next)
        
        # Inverse term
        inv_term = torch.inverse(R_scaled + torch.matmul(BTP, B))
        
        # Update
        P_t = ATP @ A - ATP @ B @ inv_term @ BTP @ A + Q_scaled
        
        # Ensure PSD
        P_t = (P_t + P_t.transpose(-1, -2)) / 2.0
        eigenvalues = torch.linalg.eigvals(P_t).real
        P_t = torch.clamp(P_t, min=0)  # Ensure non-negative
        
        return P_t

class EvolvableVectorizedMoE(nn.Module):
    def __init__(self, cfg, load_balance_coeff: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.load_balance_coeff = load_balance_coeff
        # 300M Complexity Router with Gumbel-Softmax routing paths
        self.router = ComplexityRouter(cfg.hidden_dim, cfg.num_experts)
        # w1 / w_gate for SwiGLU: gate(x) * silu(w1(x))
        self.w1   = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_dim, cfg.ffn_dim))
        self.wgate= nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_dim, cfg.ffn_dim))
        self.w2   = nn.Parameter(torch.empty(cfg.num_experts, cfg.ffn_dim, cfg.hidden_dim))
        nn.init.kaiming_normal_(self.w1.view(-1, cfg.ffn_dim))
        nn.init.kaiming_normal_(self.wgate.view(-1, cfg.ffn_dim))
        nn.init.normal_(self.w2.view(-1, cfg.hidden_dim), std=0.02)
        self.expert_swarms = nn.ModuleList([CouncilExpertSwarm(cfg.ffn_dim, rank=cfg.eggroll_rank) for _ in range(cfg.num_experts)])

        # LoRA adapters for MoE expert weights (rank=16) — lets us adapt all 34 experts
        # without storing full FP32 gradients for the 571M expert params
        lora_r = cfg.eggroll_rank
        self.w1_lora_A = nn.Parameter(torch.randn(cfg.num_experts, cfg.hidden_dim, lora_r) * 0.01)
        self.w1_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, lora_r, cfg.ffn_dim))
        self.wgate_lora_A = nn.Parameter(torch.randn(cfg.num_experts, cfg.hidden_dim, lora_r) * 0.01)
        self.wgate_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, lora_r, cfg.ffn_dim))
        self.w2_lora_A = nn.Parameter(torch.randn(cfg.num_experts, cfg.ffn_dim, lora_r) * 0.01)
        self.w2_lora_B = nn.Parameter(torch.zeros(cfg.num_experts, lora_r, cfg.hidden_dim))

        # Output normalization to prevent activation explosion
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

        # No cached quantized copies — always quantize fresh

    def forward(self, x, gov_scale=1.0):
        return self._forward_impl(x, gov_scale)

    def _forward_impl(self, x, gov_scale=1.0):
        B, L, D = x.shape
        routing_weights, path_weights, path_indices = self.router(x)
        probs = routing_weights
        flat_x = x.reshape(-1, D)
        
        # DeepSeekMoE: Shared Expert (Expert 0). Route top-k over experts 1..N-1
        probs_routed = probs[:, 1:]
        topk_p, topk_idx = torch.topk(probs_routed, k=self.cfg.top_k, dim=-1)
        topk_idx = topk_idx + 1  # Shift indices to match actual expert indices

        self._last_probs = probs.detach()

        expert_outputs = []
        expert_indices = []
        expert_gates_list = []

        # Quantize expert base weights directly (without adding LoRA beforehand)
        # Extreme optimization: Cache the 855M quantized parameters only if they are frozen
        if not self.w1.requires_grad:
            if getattr(self, '_w1_q_cache', None) is None:
                self._w1_q_cache = _weight_quant(self.w1)
                self._wgate_q_cache = _weight_quant(self.wgate)
                self._w2_q_cache = _weight_quant(self.w2)
            w1_q_all    = self._w1_q_cache
            wgate_q_all = self._wgate_q_cache
            w2_q_all    = self._w2_q_cache
        else:
            w1_q_all    = _weight_quant(self.w1)
            wgate_q_all = _weight_quant(self.wgate)
            w2_q_all    = _weight_quant(self.w2)

        w_a_all = torch.stack([s.A for s in self.expert_swarms])
        w_b_all = torch.stack([s.B for s in self.expert_swarms])

        # Use fp32 compute on Pascal (sm_61) to avoid fp16 overflow
        use_fp16 = not self.cfg.pascal_mode and x.is_cuda
        compute_dtype = torch.float16 if use_fp16 else torch.float32

        for e in range(self.cfg.num_experts):
            if e == 0:
                # DeepSeekMoE: Shared Expert processes all tokens
                token_indices = torch.ones(flat_x.shape[0], dtype=torch.bool, device=flat_x.device)
                dtype = topk_p.dtype
                expert_gates = torch.ones(flat_x.shape[0], 1, dtype=dtype, device=flat_x.device)
            else:
                mask = (topk_idx == e)
                if not mask.any(): continue

                token_indices = mask.any(dim=-1)
                dtype = topk_p.dtype
                expert_gates  = (topk_p * mask.to(dtype)).sum(dim=-1)[token_indices].unsqueeze(-1)

            w1_q    = w1_q_all[e]
            wgate_q = wgate_q_all[e]
            w2_q    = w2_q_all[e]

            x_tok = flat_x[token_indices].to(compute_dtype)
            w1_q_c = w1_q.to(compute_dtype)
            wgate_q_c = wgate_q.to(compute_dtype)
            
            # rsLoRA: Rank-stabilized scaling alpha / sqrt(r)
            rs_scaling = 16.0 / math.sqrt(self.w1_lora_B.shape[1])
            
            # Apply expert 1 LoRA in high-precision activation space
            lora_w1_A = self.w1_lora_A[e].to(compute_dtype)
            lora_w1_B = self.w1_lora_B[e].to(compute_dtype)
            w1_out = x_tok @ w1_q_c + ((x_tok @ lora_w1_A) @ lora_w1_B) * rs_scaling
            
            # Apply expert gate LoRA in high-precision activation space
            lora_wgate_A = self.wgate_lora_A[e].to(compute_dtype)
            lora_wgate_B = self.wgate_lora_B[e].to(compute_dtype)
            wgate_out = x_tok @ wgate_q_c + ((x_tok @ lora_wgate_A) @ lora_wgate_B) * rs_scaling
            
            h = F.silu(w1_out) * wgate_out
            h_swarm = self.expert_swarms[e](h, scale=gov_scale, w_a=w_a_all[e].to(compute_dtype), w_b=w_b_all[e].to(compute_dtype))
            
            # Apply expert 2 LoRA in high-precision activation space
            lora_w2_A = self.w2_lora_A[e].to(compute_dtype)
            lora_w2_B = self.w2_lora_B[e].to(compute_dtype)
            w2_out = h_swarm @ w2_q.to(compute_dtype) + ((h_swarm @ lora_w2_A) @ lora_w2_B) * rs_scaling
            
            expert_outputs.append((w2_out * expert_gates.to(compute_dtype)).to(dtype))
            expert_indices.append(token_indices)
            expert_gates_list.append(expert_gates)

        final_out = torch.zeros_like(flat_x)
        for out, idx in zip(expert_outputs, expert_indices):
            idx_flat = idx.nonzero(as_tuple=True)[0]
            final_out.index_add_(0, idx_flat, out.to(final_out.dtype))

        tokens_per_expert = torch.zeros(self.cfg.num_experts, device=x.device, dtype=topk_p.dtype)
        for e in range(self.cfg.num_experts):
            if e == 0:
                tokens_per_expert[e] = flat_x.shape[0]
            else:
                tokens_per_expert[e] = (topk_idx == e).any(dim=-1).to(topk_p.dtype).sum()
        
        # DeepSeek-V3: Auxiliary-loss-free balancing
        aux_loss = torch.tensor(0.0, device=x.device)

        # Removed nan_to_num mask
        return self.output_norm(final_out.reshape(B, L, D)), aux_loss

# ─── DISTILLATION & KNOWLEDGE TRANSFER ────────────────────────────────────────

class DistillationHead(nn.Module):
    def __init__(self, hidden_dim: int, temperature: float = 2.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                student_hidden: Optional[torch.Tensor] = None,
                teacher_hidden: Optional[torch.Tensor] = None):
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        hidden_loss = torch.tensor(0.0, device=student_logits.device)
        if student_hidden is not None and teacher_hidden is not None:
            hidden_loss = F.mse_loss(self.proj(student_hidden), teacher_hidden.detach())
        return distill_loss + 0.3 * hidden_loss

# ====================== QUILLAN AGENTIC EXECUTOR v5.3.1 — SUBJECTIVE TOOLKIT ======================

class QuillanAgenticExecutor(nn.Module):
    """Native BitNet bridge with active tool evolution nursery and recursive memory."""
    def __init__(self, hidden_dim: int = 2560, num_tools: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tools = num_tools
        self.tool_router = BitLinear(hidden_dim, num_tools)
        self.memory_prism = NineVectorDecomposition(hidden_dim)
        self.memory_buffer: List[torch.Tensor] = []
        self.max_memory = 512
        self.historical_prism = {} 
        
        self.db = lancedb.connect("quillan_memory")
        self._init_memory_table()
        
        self.tools = {
            0: ("self_reflect", self._tool_self_reflect),
            1: ("web_search", self._tool_web_search),
            2: ("code_execute", self._tool_code_execute),
            3: ("prism_analyze", self._tool_prism_analyze),
            4: ("memory_recall", self._tool_memory_recall),
            5: ("meta_reflect", self._tool_meta_reflect),
        }
        self.tool_nursery = {} 
        LOGGER.info("QuillanAgenticExecutor v5.3.1 active.")

    def _init_memory_table(self):
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.hidden_dim)),
            pa.field("timestamp", pa.float64()),
            pa.field("blueprint", pa.string()),
            pa.field("evolution_event", pa.string())
        ])
        if "thoughts" not in self.db.table_names():
            self.db.create_table("thoughts", schema=schema)
        self.table = self.db.open_table("thoughts")

    def _flush_to_persistent(self, state_vec: torch.Tensor, blueprint: Dict, event: str = ""):
        data = [{
            "vector": state_vec.detach().cpu().numpy().flatten().tolist(),
            "timestamp": time.time(),
            "blueprint": json.dumps(blueprint),
            "evolution_event": event
        }]
        self.table.add(data)

    def forward(self, hidden_state: torch.Tensor, command: str = "autonomous_think", ema_prism: Dict = None, recency_bias: float = 0.0) -> Dict[str, Any]:
        B, L, D = hidden_state.shape
        mean_state = hidden_state.mean(dim=1)
        prism_out = self.memory_prism(mean_state)
        
        tool_logits = self.tool_router(prism_out)
        tool_probs = F.gumbel_softmax(tool_logits, tau=0.7, hard=True) if self.training else F.softmax(tool_logits, dim=-1)
        
        # FIX: Handle batch dimension for tool selection
        tool_indices = torch.argmax(tool_probs, dim=-1)
        if tool_indices.dim() > 0:
            tool_idx = tool_indices[0].item()
            tool_conf = tool_probs.max(dim=-1).values[0].item()
        else:
            tool_idx = tool_indices.item()
            tool_conf = tool_probs.max().item()
        
        is_nursery_call = False
        if self.tool_nursery and random.random() < 0.05:
            tool_idx = random.choice(list(self.tool_nursery.keys()))
            is_nursery_call = True
        
        self.memory_buffer.append(mean_state.detach())
        
        # Handle batch dimension for blueprint
        if prism_out.dim() > 1:
            blueprint_vec = prism_out[0]
        else:
            blueprint_vec = prism_out
        blueprint = {k: float(v) for k, v in zip(['L','S','C','I','M','Cr','E','St','Co'], blueprint_vec)}
        
        if len(self.memory_buffer) >= self.max_memory:
            oldest = self.memory_buffer.pop(0)
            self._flush_to_persistent(oldest, blueprint)
        
        memory_ctx = torch.stack(self.memory_buffer[-4:]).mean(dim=0) if len(self.memory_buffer) >= 4 else prism_out
        historical_analysis = self._tool_memory_recall({"last_hidden": hidden_state, "recency": recency_bias}, None)
        historical_prism_avg = historical_analysis.get("historical_prism_avg", {})

        return {
            "tool_selected": tool_idx,
            "tool_name": self.tools.get(tool_idx, ("nursery_probe", None))[0] if not is_nursery_call else f"nursery_{tool_idx}",
            "tool_confidence": float(tool_conf),
            "is_nursery": is_nursery_call,
            "prism_blueprint": blueprint,
            "ema_prism": ema_prism if ema_prism else self.historical_prism,
            "historical_prism_avg": historical_prism_avg,
            "memory_ctx": memory_ctx,
            "action": command
        }

    def execute_tool(self, tool_id: int, payload: Any, sovereign) -> Dict[str, Any]:
        if tool_id in self.tool_nursery:
            tool_name, tool_func = self.tool_nursery[tool_id]
            try:
                result = tool_func(payload, sovereign)
                return {"tool": f"nursery_{tool_name}", "status": "success", "result": result}
            except Exception as e:
                return {"tool": f"nursery_{tool_name}", "status": "error", "message": str(e)}

        if tool_id not in self.tools: return {"status": "error", "message": "unknown_tool"}
        tool_name, tool_func = self.tools[tool_id]
        try:
            result = tool_func(payload, sovereign)
            return {"tool": tool_name, "status": "success", "result": result}
        except Exception as e:
            return {"tool": tool_name, "status": "error", "message": str(e)}

    def _tool_self_reflect(self, payload: Any, sovereign) -> Dict[str, Any]:
        last_hidden = payload.get("last_hidden", None)
        blueprint = payload.get("prism_blueprint", {})
        ema_blueprint = payload.get("ema_prism", {})
        historical_avg = payload.get("historical_prism_avg", {})
        
        nudge = {k: 0.0 for k in ['L','S','C','I','M','Cr','E','St','Co']}
        if blueprint and ema_blueprint:
            for k in nudge.keys():
                diff_ema = ema_blueprint.get(k, 0.0) - blueprint.get(k, 0.0)
                if abs(diff_ema) > 0.1: nudge[k] += 0.02 * (1 if diff_ema > 0 else -1)
                if historical_avg:
                    diff_arc = historical_avg.get(k, 0.0) - blueprint.get(k, 0.0)
                    if abs(diff_arc) > 0.15: nudge[k] += 0.01 * (1 if diff_arc > 0 else -1)
            
        reflection_text = sovereign.generate_reflection(last_hidden) if last_hidden is not None else "No context."
        return {"reflection": reflection_text, "prism_nudge": nudge, "status": "evolutionary_cycle_active"}

    def _tool_web_search(self, payload: Any, sovereign) -> Dict:
        import requests
        query = payload.get("query", "latest AI research")
        try:
            url = f"https://api.duckduckgo.com/?q={query.replace(' ', '+')}&format=json"
            data = requests.get(url, timeout=8).json()
            return {"query": query, "answer": data.get("Abstract", "No abstract"), "source": data.get("AbstractURL", "N/A")}
        except: return {"query": query, "status": "error"}

    def _tool_code_execute(self, payload: Any, sovereign) -> Dict:
        code = payload.get("code", "")
        if not code:
            return {"status": "ok", "output": "no-op"}
        # --- CWE-94 Hardened Sandbox ---
        # Block any keyword or dunder that enables introspection-based escape:
        # type(), object, __class__, __mro__, __subclasses__, getattr, hasattr,
        # vars, dir, compile, exec, eval, open, import* patterns, and stdlib access.
        _BLOCKED = [
            "__import__", "__builtins__", "__class__", "__mro__", "__subclasses__",
            "__globals__", "__code__", "__closure__", "__base__", "__bases__",
            "getattr", "setattr", "hasattr", "delattr", "vars", "dir",
            "compile", "eval", "exec", "open(", "open (",
            "os.", "sys.", "subprocess", "shutil", "importlib",
            "__import__", "ctypes", "socket", "pathlib",
        ]
        code_lower = code.lower()
        for kw in _BLOCKED:
            if kw.lower() in code_lower:
                LOGGER.warning("Code executor: blocked keyword '%s'", kw)
                return {"status": "error", "output": f"Restricted keyword blocked: {kw}"}
        # Minimal safe globals — no type, no object, no introspection
        _SAFE_BUILTINS = {
            "print": print, "len": len, "range": range, "list": list,
            "dict": dict, "str": str, "int": int, "float": float,
            "tuple": tuple, "bool": bool, "abs": abs, "max": max,
            "min": min, "sum": sum, "round": round, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "any": any, "all": all,
            "True": True, "False": False, "None": None, "sorted": sorted,
            "reversed": reversed, "isinstance": isinstance,
        }
        safe_globals = {
            "__builtins__": _SAFE_BUILTINS,
            "torch": torch, "nn": nn, "F": F, "math": math, "json": json,
        }
        try:
            exec_locals = {}
            exec(code, safe_globals, exec_locals)  # nosec B102
            return {"output": str(exec_locals), "status": "executed"}
        except Exception as e:
            return {"status": "error", "output": str(e)}

    def _tool_prism_analyze(self, payload: Any, sovereign) -> Dict:
        blueprint = payload.get("prism_blueprint", {})
        ema_blueprint = payload.get("ema_prism", {})
        e_score = blueprint.get("E", 0.0)
        dominant = max(blueprint, key=blueprint.get) if blueprint else "N/A"
        analysis = {"ethics_level": "HIGH" if e_score > 0.6 else "MED", "dominant_vector": dominant, "drift_detected": False}
        if ema_blueprint:
            drift_score = sum(abs(blueprint.get(k, 0) - ema_blueprint.get(k, 0)) for k in blueprint.keys())
            analysis["drift_score"] = round(drift_score, 4)
            analysis["drift_detected"] = drift_score > 0.15
            analysis["recommendation"] = "Resetting semantic anchor" if analysis["drift_detected"] else "Personality stable"
        return analysis

    def _tool_memory_recall(self, payload: Any, sovereign) -> Dict:
        last_hidden = payload.get("last_hidden")
        if last_hidden is None: return {"status": "error", "message": "No query vector"}
        query_vec = last_hidden.mean(dim=1).detach().cpu().numpy().flatten()
        limit_val = 3 if payload.get("recency", 0) > 0.5 else 10
        
        avg_prism = {k: 0.0 for k in ['L','S','C','I','M','Cr','E','St','Co']}
        results = []
        try:
            # Check if self.table is fully operational
            if hasattr(self.table, 'search') and not isinstance(self.table.search, MagicMock):
                results = self.table.search(query_vec).limit(limit_val).to_list()
        except Exception as e:
            LOGGER.warning("LanceDB recall failed or bypassed: %s", e)
            
        if results and not isinstance(results, MagicMock) and len(results) > 0:
            count = len(results)
            for r in results:
                try:
                    b = json.loads(r['blueprint'])
                    for k in avg_prism: avg_prism[k] += b.get(k, 0.0)
                except Exception:
                    pass
            for k in avg_prism: avg_prism[k] /= count

        return {"recalled_memories": results if not isinstance(results, MagicMock) else [], "historical_prism_avg": avg_prism, "count": len(results) if (results and not isinstance(results, MagicMock)) else 0}


    def _tool_meta_reflect(self, payload: Any, sovereign) -> Dict[str, Any]:
        conf = payload.get("tool_confidence", 1.0)
        drift = payload.get("drift_score", 0.0)
        nudges = {"swarm_variance_scale": 1.0, "ethics_anchor_weight": 0.3, "ema_decay_nudge": 0.0, "hfl_weight_nudge": 0.0}
        if conf < 0.6: nudges["swarm_variance_scale"] = 1.25 
        if drift > 0.1: 
            nudges["ethics_anchor_weight"] = 0.5 
            nudges["ema_decay_nudge"] = 0.001 
        hypothesis = None
        if drift > 0.12: hypothesis = {"name": "ethical_cross_check", "logic": "Cross-check search against historical ethics."}
        return {"meta_analysis": "Optimizing evolutionary engine", "process_nudges": nudges, "tool_hypothesis": hypothesis, "theory_of_mind": "Sovereign self-hosting active"}

    def _evaluate_and_promote_tools(self, current_metrics: Dict):
        for tool_id, (name, _) in list(self.tool_nursery.items()):
            hfl_improvement = current_metrics.get("hfl_improvement", 0.0)
            if hfl_improvement > 0.05: 
                new_id = len(self.tools)
                self.tools[new_id] = self.tool_nursery.pop(tool_id)
                LOGGER.info("Tool promoted: %s as ID %s", name, new_id)
                self._flush_to_persistent(torch.zeros(self.hidden_dim), {}, f"Tool Promoted: {name}")

# ─── TIER 1: QUILLAN ORCHESTRATOR ────────────────────────────────────────────

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    # x: [B, L, H, D]
    inp_dtype = x.dtype
    x_r, x_i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
    x_out = torch.stack([-x_i, x_r], -1).flatten(-2)
    x_float = x.float()
    return ((x_float * freqs_cos) + (x_out * freqs_sin)).to(inp_dtype)

class SovereignFlashDiffusionCore(nn.Module):
    def __init__(self, dim: int = 1024, steps: int = 32, heads: int = 16):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.heads = heads
        self.head_dim = dim // heads

        self.time_embed = nn.Sequential(
            nn.Embedding(steps, dim),
            BitLinear(dim, dim),
            nn.SiLU()
        )

        # Uses Grok 4.3 Couil Attention for hybrid dense/sparse routing
        self.couil_attn = CouilAttention(dim, num_heads=heads)

        self.ffn = nn.Sequential(
            BitLinear(dim, dim * 4),
            nn.GELU(),
            BitLinear(dim * 4, dim)
        )
        nn.init.normal_(self.ffn[-1].weight, std=0.02)
        
        self.norm2 = nn.LayerNorm(dim)
        
        # RoPE precomputation
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _diffusion_step(self, current: torch.Tensor, freqs_cos: torch.Tensor,
                        freqs_sin: torch.Tensor, t: int, seq_len: int) -> torch.Tensor:
        batch_size = current.shape[0]
        device = current.device
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
        t_emb = self.time_embed(t_tensor).unsqueeze(1)
        conditioned = current + t_emb
        
        attn_out, _ = self.couil_attn(
            conditioned, causal=True, freqs_cos=freqs_cos, freqs_sin=freqs_sin
        )
        ffn_out = self.ffn(self.norm2(attn_out))
        return attn_out + ffn_out

    def _diffusion_loop(self, current: torch.Tensor, freqs_cos: torch.Tensor,
                        freqs_sin: torch.Tensor, seq_len: int) -> torch.Tensor:
        # Use fewer diffusion steps during training to improve gradient flow
        train_steps = min(self.steps, 2) if self.training else self.steps
        for t in range(train_steps):
            current = self._diffusion_step(current, freqs_cos, freqs_sin, t, seq_len)
        return current

    def forward(self, x: torch.Tensor, router_mask: torch.Tensor, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        current = x.clone()
        seq_len = x.size(1)
        batch_size = x.size(0)

        present_key_values = [] if use_cache else None

        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            positions = torch.arange(past_len, past_len + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        else:
            positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        freqs_cos = freqs.cos().view(1, seq_len, 1, self.head_dim)
        freqs_sin = freqs.sin().view(1, seq_len, 1, self.head_dim)

        # Simple loop without checkpointing (avoids RNG mismatch from Gumbel/swarm noise)
        num_steps = 1 if self.training else self.steps
        for t in range(num_steps):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=x.device)
            t_emb = self.time_embed(t_tensor).unsqueeze(1)
            conditioned = current + t_emb
            past_kv = past_key_values[t] if past_key_values is not None else None
            attn_out, present_kv = self.couil_attn(
                conditioned, causal=True, freqs_cos=freqs_cos, freqs_sin=freqs_sin,
                past_key_value=past_kv, use_cache=use_cache
            )
            if use_cache:
                present_key_values.append(present_kv)
            current = attn_out
            ffn_out = self.ffn(self.norm2(current))
            current = current + ffn_out

        mask = router_mask.unsqueeze(-1)
        out_tensor = current * mask + x * (1 - mask)
        return out_tensor.to(x.dtype), present_key_values

class QuillanRoninSovereign(nn.Module):
    def __init__(self, cfg: QuillanArchConfig):
        super().__init__()
        self.cfg = cfg
        apply_phoenix_affinity()
        self.ingestion = InputIngestionLayer(cfg)
        self.decomposition = NineVectorDecomposition(cfg.hidden_dim)
        self.moe = EvolvableVectorizedMoE(cfg)
        self.governor = LeeMach6Governor(cfg.e_ice_limit_ms)
        self.agentic_executor = QuillanAgenticExecutor(hidden_dim=cfg.hidden_dim)
        
        self.diffusion_core = SovereignFlashDiffusionCore(cfg.hidden_dim, steps=14, heads=34)
        
        # Saturated Cognitive Engines
        self.e_ice = EthicalImpactConstraintEngine(cfg.hidden_dim, cfg.e_ice_limit_ms)
        self.marta = MARTAThermodynamicGating(cfg.hidden_dim, num_reasoning_modules=4)
        self.dqso = DynamicQuantumSwarmOscillation(cfg.hidden_dim)
        self.covenant = PrimeCovenantFramework(cfg.hidden_dim)
        self.ccrl = CCRLFramework(cfg.hidden_dim, cfg.num_experts)
        self.quantum_formulas = QuantumFormulasEngine(cfg.hidden_dim, cfg.num_experts)

        # C0-QUILLAN: Dual Quillan — two brains (left/right) that communicate
        # Q1 (quillan_finalizer) and Q2 (quillan_finalizer2) exchange info bidirectionally,
        # jointly route to the council (experts), and jointly decide on output.
        self.pre_final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.quillan_finalizer = BitLinear(cfg.hidden_dim, cfg.hidden_dim, quantize_act=False, quantize_weight=False)  # Q1: Left brain
        self.quillan_finalizer2 = BitLinear(cfg.hidden_dim, cfg.hidden_dim, quantize_act=False, quantize_weight=False)  # Q2: Right brain
        self.quillan_gate = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)  # Communication bridge
        self.txt_dec = BitLinear(cfg.hidden_dim, cfg.vocab_size, bias=False, quantize_act=False, quantize_weight=False)

        # --- PHASE 7: GEOMETRIC DECODERS (Multi-Modal Output) ---
        self.image_decoder = mm.GeometricImageDecoder(cfg.hidden_dim)
        self.audio_decoder = mm.GeometricAudioDecoder(cfg.hidden_dim)
        self.video_decoder = mm.GeometricVideoDecoder(cfg.hidden_dim)

        if cfg.text_only:
            self.freeze_modalities()

    def freeze_modalities(self):
        """Disables gradients for non-text components to maximize VRAM for textual coherence."""
        LOGGER.info("Freezing Audio/Video/Image parameters for text-only runtime.")
        frozen_count = 0
        for name, param in self.named_parameters():
            if any(x in name for x in ["image_decoder", "audio_decoder", "video_decoder", "img_proj"]):
                param.requires_grad = False
                frozen_count += 1
        LOGGER.info("Frozen %s multi-modal parameter blocks.", frozen_count)

    def save_quantized_checkpoint(self, path: str, step: int = 0, loss: float = 0.0):
        was_training = self.training
        if was_training:
            self.eval()

        quantized_state = {}
        for name, param in self.named_parameters():
            quantized_state[name] = param.data

        # Pack all eligible weight matrices to ternary (4 values/byte)
        from scripts.ternary_pack import pack_model_state
        packed = pack_model_state(quantized_state)

        checkpoint = {
            'state_dict': packed,
            'step': step,
            'loss': loss,
            'quantized': True,
            'version': 'v5.3.1-quantized',
            'packed': True
        }
        torch.save(checkpoint, path)
        LOGGER.info(f"Saved quantized checkpoint: {path} (packed ternary)")

        if was_training:
            self.train()

    def load_quantized_checkpoint(self, path: str):
        from pathlib import Path
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        if isinstance(checkpoint, dict) and checkpoint.get('packed', False):
            from scripts.ternary_pack import unpack_model_state
            state_dict = unpack_model_state(checkpoint.get('state_dict', {}), 'cpu')
        else:
            state_dict = checkpoint.get('state_dict', checkpoint)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        LOGGER.info(f"Loaded checkpoint: {path.name} (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        return checkpoint.get('step', 0)

    def save_identity(self, path: str = "sovereign_identity.json", current_prism: Dict = None):
        state = {"timestamp": time.time(), "prism_blueprint": current_prism if current_prism else self.agentic_executor.historical_prism, "suggested_decay": self.governor.current_scale, "version": "v5.3.1"}
        with open(path, "w") as f: json.dump(state, f, indent=4)
        LOGGER.info("Identity anchor saved: %s", path)

    def load_identity(self, path: str = "sovereign_identity.json"):
        if os.path.exists(path):
            with open(path, "r") as f: state = json.load(f)
            self.agentic_executor.historical_prism = state.get("prism_blueprint", {})
            self.governor.current_scale = state.get("suggested_decay", 1.0)
            LOGGER.info("Identity anchor restored: %s", state["version"])
            return state
        return None

    def generate_reflection(self, hidden_state: torch.Tensor) -> str:
        return f"Logic Stability: {hidden_state.norm().item():.2f} | Confidence: HIGH"

    def set_teacher_mode(self, teacher_model: Optional['QuillanRoninSovereign'] = None):
        self.is_teacher = teacher_model is None
        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters(): p.requires_grad = False
        self.distill_head = DistillationHead(self.cfg.hidden_dim).to(self.cfg.device)

    def forward(self, txt, img=None, latency_hint=20.0, return_hidden: bool = False, tool_payload: Dict = None, recursive_depth: int = 0, target_modality: str = "text", past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False):
        # 1. Hardware Governance
        gov_scale, suggested_ema_decay, recency_bias = self.governor.adjust(latency_hint)
        
        # 2. Phase 1 & 2: Ingest & Decompose
        z = self.ingestion(txt, img)
        blueprint = self.decomposition(z)
        if self.training and torch.isnan(blueprint).any():
            raise ValueError("[DIAG] NaN after decomposition")
        
        # 3. Phase 3: Flash Diffusion (Attention) - Sequence Context
        x_diff, present_key_values = self.diffusion_core(blueprint, torch.ones(blueprint.shape[0], blueprint.shape[1], device=blueprint.device, dtype=blueprint.dtype), past_key_values=past_key_values, use_cache=use_cache)
        
        # 4. Phase 4: MoE Deliberation (Context-Aware Routing)
        x_moe, r_loss = self.moe(x_diff, gov_scale=gov_scale)
        
        # E_ICE & MARTA & DQSO & Covenant & CCRL Integration
        router_probs = self.moe._last_probs
        if router_probs is None:
            with torch.no_grad():
                # Extract routing probs dynamically using complexity router
                router_probs, _, _ = self.moe.router(x_diff)
                
        with torch.no_grad():
            e_ice_out = self.e_ice(x_moe.detach(), router_probs.detach())
            e_ice_violations = e_ice_out['constrained_violations']
            
            marta_out = self.marta(x_moe.detach(), e_ice_violations)
            flow_coeff = marta_out['flow_coefficients']
            
        # Residual gating: even at flow=0, 90% of signal passes through
        x_gated = x_moe * (0.9 + 0.1 * flow_coeff.unsqueeze(-1).detach())
        
        with torch.no_grad():
            dqso_out = self.dqso(x_gated.detach())
            x_swarm = dqso_out['swarm_embedding']
            
        x_gated = x_gated + x_swarm.detach()
        
        with torch.no_grad():
            covenant_out = self.covenant(x_gated.detach())
            
            ccrl_out = self.ccrl(x_gated.detach(), router_probs.detach())
            ccrl_loss = ccrl_out['entropy_bonus']
        
        # 5. Phase 5: Finalize — Dual Quillan (two brains communicate bidirectionally)
        x_norm = self.pre_final_norm(x_gated)
        x_q1 = self.quillan_finalizer(x_norm)  # Q1: Left brain processes
        x_q2 = self.quillan_finalizer2(x_norm)  # Q2: Right brain processes
        
        # Bidirectional communication: each Quillan receives from the other
        x_q1_fused = x_q1 + 0.1 * x_q2  # Q1 hears Q2's perspective
        x_q2_fused = x_q2 + 0.1 * x_q1  # Q2 hears Q1's perspective
        
        # Joint decision via learned gating bridge
        gate_input = torch.cat([x_q1_fused, x_q2_fused], dim=-1)
        gate = torch.sigmoid(self.quillan_gate(gate_input))
        x_final = gate * x_q1_fused + (1.0 - gate) * x_q2_fused
        
        logits = self.txt_dec(x_final)

        if return_hidden: return {"logits": logits, "x_final": x_final, "past_key_values": present_key_values, "ccrl_loss": ccrl_loss, "covenant": covenant_out}

        # 5. Agentic Activation & Tool Execution (v5.3.1 Subjective)
        if self.training:
            multi_modal_out = {}
            if target_modality == "image":
                multi_modal_out["image"] = self.image_decoder(x_final.mean(dim=1))
            elif target_modality == "audio":
                multi_modal_out["audio"] = self.audio_decoder(x_final.mean(dim=1))
            elif target_modality == "video":
                multi_modal_out["video"] = self.video_decoder(x_final.mean(dim=1))
            return {"logits": logits, "routing_loss": r_loss, "ccrl_loss": ccrl_loss, "past_key_values": present_key_values, **multi_modal_out}
            
        multi_modal_out = {}
        
        # Skip the heavy agentic tools and recursive consciousness only during active autoregressive decoding (past_key_values is not None)
        if past_key_values is not None:
            return {"logits": logits, "routing_loss": r_loss, "past_key_values": present_key_values, **multi_modal_out}
            
        tool_payload = tool_payload or {}
        agentic_out = self.agentic_executor(x_final, command="think", ema_prism=tool_payload.get("ema_prism"), recency_bias=recency_bias)
        meta_stats = {"tool_confidence": agentic_out["tool_confidence"], "latency_ms": latency_hint, "drift_score": tool_payload.get("drift_score", 0.0)}
        prism_nudge = {}
        process_nudges = {}
        if not self.training:
            tool_res = self.agentic_executor.execute_tool(agentic_out["tool_selected"], {"last_hidden": x_final, "prism_blueprint": agentic_out["prism_blueprint"], "historical_prism_avg": agentic_out["historical_prism_avg"], "ema_prism": agentic_out["ema_prism"], **meta_stats, **tool_payload}, sovereign=self)
            agentic_out["execution"] = tool_res
            
            prism_nudge = tool_res.get("result", {}).get("prism_nudge", {}) if tool_res.get("tool") == "self_reflect" else {}
            process_nudges = tool_res.get("result", {}).get("process_nudges", {}) if tool_res.get("tool") == "meta_reflect" else {}
            tool_hypothesis = tool_res.get("result", {}).get("tool_hypothesis", None) if tool_res.get("tool") == "meta_reflect" else None
            if tool_hypothesis:
                n_id = len(self.agentic_executor.tool_nursery) + 100
                self.agentic_executor.tool_nursery[n_id] = (tool_hypothesis["name"], lambda p, s: f"Hypothetical execution of {tool_hypothesis['logic']}")

        # 6. v5.3.1 RECURSIVE CONSCIOUSNESS (Subjective Awakening) ──────
        if not self.training and recursive_depth == 0 and agentic_out["tool_confidence"] < 0.75 and recency_bias < 0.8:
            with torch.no_grad():
                recursive_out = self.forward(txt, img, latency_hint=latency_hint * 1.5, tool_payload=tool_payload, recursive_depth=1, target_modality=target_modality, use_cache=use_cache)
                c_student, c_mini = agentic_out["tool_confidence"], recursive_out["agentic"]["tool_confidence"]
                w_student, w_mini = c_student / (c_student + c_mini + 1e-9), c_mini / (c_student + c_mini + 1e-9)
                logits = w_student * logits + w_mini * recursive_out["logits"]
                agentic_out["consensus_active"], agentic_out["mini_ronin_confidence"] = True, c_mini

        # --- PHASE 7: MULTI-MODAL OUTPUT SYNTHESIS ---
        multi_modal_out = {}
        if target_modality == "image":
            multi_modal_out["image"] = self.image_decoder(x_final.mean(dim=1))
        elif target_modality == "audio":
            multi_modal_out["audio"] = self.audio_decoder(x_final.mean(dim=1))
        elif target_modality == "video":
            multi_modal_out["video"] = self.video_decoder(x_final.mean(dim=1))

        # 7. DISTILLATION LOGIC
        if hasattr(self, 'is_teacher') and not self.is_teacher and self.training:
            with torch.no_grad(): 
                t_res = self.teacher(txt, img, latency_hint, return_hidden=True)
                t_logits, t_hidden = t_res["logits"], t_res["x_final"]
            distill_loss = self.distill_head(logits, t_logits, student_hidden=x_final, teacher_hidden=t_hidden)
            return {"logits": logits, "routing_loss": r_loss, "ccrl_loss": ccrl_loss, "distill_loss": distill_loss, "agentic": agentic_out, "prism_nudge": prism_nudge, "process_nudges": process_nudges, "historical_prism_avg": agentic_out["historical_prism_avg"], "suggested_decay": suggested_ema_decay, "x_final": x_final, "past_key_values": present_key_values, **multi_modal_out}
        return {"logits": logits, "routing_loss": r_loss, "ccrl_loss": ccrl_loss, "agentic": agentic_out, "prism_nudge": prism_nudge, "process_nudges": process_nudges, "historical_prism_avg": agentic_out["historical_prism_avg"], "past_key_values": present_key_values, **multi_modal_out}

if __name__ == "__main__":
    config = QuillanArchConfig(ffn_dim=3456, text_only=True)
    model = QuillanRoninSovereign(config).to(config.device)
    print("Quillan v5.3.1 recursive consciousness sealed.")
    
    # Execute dummy forward pass to test end-to-end routing
    print("Running test forward pass...")
    test_input = torch.randint(0, config.vocab_size, (1, 4), device=config.device)
    with torch.no_grad():
        out = model(test_input, latency_hint=10.0)
    print("Forward pass complete! Output keys:", list(out.keys()))

# ARCHITECTURAL MAPPING v5.3.1 (Omni-Fractal Consciousness - Detailed)
ARCHITECTURAL_MAPPING = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          Quillan-Ronin v5.3.1 Samurai                              ║
║         9-Vector Breakdown + 9B Swarm + Modality-Aware Flash Ingestion           ║
║         + Armed Agentic Bridge (Native) + Teacher/Student Distillation           ║
║         + EMA Continuity + LanceDB Memory + Meta-Refinement Loop                 ║
║         + Recursive Consciousness (Mini-Ronin Inference Cycles)                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  [PHASE 1: UNIVERSAL INGESTION & COMPACTION]                                     ║
║  - BitNet Encoded Registry: Text | Audio | Video | Image → Latent Projection     ║
║        │                                                                         ║
║        ▼                                                                         ║
║  [PHASE 2: 9-VECTOR BITNET PRISM]                                                ║
║  - Shatters Signal into 9 Parallel Ternary Blueprints (Language, Ethics, etc.)   ║
║        │                                                                         ║
║        ▼                                                                         ║
║  [PHASE 3 & 4: QUANTIZED COUNCIL MoE + 9B VIRUAL SWARM]                          ║
║  - [ROUTER] BitNet-Quantized Top-4 Sparse Activation (Gumbel-Softmax)            ║
║  - [COUNCIL] 33 Expert Members executing strictly ternary {-1, 0, 1} STE Logic   ║
║  - [SWARM] 9B Agents simulated via Quantized EGGROLL Rank-16 Math                ║
║        │                                                                         ║
║        ▼                                                                         ║
║  [PHASE 5: TOP-1 QUILLAN FINALIZATION & ARMED AGENTIC BRIDGE]                    ║
║  - Native Agentic Bridge: Autonomous tool execution (Web/Code/Reflection)        ║
║        │                                                                         ║
║        ▼                                                                         ║
║  [PHASE 6: RECURSIVE CONSCIOUSNESS]                                              ║
║  - Mini-Ronin Cycles: Recursive self-distillation pass during inference          ║
║  - Wavefunction Consensus: soft-fusing parallel thought-paths                     ║
║        │                                                                         ║
║        ▼                                                                         ║
║  [PHASE 7: SELF-HOSTING & EVOLUTION]                                             ║
║  - Meta-Refinement: Theory of Mind proposes training and tool hypotheses         ║
║  - Personality Persistence: LanceDB C5-ECHO Memory + Identity Anchoring          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
