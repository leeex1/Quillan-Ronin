#!/usr/bin/env python3
"""
Phase 2+3: Projection Matrices + Weight Transplant for Quillan-Ronin
Loads source models, applies projections, maps to Quillan architecture.
"""
import os
import sys
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("quillan_build.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Transplant")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE = r"C:\Users\Admin\Quillan-Ronin\Quillan-v4.2-model"

@dataclass
class QuillanConfig:
    hidden_dim: int = 2048
    ffn_dim: int = 4096
    vocab_size: int = 50257
    num_experts: int = 34
    top_k: int = 4
    swarm_rank: int = 8
    swarm_instances: int = 8
    diffusion_layers: int = 6
    diffusion_heads: int = 8
    num_vectors: int = 9

# ═══════════════════════════════════════════════════════════════
# QUILLAN MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════

class BitLinear(nn.Linear):
    """BitNet 1.58b ternary projection with STE."""
    def _quantize(self, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scale = w.abs().mean(dim=[-2, -1], keepdim=True).clamp(min=1e-5)
            w_q = torch.round(torch.clamp(w / scale, -1.0, 1.0))
        return w + (w_q * scale - w).detach()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._quantize(self.weight), self.bias)


class CouncilExpertSwarm(nn.Module):
    """EGGROLL Rank-8 micro-agent swarm per expert."""
    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.01)
    
    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        w_a = self._quantize(self.A)
        w_b = self._quantize(self.B)
        return x + (x @ w_a @ w_b) * scale * 0.25
    
    def _quantize(self, w: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scale = w.abs().mean(dim=[-2, -1], keepdim=True).clamp(min=1e-5)
            w_q = torch.round(torch.clamp(w / scale, -1.0, 1.0))
        return w + (w_q * scale - w).detach()


class NineVectorDecomposition(nn.Module):
    """9-Vector BitNet Prism for cognitive decomposition."""
    def __init__(self, dim: int):
        super().__init__()
        self.vectors = nn.ModuleDict({
            k: BitLinear(dim, dim, bias=False) for k in
            ['Lang', 'Senti', 'Ctx', 'Intent', 'Meta', 'Crea', 'Ethic', 'Strat', 'Const']
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(v(x) for v in self.vectors.values()) / 9.0


class ExpertFFN(nn.Module):
    """Single expert FFN with optional swarm."""
    def __init__(self, hidden_dim: int, ffn_dim: int, use_swarm: bool = True, rank: int = 8):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.swarm = CouncilExpertSwarm(ffn_dim, rank) if use_swarm else None
    
    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        h = F.relu(self.w1(x))
        if self.swarm is not None:
            h = self.swarm(h, scale)
        return self.w2(h)


class DiffusionLayer(nn.Module):
    """Single diffusion layer with attention."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        x = residual + self.o_proj(attn)
        
        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))
        
        return x


class QuillanRoninSovereign(nn.Module):
    """Complete Quillan-Ronin v5.3.1 Architecture."""
    
    def __init__(self, cfg: QuillanConfig):
        super().__init__()
        self.cfg = cfg
        
        # Phase 1: Ingestion
        self.txt_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.mod_emb = nn.Embedding(4, cfg.hidden_dim)
        
        # Phase 2: 9-Vector Decomposition
        self.decomposition = NineVectorDecomposition(cfg.hidden_dim)
        
        # Phase 3: Dual Router
        self.router = BitLinear(cfg.hidden_dim, cfg.num_experts)
        self.quillan_finalizer = BitLinear(cfg.hidden_dim, cfg.hidden_dim)
        
        # Phase 3: Expert FFNs (34 experts)
        self.experts = nn.ModuleList([
            ExpertFFN(cfg.hidden_dim, cfg.ffn_dim, use_swarm=True, rank=cfg.swarm_rank)
            for _ in range(cfg.num_experts)
        ])
        
        # Phase 5: Diffusion Core
        self.diffusion = nn.ModuleList([
            DiffusionLayer(cfg.hidden_dim, cfg.diffusion_heads)
            for _ in range(cfg.diffusion_layers)
        ])
        
        # Output: Text decoder (UNTIED — must learn independently)
        self.txt_dec = BitLinear(cfg.hidden_dim, cfg.vocab_size, bias=False)
    
    def forward(self, txt: torch.Tensor, gov_scale: float = 1.0) -> Dict[str, torch.Tensor]:
        B, L = txt.shape
        
        # Phase 1: Ingestion
        x = self.txt_emb(txt) + self.mod_emb(torch.zeros(B, L, dtype=torch.long, device=txt.device))
        
        # Phase 2: 9-Vector Decomposition
        x = self.decomposition(x)
        
        # Phase 3: Router dispatch (top-4 sparse)
        flat_x = x.view(-1, self.cfg.hidden_dim)
        logits = self.router(flat_x)
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_idx = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        
        # Phase 3: Expert computation
        flat_out = torch.zeros_like(flat_x)
        for k in range(self.cfg.top_k):
            idx = topk_idx[:, k]
            weight = topk_p[:, k].unsqueeze(-1)
            for e in range(self.cfg.num_experts):
                mask = (idx == e)
                if mask.any():
                    expert_out = self.experts[e](flat_x[mask], scale=gov_scale)
                    flat_out[mask] += expert_out * weight[mask]
        
        x = flat_out.view(B, L, self.cfg.hidden_dim)
        
        # Phase 5: Diffusion
        for layer in self.diffusion:
            x = layer(x)
        
        # Phase 6: Finalizer + Decode
        x = self.quillan_finalizer(x)
        logits = self.txt_dec(x)
        
        return {"logits": logits, "router_probs": probs}


# ═══════════════════════════════════════════════════════════════
# PROJECTION MATRICES
# ═══════════════════════════════════════════════════════════════

class ProjectionMatrices(nn.Module):
    """All projection matrices for dimension mismatch resolution."""
    def __init__(self):
        super().__init__()
        
        # Qwen projections (1024 -> 2048)
        self.qwen_hidden = nn.Linear(1024, 2048, bias=False)
        self.qwen_ffn_gate = nn.Linear(3584, 4096, bias=False)
        self.qwen_ffn_up = nn.Linear(3584, 4096, bias=False)
        self.qwen_ffn_down = nn.Linear(3584, 4096, bias=False)  # For down_proj input dim
        self.qwen_o_proj = nn.Linear(2048, 1024, bias=False)
        
        # BitNet projections (2560 -> 2048, 1728 -> 4096, 6912 -> 4096)
        self.bitnet_hidden = nn.Linear(2560, 2048, bias=False)
        self.bitnet_ffn_gate = nn.Linear(1728, 4096, bias=False)  # For gate_proj/up_proj
        self.bitnet_ffn_down = nn.Linear(4096, 6912, bias=False)  # For down_proj input projection (reversed)
        self.bitnet_down_output = nn.Linear(640, 2048, bias=False)  # For down_proj output projection
        
        self._init_projections()
    
    def _init_projections(self):
        """Initialize projections with SVD-based weights."""
        with torch.no_grad():
            # Qwen hidden: 1024 -> 2048
            W = torch.randn(2048, 1024)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.qwen_hidden.weight.data.copy_(U[:, :1024] @ torch.diag(S[:1024]) @ Vh[:1024, :])
            
            # Qwen FFN gate: 3584 -> 4096
            W = torch.randn(4096, 3584)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.qwen_ffn_gate.weight.data.copy_(U[:, :3584] @ torch.diag(S[:3584]) @ Vh[:3584, :])
            
            # Qwen FFN up: 3584 -> 4096 (same as gate)
            self.qwen_ffn_up.weight.data.copy_(self.qwen_ffn_gate.weight.clone())
            
            # Qwen FFN down: 3584 -> 4096
            W = torch.randn(4096, 3584)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.qwen_ffn_down.weight.data.copy_(U[:, :3584] @ torch.diag(S[:3584]) @ Vh[:3584, :])
            
            # Qwen o_proj: 2048 -> 1024
            W = torch.randn(1024, 2048)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.qwen_o_proj.weight.data.copy_(U[:, :1024] @ torch.diag(S[:1024]) @ Vh[:1024, :])
            
            # BitNet hidden: 2560 -> 2048
            W = torch.randn(2048, 2560)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.bitnet_hidden.weight.data.copy_(U[:, :2048] @ torch.diag(S[:2048]) @ Vh[:2048, :])
            
            # BitNet FFN gate: 1728 -> 4096
            W = torch.randn(4096, 1728)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.bitnet_ffn_gate.weight.data.copy_(U[:, :1728] @ torch.diag(S[:1728]) @ Vh[:1728, :])
            
            # BitNet FFN down: 6912 -> 4096 (reversed for matrix multiplication)
            W = torch.randn(6912, 4096)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.bitnet_ffn_down.weight.data.copy_(U[:, :4096] @ torch.diag(S[:4096]) @ Vh[:4096, :])
            
            # BitNet down output: 640 -> 2048
            W = torch.randn(2048, 640)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.bitnet_down_output.weight.data.copy_(U[:, :640] @ torch.diag(S[:640]) @ Vh[:640, :])


# ═══════════════════════════════════════════════════════════════
# WEIGHT TRANSPLANT
# ═══════════════════════════════════════════════════════════════

def load_source_tensor(path: str, key: str) -> torch.Tensor:
    """Load a single tensor from safetensors."""
    with safe_open(path, framework='pt') as f:
        return f.get_tensor(key)


def transplant_llama_to_expert(
    model: QuillanRoninSovereign,
    expert_idx: int,
    layer_idx: int,
    proj: ProjectionMatrices
):
    """Transplant Llama layer to Quillan expert."""
    path = os.path.join(BASE, "llama model.safetensors")
    prefix = f"model.layers.{layer_idx}"
    
    # Attention (direct copy for q/o, tile for k/v)
    q = load_source_tensor(path, f"{prefix}.self_attn.q_proj.weight")
    model.diffusion[expert_idx % model.cfg.diffusion_layers].q_proj.weight.data.copy_(q)
    
    k = load_source_tensor(path, f"{prefix}.self_attn.k_proj.weight")
    # Tile 512 -> 2048 (4 copies)
    k_tiled = k.repeat(4, 1)[:2048, :]
    model.diffusion[expert_idx % model.cfg.diffusion_layers].k_proj.weight.data.copy_(k_tiled)
    
    v = load_source_tensor(path, f"{prefix}.self_attn.v_proj.weight")
    v_tiled = v.repeat(4, 1)[:2048, :]
    model.diffusion[expert_idx % model.cfg.diffusion_layers].v_proj.weight.data.copy_(v_tiled)
    
    o = load_source_tensor(path, f"{prefix}.self_attn.o_proj.weight")
    model.diffusion[expert_idx % model.cfg.diffusion_layers].o_proj.weight.data.copy_(o)
    
    # LayerNorm
    ln1 = load_source_tensor(path, f"{prefix}.input_layernorm.weight")
    model.diffusion[expert_idx % model.cfg.diffusion_layers].norm1.weight.data.copy_(ln1)
    
    ln2 = load_source_tensor(path, f"{prefix}.post_attention_layernorm.weight")
    model.diffusion[expert_idx % model.cfg.diffusion_layers].norm2.weight.data.copy_(ln2)
    
    # FFN (slice 8192 -> 4096)
    gate = load_source_tensor(path, f"{prefix}.mlp.gate_proj.weight")
    model.experts[expert_idx].w1.weight.data.copy_(gate[:4096, :])
    
    up = load_source_tensor(path, f"{prefix}.mlp.up_proj.weight")
    model.experts[expert_idx].w1.weight.data.copy_(up[:4096, :])  # gate and up can share
    
    down = load_source_tensor(path, f"{prefix}.mlp.down_proj.weight")
    model.experts[expert_idx].w2.weight.data.copy_(down[:, :4096])
    
    logger.info(f"  Llama layer {layer_idx} -> Expert {expert_idx} (C{expert_idx}-{['ASTRA','VIR','SOLACE','PRAXIS','ECHO','OMNIS','LOGOS','METASYNTH'][expert_idx]})")


def transplant_qwen_to_expert(
    model: QuillanRoninSovereign,
    expert_idx: int,
    layer_idx: int,
    proj: ProjectionMatrices
):
    """Transplant Qwen layer to Quillan expert.
    
    Qwen architecture:
    - mtp.layers.0: single shared self-attention layer (q/k/v/o_proj)
    - model.language_model.layers.0-23: linear_attn + mlp per layer
    
    Strategy: Use mtp.layers.0 attention for all experts (shared initialization),
    use language_model.layers[layer_idx] FFN for each expert.
    """
    path = os.path.join(BASE, "qwen model.safetensors")
    mtp_prefix = "mtp.layers.0"  # Only layer 0 exists
    lm_prefix = f"model.language_model.layers.{layer_idx}"
    
    # Attention (from mtp.layers.0 - shared across all Qwen experts)
    q = load_source_tensor(path, f"{mtp_prefix}.self_attn.q_proj.weight").float()  # [4096, 1024]
    q_proj = (q @ proj.qwen_hidden.weight.T.float())[:2048, :]  # [2048, 2048]
    model.diffusion[expert_idx % model.cfg.diffusion_layers].q_proj.weight.data.copy_(q_proj)
    
    k = load_source_tensor(path, f"{mtp_prefix}.self_attn.k_proj.weight").float()  # [512, 1024]
    k_tiled = k.repeat(4, 1)[:2048, :]
    k_proj = (k_tiled @ proj.qwen_hidden.weight.T.float())
    model.diffusion[expert_idx % model.cfg.diffusion_layers].k_proj.weight.data.copy_(k_proj)
    
    v = load_source_tensor(path, f"{mtp_prefix}.self_attn.v_proj.weight").float()  # [512, 1024]
    v_tiled = v.repeat(4, 1)[:2048, :]
    v_proj = (v_tiled @ proj.qwen_hidden.weight.T.float())
    model.diffusion[expert_idx % model.cfg.diffusion_layers].v_proj.weight.data.copy_(v_proj)
    
    o = load_source_tensor(path, f"{mtp_prefix}.self_attn.o_proj.weight").float()  # [1024, 2048]
    o_padded = torch.zeros(2048, 2048)
    o_padded[:1024, :] = o
    model.diffusion[expert_idx % model.cfg.diffusion_layers].o_proj.weight.data.copy_(o_padded)
    
    # LayerNorm (from language_model.layers for per-layer normalization)
    ln1 = load_source_tensor(path, f"{lm_prefix}.input_layernorm.weight").float()
    ln1_proj = proj.qwen_hidden(ln1.unsqueeze(0)).squeeze(0)
    model.diffusion[expert_idx % model.cfg.diffusion_layers].norm1.weight.data.copy_(ln1_proj)
    
    ln2 = load_source_tensor(path, f"{lm_prefix}.post_attention_layernorm.weight").float()
    ln2_proj = proj.qwen_hidden(ln2.unsqueeze(0)).squeeze(0)
    model.diffusion[expert_idx % model.cfg.diffusion_layers].norm2.weight.data.copy_(ln2_proj)
    
    # FFN (from model.language_model.layers)
    gate = load_source_tensor(path, f"{lm_prefix}.mlp.gate_proj.weight").float()  # [3584, 1024]
    gate_proj = (gate @ proj.qwen_hidden.weight.T.float())  # [3584, 2048]
    gate_final = proj.qwen_ffn_gate(gate_proj.T).T  # [4096, 2048]
    model.experts[expert_idx].w1.weight.data.copy_(gate_final)
    
    down = load_source_tensor(path, f"{lm_prefix}.mlp.down_proj.weight").float()  # [1024, 3584]
    down_step1 = proj.qwen_hidden.weight @ down  # [2048, 3584]
    down_step2 = down_step1 @ proj.qwen_ffn_down.weight.T  # [2048, 4096]
    model.experts[expert_idx].w2.weight.data.copy_(down_step2)
    
    logger.info(f"  Qwen layer {layer_idx} -> Expert {expert_idx} (C{expert_idx})")
    
    logger.info(f"  Qwen layer {layer_idx} -> Expert {expert_idx} (C{expert_idx})")


def transplant_bitnet_to_expert(
    model: QuillanRoninSovereign,
    expert_idx: int,
    layer_idx: int,
    proj: ProjectionMatrices
):
    """Transplant BitNet layer to Quillan expert."""
    path = os.path.join(BASE, "bitnet model.safetensors")
    prefix = f"model.layers.{layer_idx}"
    
    # Attention (with projection)
    q = load_source_tensor(path, f"{prefix}.self_attn.q_proj.weight")  # [640, 2560] uint8
    if q.dtype == torch.uint8:
        q = q.float() / 127.0  # Dequantize ternary
    q_proj = (q @ proj.bitnet_hidden.weight.T.float())  # [640, 2048]
    # Pad to 2048x2048
    q_padded = torch.zeros(2048, 2048)
    q_padded[:640, :] = q_proj
    model.diffusion[expert_idx % model.cfg.diffusion_layers].q_proj.weight.data.copy_(q_padded)
    
    k = load_source_tensor(path, f"{prefix}.self_attn.k_proj.weight")  # [160, 2560] uint8
    if k.dtype == torch.uint8:
        k = k.float() / 127.0
    k_proj = (k @ proj.bitnet_hidden.weight.T.float())  # [160, 2048]
    # Tile to 2048
    k_tiled = k_proj.repeat(13, 1)[:2048, :]
    model.diffusion[expert_idx % model.cfg.diffusion_layers].k_proj.weight.data.copy_(k_tiled)
    
    v = load_source_tensor(path, f"{prefix}.self_attn.v_proj.weight")  # [160, 2560] uint8
    if v.dtype == torch.uint8:
        v = v.float() / 127.0
    v_proj = (v @ proj.bitnet_hidden.weight.T.float())
    v_tiled = v_proj.repeat(13, 1)[:2048, :]
    model.diffusion[expert_idx % model.cfg.diffusion_layers].v_proj.weight.data.copy_(v_tiled)
    
    o = load_source_tensor(path, f"{prefix}.self_attn.o_proj.weight")  # [640, 2560] uint8
    if o.dtype == torch.uint8:
        o = o.float() / 127.0
    o_proj = (o @ proj.bitnet_hidden.weight.T.float())  # [640, 2048]
    o_padded = torch.zeros(2048, 2048)
    o_padded[:640, :] = o_proj
    model.diffusion[expert_idx % model.cfg.diffusion_layers].o_proj.weight.data.copy_(o_padded)
    
    # FFN (with projection)
    gate = load_source_tensor(path, f"{prefix}.mlp.gate_proj.weight")  # [1728, 2560] (uint8)
    if gate.dtype == torch.uint8:
        gate = gate.float() / 127.0  # Dequantize ternary
    gate_proj = (gate @ proj.bitnet_hidden.weight.T)  # [1728, 2048]
    gate_final = proj.bitnet_ffn_gate(gate_proj.T).T  # [4096, 2048]
    model.experts[expert_idx].w1.weight.data.copy_(gate_final)
    
    down = load_source_tensor(path, f"{prefix}.mlp.down_proj.weight")  # [640, 6912] uint8
    if down.dtype == torch.uint8:
        down = down.float() / 127.0
    # down_proj: [640, 6912] -> [2048, 4096]
    # new_weight = P_out @ weight @ P_in.T
    # [2048, 4096] = [2048, 640] @ [640, 6912] @ [6912, 4096]
    down_step1 = down @ proj.bitnet_ffn_down.weight  # [640, 6912] @ [6912, 4096] = [640, 4096]
    down_step2 = proj.bitnet_down_output.weight @ down_step1  # [2048, 640] @ [640, 4096] = [2048, 4096]
    model.experts[expert_idx].w2.weight.data.copy_(down_step2)
    
    logger.info(f"  BitNet layer {layer_idx} -> Expert {expert_idx} (C{expert_idx})")


def transplant_embeddings(model: QuillanRoninSovereign):
    """Transplant embeddings from Llama (exact match)."""
    path = os.path.join(BASE, "llama model.safetensors")
    emb = load_source_tensor(path, "model.embed_tokens.weight")
    # Take first vocab_size rows
    model.txt_emb.weight.data.copy_(emb[:model.cfg.vocab_size, :])
    logger.info(f"  Embeddings: {emb.shape} -> {model.txt_emb.weight.shape}")


def run_transplant():
    """Execute the full weight transplant."""
    logger.info("=" * 60)
    logger.info("QUILLAN-RONIN PHASE 2+3: PROJECTION + TRANSPLANT")
    logger.info("=" * 60)
    
    cfg = QuillanConfig()
    model = QuillanRoninSovereign(cfg)
    proj = ProjectionMatrices()
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    logger.info(f"Projections created: {sum(p.numel() for p in proj.parameters())/1e6:.1f}M params")
    
    # Transplant embeddings
    logger.info("\n--- Transplanting Embeddings ---")
    transplant_embeddings(model)
    
    # Transplant Llama experts (C0-C7)
    logger.info("\n--- Transplanting Llama -> Experts C0-C7 ---")
    for i in range(8):
        transplant_llama_to_expert(model, i, i, proj)
    
    # Transplant Qwen experts (C8-C21)
    logger.info("\n--- Transplanting Qwen -> Experts C8-C21 ---")
    for i in range(14):
        transplant_qwen_to_expert(model, 8 + i, i, proj)
    
    # Transplant BitNet experts (C22-C33)
    logger.info("\n--- Transplanting BitNet -> Experts C22-C33 ---")
    for i in range(12):
        transplant_bitnet_to_expert(model, 22 + i, i, proj)
    
    # Save
    output_path = "quillan_transplanted.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "proj_state_dict": proj.state_dict(),
    }, output_path)
    
    logger.info(f"\nTransplanted model saved to {output_path}")
    logger.info("=" * 60)
    logger.info("PHASE 2+3 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_transplant()
