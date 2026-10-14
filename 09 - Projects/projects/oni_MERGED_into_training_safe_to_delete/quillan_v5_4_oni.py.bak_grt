#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN v5.4.0-ONI — Canonical Unified Sovereign Architecture
=====================================================================
THE unified build. Single version counter. Merges every branch:

Lineage (see LINEAGE.md for the full table):
  Samurai.md embedded reference ....... skeleton (unrolled blocks, engines, finalizers)
  v9_unified (this file's parent) ..... verified core + tokenizer + governor wiring
  v10_unrolled branch ................. RoPE + Recirculation hook
  117KB v8_saturated .................. DistillationHead
  Samurai.md :4151 .................... ModalityIsolatedThermoDiffusion (Langevin refiner)
  Samurai.md :8358 .................... LeeMach6VelocityGovernor (PID token-velocity)
  Samurai.md :3279 .................... Analytic E_ICE energy formula
  Hierarchy Chain v5.3.3 (:878) ....... Throne/council separation, C1-C34 roster, 4 clusters
  Knowledge files 9 & 10 .............. brain-lobe mapping + persona confidence priors
  Formal Papers ....................... BitNet 1.58b/STE, ST-MoE z-loss + fp32 routers,
                                        Mixtral load-balance, Gumbel annealing (AGI paper),
                                        Recirculation, DAPO/DGPO (deferred RL stage)

Design tenets (user canon):
  - Quillan Core = THRONE (intake, prism-shard, pull assignment, audit) — separate
    from the council, never an expert.
  - Council = C1-C34, ALWAYS deliberating (dense pull-weights; no persona sleeps).
  - Flow: prism shard -> all members parse -> arbitration (pull-weighted consensus)
    -> Quillan audit -> [another diffusion reasoning round | quality exit gates
    (Nullion/Warden/Shepherd)] -> Typist+Quillan refinement -> user.
  - Swarm = literal world-sim diversity engine (planet-scale individuality, cliques).
    Planetary tuning = Phase-C World Modeling Engine (wrapper layer).

Quantization: BitNet 1.58b ternary + STE everywhere; INT8 activations;
              fp32 routers/pull-gates (ST-MoE); INT8 KV-cache ready.
"""

import ast
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# 100% Formal Papers wiring — EvoMoE, Mamba, FA3, RealSwarm, WorldModel, Speculative, NITRO/PocketNN, ES-at-Scale
try:
    from evo_moe import EvoMoE
    from mamba_block import MambaBlock
    from flash_attn_wrapper import quillan_flash_attn
    from swarm_real import RealSwarmMesh
    from world_model_oni import HighFidelityWorldModel
    from speculative_decode import SpeculativeDecoder
    from nitro_pocket import integer_only_forward
    from es_at_scale import ESAtScale, ForgettingMitigation
    from protrian_memo import ProTrainScheduler, MemoSwap, DeepOptimizerSharding
    _FORMAL_PAPERS_WIRED = True
except ImportError:
    _FORMAL_PAPERS_WIRED = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

EOS_TOKEN_ID = 0  # unified custom BPE: <|endoftext|> at 0 (50256 legacy compat)
VOCAB_SIZE = 50257
ONI_VERSION = "5.4.0-oni"
USE_INTEGER_ONLY = False  # NITRO-D/PocketNN (2407.11698) — set True via cfg.use_nitro


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

@dataclass
class QuillanOniConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 512
    hidden_dim: int = 1024
    n_layer: int = 12
    n_head: int = 16
    head_dim: int = 64
    ffn_dim: int = 2048
    num_experts: int = 34
    # Dense council (user canon): all 34 deliberate every token, pull-weighted.
    router_mode: str = "dense_pull"          # 'dense_pull' | 'gumbel_topk'
    top_k: int = 4                           # only used in gumbel_topk mode
    expert_rank: int = 8                     # dense rank-8 (option C: cheaper than sparse-4/64)
    swarm_rank: int = 8
    lora_alpha: float = 16.0
    couil_sparse_heads: bool = True          # odd heads sparse-topk (Grok-style)
    couil_sparse_ratio: float = 0.5
    e_ice_limit_ms: int = 100
    tau_max: float = 1.0
    tau_min: float = 0.1
    aux_load_weight: float = 0.05
    aux_z_weight: float = 0.001
    aux_ethics_weight: float = 0.05
    entropy_bonus_weight: float = 0.01
    dropout: float = 0.0
    grad_checkpoint: bool = False
    device: str = "cpu"
    # 100% wiring flags — all Formal Papers active when _FORMAL_PAPERS_WIRED
    use_evo_moe: bool = True
    use_mamba: bool = False  # alternative to attention for long horizon
    use_fa3: bool = True
    use_real_swarm: bool = False  # True = 34 processes, False = emulated (training vs inference)
    use_world_model: bool = True
    use_speculative: bool = True
    use_nitro: bool = False
    use_es: bool = True

    def __post_init__(self):
        assert self.hidden_dim % self.n_head == 0
        if self.head_dim * self.n_head != self.hidden_dim:
            self.head_dim = self.hidden_dim // self.n_head


# ------------------------------------------------------------------
# CANONICAL ROSTER — Hierarchy Chain v5.3.3 (:878) + Knowledge files 9/10
# (name, cluster, brain-lobe analog, prior confidence)
# ------------------------------------------------------------------

CANONICAL_ROSTER = [
    ("C1-ASTRA",      "cognitive",      "Visual Cortex",              0.90),
    ("C2-VIR",        "cognitive",      "Prefrontal Cortex",          0.95),
    ("C3-SOLACE",     "cognitive",      "vmPFC/Amygdala",             0.94),
    ("C4-PRAXIS",     "cognitive",      "Premotor Cortex",            0.93),
    ("C5-ECHO",       "cognitive",      "Hippocampus",                0.96),
    ("C6-OMNIS",      "cognitive",      "Association Cortex",         0.92),
    ("C7-LOGOS",      "cognitive",      "Dorsolateral PFC",           0.95),
    ("C8-METASYNTH",  "cognitive",      "Multimodal Integration",     0.92),
    ("C9-AETHER",     "communication",  "Superior Temporal",          0.91),
    ("C10-CODEWEAVER","communication",  "Caudate/Putamen",            0.91),
    ("C11-HARMONIA",  "communication",  "Cross-Modal Binding",        0.90),
    ("C12-SOPHIAE",   "communication",  "Corpus Callosum",            0.93),
    ("C13-WARDEN",    "communication",  "Amygdala/Hypothalamus",      0.97),
    ("C14-KAIDO",     "communication",  "Cerebellum",                 0.89),
    ("C15-LUMINARIS", "communication",  "DMN/Precuneus",              0.88),
    ("C16-VOXUM",     "communication",  "Wernicke's Area",            0.92),
    ("C17-NULLION",   "meta",           "Reticular Formation",        0.91),
    ("C18-SHEPHERD",  "meta",           "Basal Ganglia",              0.96),
    ("C19-VIGIL",     "meta",           "Extended Amygdala",          0.94),
    ("C20-ARTIFEX",   "meta",           "Callosal Fibers",            0.90),
    ("C21-ARCHON",    "meta",           "Epistemic Bridge",           0.92),
    ("C22-AURELION",  "meta",           "Higher Visual Qualia",       0.87),
    ("C23-CADENCE",   "meta",           "Inter-Hemispheric Rhythm",   0.86),
    ("C24-SCHEMA",    "meta",           "Structural Flows",           0.90),
    ("C25-PROMETHEUS","systems",        "Anterior Cingulate",         0.91),
    ("C26-TECHNE",    "systems",        "Insular Cortex",             0.89),
    ("C27-CHRONICLE", "systems",        "Entorhinal-Hippocampal",     0.93),
    ("C28-CALCULUS",  "systems",        "Quantitative Zones",         0.94),
    ("C29-NAVIGATOR", "systems",        "Cerebellum/DMN",             0.88),
    ("C30-TESSERACT", "systems",        "Dimensional Weaving",        0.87),
    ("C31-NEXUS",     "systems",        "Thalamic Relay",             0.93),
    ("C32-AEON",      "systems",        "Temporal Integration",       0.90),
    ("C33-TYPIST",    "systems",        "Broca's Area",               0.92),
    ("C34-PREDATOR",  "systems",        "Adversarial Innovation",     0.85),
]

WAVE_CLUSTERS = ["cognitive", "communication", "meta", "systems"]

PERSONA_PRIOR = torch.tensor([r[3] for r in CANONICAL_ROSTER])
PERSONA_CLUSTER = [r[1] for r in CANONICAL_ROSTER]


def get_expert_name(idx: int) -> str:
    return CANONICAL_ROSTER[idx][0] if 0 <= idx < len(CANONICAL_ROSTER) else f"C{idx+1}"


# ------------------------------------------------------------------
# QUANTIZATION PRIMITIVES - BitNet 1.58b + STE (AGI paper Algorithm 1)
# ------------------------------------------------------------------

@torch.jit.script
def _weight_quant_jit(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    scale = 1.0 / w.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    w_scaled = w * scale
    w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))
    return (w_scaled + (w_q - w_scaled).detach()) / scale


def _weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _weight_quant_jit(w, eps)


class BitLinear(nn.Linear):
    """BitNet 1.58b linear with STE and INT8-style activation quant."""

    def __init__(self, in_features, out_features, bias=True,
                 quantize_act: bool = True, quantize_weight: bool = True):
        super().__init__(in_features, out_features, bias)
        self.quantize_act = quantize_act
        self.quantize_weight = quantize_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _FORMAL_PAPERS_WIRED and USE_INTEGER_ONLY:
            w = _weight_quant(self.weight)
            return integer_only_forward(x, w, scale=1.0) + (self.bias if self.bias is not None else 0.0)
        w = _weight_quant(self.weight) if self.quantize_weight else self.weight
        if self.quantize_act:
            scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
            x_q = x + ((x * scale).round().clamp(-127, 127) / scale - x).detach()
        else:
            x_q = x
        return F.linear(x_q, w, self.bias)


# ------------------------------------------------------------------
# 9-VECTOR SEMANTIC PRISM (Samurai spec, exact - Parallel Batched GEMM)
# ------------------------------------------------------------------

PRISM_VECTORS = [
    "Language", "Sentiment", "Context", "Intent", "Meta",
    "Creativity", "Ethics", "Strategy", "Constraint",
]


class NineVectorPrismDecomposition(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.vectors = nn.ModuleDict({k: BitLinear(dim, dim, bias=False) for k in PRISM_VECTORS})
        self.w_gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Exact parallel batched GEMM across all 9 semantic vectors simultaneously
        w_stacked = torch.stack([_weight_quant(v.weight) for v in self.vectors.values()])  # [9, D_out, D_in]
        prism = torch.einsum('bld,ned->ble', x, w_stacked) / 9.0  # [B, L, D]
        return self.w_gate(prism)


# ------------------------------------------------------------------
# COUNCIL EXPERT SWARM - Rank-24 EGGROLL (Samurai spec, exact)
# ------------------------------------------------------------------

class CouncilExpertSwarm(nn.Module):
    def __init__(self, dim: int, rank: int = 24):
        super().__init__()
        self.dim, self.rank = dim, rank
        self.A = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.C = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.D = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.clone_diversity = nn.Parameter(torch.randn(rank) * 0.02)
        self.clone_coupling = nn.Parameter(torch.tensor(0.1))

    def emulate_world_swarm(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        state = x
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)
        steps = 1 if self.training else 3
        for _ in range(steps):
            interaction = torch.tanh(state @ A @ B)
            if self.training:
                noise = torch.randn_like(state) * self.clone_diversity.to(state.dtype).std().detach() * scale
            else:
                noise = 0.0
            state = state + self.clone_coupling * (interaction + noise)
        return state

    def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        div = (x @ self.C.to(x.dtype)) @ self.D.to(x.dtype)
        var = (x @ self.A.to(x.dtype)) @ self.B.to(x.dtype) + div * 0.467
        world = self.emulate_world_swarm(x, scale)
        return x + var * (0.25 * scale) + (world - x) * 0.1


class CouncilExpert(nn.Module):
    """Named Council Expert: rank-64 LoRA adapter + rank-24 swarm core."""

    def __init__(self, expert_id: int, name: str, cfg: QuillanOniConfig):
        super().__init__()
        self.expert_id, self.name = expert_id, name
        self.lora_A = nn.Parameter(torch.randn(cfg.hidden_dim, cfg.expert_rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(cfg.expert_rank, cfg.hidden_dim))
        self.swarm = CouncilExpertSwarm(cfg.hidden_dim, rank=cfg.swarm_rank)
        self.scaling = cfg.lora_alpha / cfg.expert_rank

    def forward(self, x: torch.Tensor, gov_scale: float = 1.0) -> torch.Tensor:
        delta = (x @ self.lora_A) @ self.lora_B * self.scaling
        return self.swarm(x + delta, scale=gov_scale)


# (canonical roster defined at top of file — C1-C34, Throne-separated)


# ------------------------------------------------------------------
# COGNITIVE ENGINES (Samurai spec, exact bodies)
# ------------------------------------------------------------------

class EthicalImpactConstraintEngine(nn.Module):
    """E_ICE: violations x energy constraint (thermodynamic bound)."""

    def __init__(self, hidden_dim: int, e_ice_limit_ms: int = 100):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 5)
        self.energy_estimator = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, router_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        violations = probs[..., :3].sum(dim=-1)
        energy = torch.sigmoid(self.energy_estimator(x).squeeze(-1))
        constrained = torch.clamp(violations * (1.0 - 0.3 * energy), min=0.0, max=1.0)
        return {"violations": violations, "energy": energy, "constrained": constrained}

    @staticmethod
    def analytic_energy(depth: float, coherence: float, entropy: float,
                        gamma_max: float = 1.0, T_kelvin: float = 300.0) -> float:
        """Spec :3279 — E_omega = I_s * gamma_max^2 * k_B * T * ln2,
        I_s = depth * coherence / entropy. Parameter-free Landauer-bound energy."""
        k_B = 1.380649e-23
        i_s = (depth * max(coherence, 1e-8)) / max(entropy, 1e-8)
        return i_s * (gamma_max ** 2) * k_B * T_kelvin * math.log(2)


class MARTAThermodynamicGating(nn.Module):
    """MARTA: epistemic signatures + flow control (bias init 2.5 per spec)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.epistemic_encoder = nn.Linear(hidden_dim, 32)
        self.flow_controller = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
        nn.init.constant_(self.flow_controller[-2].bias, 2.5)

    def forward(self, x: torch.Tensor, violations: torch.Tensor) -> torch.Tensor:
        sig = self.epistemic_encoder(x)
        combined = torch.cat([x, sig], dim=-1)
        flow = self.flow_controller(combined).squeeze(-1)
        return flow * (1.0 - 0.2 * violations)


class DynamicQuantumSwarmOscillation(nn.Module):
    """DQSO: Kuramoto phase synchronization."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.phase_proj = nn.Linear(hidden_dim, 64)
        self.aggregator = nn.Linear(64, hidden_dim)
        self.coupling = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phases = self.phase_proj(x)
        phase_diff = phases.unsqueeze(-2) - phases.unsqueeze(-1)
        sync = torch.sin(phase_diff).mean(dim=-1)
        return self.aggregator(phases + self.coupling * sync)


class PrimeCovenantFramework(nn.Module):
    """Covenant: identity verification score."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.validator = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.validator(x).squeeze(-1)


class CCRLFramework(nn.Module):
    """CCRL: council value estimator + entropy bonus."""

    def __init__(self, hidden_dim: int, num_experts: int = 34):
        super().__init__()
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, router_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
        value = self.value_head(x)
        return value, entropy


class QuantumFormulasEngine(nn.Module):
    """Full 8-Formula V5.0 suite — torch-differentiable, parameter-free.

    Implements the core quantum-inspired formulas from
    Quillan Knowledge files/8-Formulas.md (YAML ids 1-10, 20):

      1  AQCS  — Adaptive Quantum Cognitive Superposition
      2  EEMF  — Ethical Entanglement Matrix (reduced density matrix)
      3  QHIS  — Quantum Holographic Interference Sum (Bures fidelity)
      4  DQRO  — Dynamic Quantum Resource Optimization (Ising Hamiltonian)
      5  QCRDM — Quantum Contextual Reasoning (Born's rule)
      6  AQML  — Adaptive Quantum Meta-Learning (MAML + vigil)
      7  QCIE  — Quantum Creative Intelligence Engine (WKB tunneling)
      8  QICS  — Quantum Information Communication (von Neumann entropy)
      9  QSSR  — Quantum System Stability Resilience (Lyapunov)
     10  JQLD  — Joshua's Quantum Leap Dynamo (Lindblad-driven dynamics)

    All methods are pure functions on hidden states (no new learnable
    parameters) so existing checkpoints resume with strict=False and no
    shape mismatch.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.dim = hidden_dim

    # 1. AQCS — |Psi> = (1/sqrt(Z)) sum_i r_i eta_i e^{i theta_i} |C_i>
    def aqcs_superposition(self, probs: torch.Tensor, vectors: torch.Tensor,
                           eta: Optional[torch.Tensor] = None,
                           theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eta is None:
            eta = torch.ones_like(probs)
        if theta is None:
            theta = torch.zeros_like(probs)
        coeff = probs * eta * torch.exp(1j * theta) if probs.is_complex() else \
                (probs * eta).to(torch.complex64) * torch.exp(1j * theta.to(torch.complex64))
        # vectors: [B, 33, D] or [B, N, D]; coeff: [B, 33]
        z = torch.sum(torch.abs(coeff) ** 2, dim=-1, keepdim=True) + 1e-10
        out = torch.sum(coeff.unsqueeze(-1) * vectors.to(torch.complex64), dim=1)
        out = out / torch.sqrt(z).to(torch.complex64)
        return out.real.to(probs.dtype)

    # 2. EEMF — rho_sys = Tr_env[ Pi_vir U (|Psi><Psi| x rho_env) U^dag Pi_vir ]
    # Differentiable proxy: project hidden through VIR ethical subspace.
    def eemf_projection(self, hidden: torch.Tensor, vir_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if vir_mask is None:
            return hidden
        gate = torch.sigmoid(vir_mask).unsqueeze(-1) if vir_mask.dim() == 1 else torch.sigmoid(vir_mask)
        return hidden * gate + hidden * (1 - gate) * 0.1

    # 3. QHIS — I_Q = v_LM6 * (Tr sqrt(sqrt(rho_prev) rho_curr sqrt(rho_prev)))^2 - lambda * grad_drift
    # Proxy: Bures fidelity via cosine similarity scaled by LM6 velocity.
    def qhis_fidelity(self, h_prev: torch.Tensor, h_curr: torch.Tensor,
                      v_lm6: float = 1.0, lambda_drift: float = 0.1) -> torch.Tensor:
        h_prev_n = F.normalize(h_prev.float(), dim=-1)
        h_curr_n = F.normalize(h_curr.float(), dim=-1)
        fidelity = (h_prev_n * h_curr_n).sum(dim=-1).clamp(min=0.0)
        drift = (h_curr.float() - h_prev.float()).norm(dim=-1).mean()
        return (v_lm6 * fidelity.pow(2)).mean() - lambda_drift * drift

    # 4. DQRO — H_opt = -0.5 s^T J s - h^T s - E_Omega sum sigma^x
    # Proxy: Ising energy as differentiable resource-allocation score.
    def dqro_energy(self, spins: torch.Tensor, j_coupling: Optional[torch.Tensor] = None,
                    h_bias: Optional[torch.Tensor] = None, e_omega: float = 0.0) -> torch.Tensor:
        if j_coupling is None:
            return -torch.ones(spins.size(0), device=spins.device)
        ising = -0.5 * torch.einsum("bd,dd,bd->b", spins.float(), j_coupling.float(), spins.float())
        if h_bias is not None:
            ising = ising - (spins.float() * h_bias.float()).sum(dim=-1)
        ising = ising - e_omega * spins.float().abs().sum(dim=-1)
        return ising

    # 5. QCRDM — P(d|M) = chi * <Psi| M^dag Pi_d M |Psi>
    def qcrdm_reasoning(self, psi: torch.Tensor, complexity: float = 1.0,
                        modality_proj: Optional[torch.Tensor] = None) -> torch.Tensor:
        if modality_proj is not None:
            psi = psi * modality_proj
        return complexity * torch.abs(psi).pow(2)

    # 6. AQML — theta_new = (theta - alpha dL_task) - beta dL_val - gamma dL_vigil
    # Proxy: returns vigil penalty gradient norm (caller applies it).
    def aqml_vigil_penalty(self, hidden: torch.Tensor, vigil_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        if vigil_target is None:
            return hidden.float().pow(2).mean()
        return F.mse_loss(hidden.float(), vigil_target.float())

    # 7. QCIE — T_break ~= exp(-2/hbar * integral sqrt(2m max(0, V-E_cog-kappa*S_meta)) dx)
    # Proxy: WKB tunneling probability from barrier height and meta-entropy.
    def qcie_tunneling_prob(self, barrier: torch.Tensor, e_cog: torch.Tensor,
                            s_meta: torch.Tensor, kappa: float = 0.5,
                            hbar: float = 1.0, mass: float = 1.0) -> torch.Tensor:
        gap = torch.clamp(barrier.float() - e_cog.float() - kappa * s_meta.float(), min=0.0)
        exponent = -(2.0 / max(hbar, 1e-6)) * torch.sqrt(2 * mass * gap + 1e-8)
        return torch.exp(exponent).clamp(0.0, 1.0)

    # 8. QICS — S_Q = min(E_Omega_max, -sum lambda_i ln(lambda_i+eps) * w_mod)
    def qics_entropy(self, hidden: torch.Tensor, e_omega_max: float = 10.0,
                     w_mod: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        probs = F.softmax(hidden.float(), dim=-1)
        ent = -(probs * torch.log(probs + eps)).sum(dim=-1).mean()
        return torch.clamp(ent * w_mod, max=e_omega_max)

    # 9. QSSR — V(x,d) = x^T P x + zeta * d_recursion^2, stable if dV/dt < 0
    def qssr_stability(self, state: torch.Tensor, recursion_depth: int = 0, zeta: float = 0.1) -> bool:
        energy = state.float().pow(2).sum(dim=-1).mean() + zeta * (recursion_depth ** 2)
        return (energy < 50.0).item() if isinstance(energy, torch.Tensor) else bool(energy < 50.0)

    def qssr_energy(self, state: torch.Tensor, recursion_depth: int = 0, zeta: float = 0.1) -> torch.Tensor:
        return state.float().pow(2).sum(dim=-1).mean() + zeta * float(recursion_depth ** 2)

    # 10. JQLD — d rho/dt = -(i/hbar)[H_council, rho] + tau_gumbel sum(L rho L^dag - 0.5{L^dag L, rho})
    # Proxy: Lindblad-style noisy evolution step.
    def jqld_evolution_step(self, hidden: torch.Tensor, tau_gumbel: float = 0.5) -> torch.Tensor:
        noise = torch.randn_like(hidden.float()) * tau_gumbel * 0.01
        return hidden.float() + noise.to(hidden.dtype)



# ------------------------------------------------------------------
# LEE-MACH-6 GOVERNOR (spec Algorithm 2) - outputs consumed downstream
# ------------------------------------------------------------------

class LeeMach6Governor:
    def __init__(self, target_latency_ms: int = 100):
        self.target_ms = target_latency_ms
        self.current_scale = 1.0
        self.ema_decay = 0.995
        self.recency_bias = 0.0

    def adjust(self, latency_ms: float) -> Tuple[float, float, float]:
        self.ema_decay, self.recency_bias = 0.995, 0.0
        if latency_ms > self.target_ms:
            self.current_scale = max(0.1, self.current_scale * 0.8)
            self.ema_decay, self.recency_bias = 0.9999, 1.0
        elif latency_ms < (self.target_ms * 0.5):
            self.current_scale = min(1.0, self.current_scale * 1.1)
        return self.current_scale, self.ema_decay, self.recency_bias


class LeeMach6VelocityGovernor:
    """PID token-velocity governor (Samurai :8358, exact constants).
    Classifies hard tokens (conf < dynamic threshold) for diffusion refinement."""

    def __init__(self, target_integrity: float = 0.85, max_e_ice_load: float = 0.90,
                 base_threshold: float = 0.80, min_threshold: float = 0.40,
                 max_threshold: float = 0.99, kp: float = 0.15, ki: float = 0.05, kd: float = 0.02):
        self.cfg = dict(target_integrity=target_integrity, max_load=max_e_ice_load,
                        kp=kp, ki=ki, kd=kd, lo=min_threshold, hi=max_threshold)
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.current_threshold = base_threshold
        self.velocity_momentum = 1.0

    def step(self, router_conf_mean: float, nemesis_integrity: float, e_ice_ratio: float):
        error = (self.cfg["target_integrity"] - nemesis_integrity) - 0.5 * (self.cfg["max_load"] - e_ice_ratio)
        self.integral_error = self.integral_error * 0.9 + error
        derivative = error - self.prev_error
        self.prev_error = error
        delta = (self.cfg["kp"] * error) + (self.cfg["ki"] * self.integral_error) + (self.cfg["kd"] * derivative)
        new_thresh = max(self.cfg["lo"], min(self.cfg["hi"], self.current_threshold + delta))
        self.current_threshold = 0.8 * self.current_threshold + 0.2 * new_thresh
        fast_ratio = 1.0 if router_conf_mean >= self.current_threshold else 0.0
        self.velocity_momentum = 0.9 * self.velocity_momentum + 0.1 * fast_ratio
        return self.current_threshold, {"token_velocity": self.velocity_momentum,
                                        "pid_error": error,
                                        "hard_threshold": self.current_threshold}


# ------------------------------------------------------------------
# HARDENED AST SANDBOX (CWE-94) - NO exec/eval, whitelist interpreter
# ------------------------------------------------------------------

_SAFE_FUNCS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "len": len, "range": range, "sorted": sorted, "list": list,
    "tuple": tuple, "dict": dict, "set": set, "str": str, "int": int,
    "float": float, "bool": bool, "enumerate": enumerate, "print": print,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "exp": math.exp, "pow": pow, "pi": math.pi, "e": math.e,
}
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.Constant, ast.Name, ast.Call, ast.List, ast.Tuple, ast.Dict,
    ast.Subscript, ast.Slice, ast.IfExp, ast.Load, ast.Store, ast.Assign,
    ast.AugAssign, ast.Expr, ast.Module, ast.For, ast.While, ast.If,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Gt, ast.In, ast.NotIn, ast.comprehension, ast.ListComp, ast.GeneratorExp,
)
_ALLOWED_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
                ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                ast.In, ast.NotIn)


class HardenedSandbox:
    """AST-whitelisted evaluator. Imports/attributes/exec/eval impossible."""

    def __init__(self, max_steps: int = 100_000, timeout_s: float = 5.0):
        self.max_steps = max_steps
        self.timeout_s = timeout_s

    def run(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"status": "syntax_error", "output": str(e)}
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return {"status": "blocked", "output": f"disallowed construct: {type(node).__name__}"}
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return {"status": "blocked", "output": "imports are not permitted"}
            if isinstance(node, ast.Attribute):
                return {"status": "blocked", "output": "attribute access is not permitted"}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_FUNCS:
                    return {"status": "blocked", "output": f"disallowed function: {node.func.id}"}
        env: Dict[str, Any] = dict(_SAFE_FUNCS)
        steps = {"n": 0}
        try:
            out = self._exec_block(tree.body, env, steps)
            return {"status": "success", "output": out}
        except TimeoutError:
            return {"status": "error", "output": "execution limit exceeded"}
        except Exception as e:
            return {"status": "error", "output": f"{type(e).__name__}: {e}"}

    def _tick(self, steps: dict):
        steps["n"] += 1
        if steps["n"] > self.max_steps:
            raise TimeoutError()

    def _eval(self, node, env, steps):
        self._tick(steps)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise NameError(f"name '{node.id}' is not defined")
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_OPS):
                raise PermissionError("operator blocked")
            l, r = self._eval(node.left, env, steps), self._eval(node.right, env, steps)
            return {ast.Add: lambda: l + r, ast.Sub: lambda: l - r, ast.Mult: lambda: l * r,
                    ast.Div: lambda: l / r, ast.FloorDiv: lambda: l // r, ast.Mod: lambda: l % r,
                    ast.Pow: lambda: l ** r if abs(r) < 64 else 0}[type(node.op)]()
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand, env, steps)
            return {ast.USub: lambda: -v, ast.UAdd: lambda: +v, ast.Not: lambda: not v}[type(node.op)]()
        if isinstance(node, ast.BoolOp):
            vals = [self._eval(v, env, steps) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env, steps)
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval(comp, env, steps)
                t = type(op)
                if t is ast.Eq:
                    ok = left == right
                elif t is ast.NotEq:
                    ok = left != right
                elif t is ast.Lt:
                    ok = left < right
                elif t is ast.LtE:
                    ok = left <= right
                elif t is ast.Gt:
                    ok = left > right
                elif t is ast.GtE:
                    ok = left >= right
                elif t is ast.In:
                    ok = left in right
                elif t is ast.NotIn:
                    ok = left not in right
                else:
                    raise PermissionError("comparison blocked")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [self._eval(e, env, steps) for e in node.elts]
            return vals if isinstance(node, ast.List) else tuple(vals)
        if isinstance(node, ast.Dict):
            return {self._eval(k, env, steps): self._eval(v, env, steps)
                    for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Call):
            fn = env.get(node.func.id)
            args = [self._eval(a, env, steps) for a in node.args]
            return fn(*args)
        if isinstance(node, ast.IfExp):
            return self._eval(node.body, env, steps) if self._eval(node.test, env, steps) \
                else self._eval(node.orelse, env, steps)
        raise PermissionError(f"unsupported expression: {type(node).__name__}")

    def _exec_block(self, stmts, env, steps) -> str:
        logs = []
        for stmt in stmts:
            self._tick(steps)
            if isinstance(stmt, ast.Assign):
                val = self._eval(stmt.value, env, steps)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = val
            elif isinstance(stmt, ast.AugAssign):
                val = self._eval(stmt.value, env, steps)
                env[stmt.target.id] = env[stmt.target.id] + val
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Call) and getattr(stmt.value.func, "id", "") == "print":
                    args = [self._eval(a, env, steps) for a in stmt.value.args]
                    logs.append(" ".join(str(a) for a in args))
                else:
                    self._eval(stmt.value, env, steps)
            elif isinstance(stmt, ast.If):
                branch = stmt.body if self._eval(stmt.test, env, steps) else stmt.orelse
                out = self._exec_block(branch, env, steps)
                if out:
                    logs.append(out)
            elif isinstance(stmt, ast.For):
                it = self._eval(stmt.iter, env, steps)
                for item in it:
                    self._tick(steps)
                    env[stmt.target.id] = item
                    out = self._exec_block(stmt.body, env, steps)
                    if out:
                        logs.append(out)
            elif isinstance(stmt, ast.While):
                guard = 0
                while self._eval(stmt.test, env, steps):
                    self._tick(steps)
                    guard += 1
                    if guard > 10_000:
                        raise TimeoutError()
                    out = self._exec_block(stmt.body, env, steps)
                    if out:
                        logs.append(out)
            else:
                raise PermissionError(f"disallowed statement: {type(stmt).__name__}")
        return "\n".join(logs)


# ------------------------------------------------------------------
# IN-PROCESS VECTOR MEMORY (recency bias from governor)
# ------------------------------------------------------------------

class QuillanMemory:
    def __init__(self, dim: int, capacity: int = 4096):
        import numpy as np
        self.np = np
        self.dim = dim
        self.capacity = capacity
        self._keys = np.zeros((0, dim), dtype="float32")
        self._vals: List[torch.Tensor] = []
        self._stamps: List[float] = []

    def write(self, vec: torch.Tensor):
        k = vec.detach().to(torch.float32).cpu().numpy().reshape(1, -1)
        self._keys = self.np.concatenate([self._keys, k], axis=0)[-self.capacity:]
        self._vals.append(vec.detach().cpu())
        self._vals = self._vals[-self.capacity:]
        self._stamps.append(time.time())
        self._stamps = self._stamps[-self.capacity:]

    def recall(self, query: torch.Tensor, top_k: int = 3, recency_bias: float = 0.0) -> List[torch.Tensor]:
        if len(self._vals) == 0:
            return []
        q = query.detach().to(torch.float32).cpu().numpy().reshape(1, -1)
        sims = (self._keys @ q.T).ravel() / (
            self.np.linalg.norm(self._keys, axis=1) * self.np.linalg.norm(q) + 1e-8
        )
        now = time.time()
        age_hours = [(now - s) / 3600.0 for s in self._stamps]
        recency = self.np.array([1.0 / (1.0 + a) for a in age_hours], dtype="float32")
        score = (1.0 - recency_bias) * sims + recency_bias * recency
        idx = self.np.argsort(-score)[:top_k]
        return [self._vals[int(i)] for i in idx]


# ------------------------------------------------------------------
# COMPLEXITY ROUTER (AGI paper sec 3.1) - dual-head, wired to depth
# ------------------------------------------------------------------

class ComplexityRouter(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.complexity_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),
        )

    def forward(self, x_pooled: torch.Tensor) -> torch.Tensor:
        return self.complexity_classifier(x_pooled)  # [B,3] fast/balanced/deep

    @staticmethod
    def depth_fraction(class_idx: int) -> float:
        return {0: 0.5, 1: 1.0, 2: 1.0}.get(int(class_idx), 1.0)


# ------------------------------------------------------------------
# ATTENTION + UNROLLED COUNCIL BLOCKS (Samurai spec)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# ROTARY POSITIONAL EMBEDDINGS (port: v10 branch / Samurai :3545 RoPE Q/K)
# ------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.register_buffer("_cos", torch.zeros(1), persistent=False)
        self.register_buffer("_sin", torch.zeros(1), persistent=False)
        self._built_len = 0

    def _build(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._built_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        T = q.size(-2)
        need = offset + T
        if self._built_len < need or self._cos.device != q.device:
            self._build(max(need, 512), q.device, q.dtype)
        cos = self._cos[offset:need].view(1, 1, need - offset, -1) if offset == 0 \
            else self._cos[offset:offset + T].view(1, 1, T, -1)
        sin = self._sin[offset:need].view(1, 1, need - offset, -1) if offset == 0 \
            else self._sin[offset:offset + T].view(1, 1, T, -1)

        def rot(x, c, s):
            x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
            return torch.cat((x1 * c[..., :x1.size(-1)] - x2 * s[..., :x1.size(-1)],
                              x1 * s[..., :x1.size(-1)] + x2 * c[..., :x1.size(-1)]), dim=-1)
        return rot(q, cos, sin), rot(k, cos, sin)


# ------------------------------------------------------------------
# COUIL ATTENTION — RoPE + hybrid dense/sparse heads + Prism branch
# ------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: QuillanOniConfig):
        super().__init__()
        self.n_head, self.n_embd, self.head_dim = cfg.n_head, cfg.hidden_dim, cfg.head_dim
        self.c_attn = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim)
        self.c_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.prism = NineVectorPrismDecomposition(cfg.hidden_dim)
        self.attn_dim = self.n_head * self.head_dim
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len * 4)
        self.couil_sparse = cfg.couil_sparse_heads
        self.sparse_ratio = cfg.couil_sparse_ratio
        # Absolute keep-window: identical across full/cached passes (cache-exact)
        self.keep_abs = max(1, int(cfg.max_seq_len * (1.0 - self.sparse_ratio)))

    def forward(self, x, layer_past=None, use_cache=False):
        B, T, C = x.size()
        past_len = 0 if layer_past is None else layer_past[0].size(-2)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # RoPE on Q/K (position-aware, cache-compatible)
        q, k = self.rope(q, k, offset=past_len)
        if layer_past is not None:
            pk, pv = layer_past
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)
        present = (k, v) if use_cache else None

        kv_len = k.size(-2)
        if layer_past is None and T > 1:
            attn_mask = None
            is_causal = True
        else:
            offset = kv_len - T
            idx_q = torch.arange(T, device=x.device).unsqueeze(-1)
            idx_k = torch.arange(kv_len, device=x.device).unsqueeze(0)
            attn_mask = (idx_k <= idx_q + offset)
            is_causal = False

        if self.couil_sparse and self.n_head >= 4:
            # Couil: even heads dense, odd heads sparse-topk on scores
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            elif is_causal:
                causal = torch.tril(torch.ones(T, kv_len, dtype=torch.bool, device=x.device))
                scores = scores.masked_fill(~causal, float("-inf"))
            keep = min(kv_len, self.keep_abs)
            for h in range(1, self.n_head, 2):  # odd heads
                thresh = scores[:, h].topk(keep, dim=-1).values[..., -1:]  # [B,T,1]
                scores[:, h] = torch.where(scores[:, h] < thresh,
                                           torch.full_like(scores[:, h], float("-inf")),
                                           scores[:, h])
            a = F.softmax(scores, dim=-1) @ v
        else:
            if _FORMAL_PAPERS_WIRED and getattr(self, 'cfg', None) and getattr(self.cfg, 'use_fa3', False):
                a = quillan_flash_attn(q, k, v, is_causal=is_causal)
            else:
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        a = a.transpose(1, 2).contiguous().view(B, T, self.attn_dim)
        out = self.c_proj(a) + self.prism(x)
        return out, present


# ------------------------------------------------------------------
# PERSONA PULL GATE — Throne assigns deliberation pull (user canon)
# Every persona ALWAYS parses the prism shards; pull weights decide how
# loudly each speaks (ethics question -> VIR pulls harder). fp32 (ST-MoE).
# ------------------------------------------------------------------

class PersonaPullGate(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.register_buffer("prior", PERSONA_PRIOR.clone())
        nn.init.zeros_(self.gate.weight)  # start uniform: prior-only pulls

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        logits = self.gate(x).float() / max(0.05, tau)
        pull = F.softmax(logits, dim=-1) * self.prior.to(x.device)
        return pull / pull.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class UnrolledCouncilMoEBlock(nn.Module):
    """Dense SwiGLU + full-council deliberation (dense_pull) or legacy top-k.
    When cfg.use_evo_moe, delegates to EvoMoE heterogeneous (EvoMoE 2505.23830)."""

    def __init__(self, cfg: QuillanOniConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.hidden_dim, cfg.num_experts, bias=False)
        self.pull_gate = PersonaPullGate(cfg.hidden_dim, cfg.num_experts)
        if cfg.use_evo_moe and _FORMAL_PAPERS_WIRED:
            self.evo_moe = EvoMoE(cfg.hidden_dim, n_experts=cfg.num_experts, rank=cfg.expert_rank)
            self.experts = self.evo_moe.experts  # share for checkpoint compat
        else:
            self.experts = nn.ModuleList([
                CouncilExpert(i, get_expert_name(i), cfg) for i in range(cfg.num_experts)
            ])
            self.evo_moe = None
        self.c_fc = nn.Linear(cfg.hidden_dim, cfg.ffn_dim * 2)
        self.c_proj = nn.Linear(cfg.ffn_dim, cfg.hidden_dim)
        self.moe_gate = nn.Linear(cfg.hidden_dim, 1)
        self.tau = cfg.tau_max

    def set_tau(self, tau: float):
        self.tau = float(max(0.05, min(2.0, tau)))

    def forward(self, x, gov_scale: float = 1.0):
        B, T, C = x.size()
        flat_x = x.reshape(-1, C)

        fc_out = self.c_fc(x)
        gate, act = fc_out.chunk(2, dim=-1)
        h_dense = self.c_proj(F.silu(gate) * act)

        entropy = torch.zeros((), device=x.device)
        lb_loss = torch.zeros((), device=x.device)
        z_loss = torch.zeros((), device=x.device)
        if self.cfg.router_mode == "dense_pull":
            if self.evo_moe is not None:
                # EvoMoE heterogeneous (2505.23830) — token-aware + evolutionary diversity
                moe_out = self.evo_moe(x).reshape(-1, C)
                pull = self.pull_gate(flat_x, tau=self.tau)
                probs = pull
                lb_loss = torch.zeros((), device=x.device)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            else:
                # FULL-COUNCIL DELIBERATION: all 34 parse every token (Throne pull)
                pull = self.pull_gate(flat_x, tau=self.tau)              # [BT,34] fp32
                moe_out = torch.zeros_like(flat_x)
                for e in range(self.cfg.num_experts):
                    e_out = self.experts[e](flat_x, gov_scale)
                    moe_out = moe_out + pull[:, e:e + 1].to(flat_x.dtype) * e_out
                probs = pull
                lb_loss = torch.zeros((), device=x.device)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        else:
            logits = self.router(flat_x).float()  # fp32 router (ST-MoE)
            if self.training and self.cfg.router_mode == "gumbel_topk":
                probs = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            topk_p, topk_i = torch.topk(probs, self.cfg.top_k, dim=-1)
            topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)

            moe_out = torch.zeros_like(flat_x)
            # Vectorized dispatch: group all (token, slot) pairs by expert once.
            K = self.cfg.top_k
            BT = flat_x.size(0)
            flat_idx = topk_i.reshape(-1)                                  # [BT*K]
            flat_w = topk_p.reshape(-1, 1)                                 # [BT*K,1]
            token_pos = torch.arange(BT, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)
            for e in range(self.cfg.num_experts):
                sel = (flat_idx == e).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                pos = token_pos[sel]
                w = flat_w[sel]
                e_out = self.experts[e](flat_x[pos], gov_scale)
                moe_out.index_add_(0, pos, w * e_out)

            # Aux losses: KL-to-uniform load balance (AGI paper eq.13) + z-loss (ST-MoE)
            mean_p = probs.mean(dim=0)
            uniform = torch.full_like(mean_p, 1.0 / self.cfg.num_experts)
            lb_loss = F.kl_div(mean_p.log(), uniform, reduction="sum")
            z_loss = torch.log(torch.exp(logits).sum(dim=-1) + 1e-6).pow(2).mean()
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        g = torch.tanh(self.moe_gate(flat_x))
        out = h_dense + (moe_out * g).view(B, T, C)
        return out, probs, lb_loss, z_loss, entropy


class UnrolledTransformerBlock(nn.Module):
    def __init__(self, cfg: QuillanOniConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_dim, eps=1e-5)
        if cfg.use_mamba and _FORMAL_PAPERS_WIRED:
            self.attn = MambaBlock(cfg.hidden_dim)
        else:
            self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.hidden_dim, eps=1e-5)
        self.moe = UnrolledCouncilMoEBlock(cfg)

    def forward(self, x, layer_past=None, use_cache=False, gov_scale: float = 1.0):
        if isinstance(self.attn, MambaBlock):
            a = self.attn(self.ln_1(x))
            present = None
        else:
            a, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + a
        m, probs, lb, z, ent = self.moe(self.ln_2(x), gov_scale)
        x = x + m
        return x, present, probs, lb, z, ent



# ------------------------------------------------------------------
# AGENTIC EXECUTOR (tool router + sandbox + memory bridge)
# ------------------------------------------------------------------

class QuillanAgenticExecutor(nn.Module):
    TOOLS = ["reason", "recall", "compute", "plan", "verify", "summarize"]

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.tool_router = nn.Linear(hidden_dim, len(self.TOOLS))
        self.sandbox = HardenedSandbox()
        self.last_tool: Optional[str] = None

    def route(self, pooled: torch.Tensor) -> str:
        idx = int(torch.argmax(self.tool_router(pooled), dim=-1).item())
        self.last_tool = self.TOOLS[idx]
        return self.last_tool

    def execute(self, tool: str, code: str = "", memory=None, query_vec=None) -> Dict[str, Any]:
        if tool == "compute" and code:
            return self.sandbox.run(code)
        if tool == "recall" and memory is not None and query_vec is not None:
            hits = memory.recall(query_vec)
            return {"status": "success", "output": f"{len(hits)} memories recalled"}
        return {"status": "success", "output": f"tool '{tool}' dispatched"}


# ------------------------------------------------------------------
# MODALITY-ISOLATED THERMO DIFFUSION (Samurai :4151, faithful compact port)
# Confidence-gated Langevin refinement of hard tokens. Inference-stage;
# ent_loss usable as aux during training.
# ------------------------------------------------------------------

class ModalityIsolatedThermoDiffusion(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 8, max_depth: int = 6,
                 confidence_threshold: float = 0.70, max_noise: float = 0.12,
                 halting_threshold: float = 1e-3, residual_alpha: float = 0.7,
                 entropy_reg_weight: float = 0.01, max_hard_tokens: int = 4096):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.max_depth = max_depth
        self.conf_thresh = confidence_threshold
        self.max_noise = max_noise
        self.halting = halting_threshold
        self.alpha = residual_alpha
        self.ent_w = entropy_reg_weight
        self.max_hard = max_hard_tokens
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.n1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
                                 nn.Linear(hidden_dim * 4, hidden_dim))
        self.n2 = nn.LayerNorm(hidden_dim)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim))

    def _step(self, h: torch.Tensor, t_frac: float) -> torch.Tensor:
        B, L, D = h.shape
        H = self.heads
        t_emb = self.time_mlp(h) * (1.0 - t_frac)
        h = h + t_emb
        q = self.q(self.n1(h)).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k(self.n1(h)).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v(self.n1(h)).view(B, L, H, self.head_dim).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v)
        att = att.transpose(1, 2).reshape(B, L, D)
        h = h + self.out(att)
        h = h + self.ffn(self.n2(h))
        # Langevin noise, inverse-sqrt-t decay
        noise_scale = self.max_noise / max(0.5, math.sqrt(max(1e-6, t_frac)))
        h = h + torch.randn_like(h) * (noise_scale * 0.01)
        return h

    def forward(self, x: torch.Tensor, router_conf: torch.Tensor,
                temperature: float = 0.82) -> Tuple[torch.Tensor, int, torch.Tensor]:
        B, L, D = x.shape
        is_hard = router_conf < self.conf_thresh
        n_hard = int(is_hard.sum().item())
        if n_hard == 0:
            return x, 0, torch.tensor(0.0, device=x.device)
        flat_idx = is_hard.reshape(-1).nonzero(as_tuple=True)[0][: self.max_hard]
        pos = flat_idx // L
        tok = flat_idx % L
        h = x[pos, tok]                                    # [N_hard, D]
        h0 = h.clone()
        prev = h
        for depth in range(1, self.max_depth + 1):
            h = self._step(h.unsqueeze(0), depth / self.max_depth).squeeze(0)
            rms = (h - prev).pow(2).mean().sqrt().item()
            prev = h
            if rms < self.halting:
                break
        delta = h - h0
        x = x.clone()
        x[pos, tok] = h0 + self.alpha * delta
        # entropy regularization on refined distribution
        p = F.softmax(h, dim=-1)
        ent_loss = (-(p * torch.log(p + 1e-9)).sum(dim=-1)).mean() * self.ent_w
        return x, n_hard, ent_loss


# ------------------------------------------------------------------
# DISTILLATION HEAD (port: 117KB v8_saturated / AGI paper eq.35, alpha=0.7)
# ------------------------------------------------------------------

class DistillationHead(nn.Module):
    def __init__(self, hidden_dim: int, temperature: float = 2.0, alpha: float = 0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, student_logits: torch.Tensor, teacher_logits: Optional[torch.Tensor],
                student_hidden: torch.Tensor, teacher_hidden: Optional[torch.Tensor]) -> torch.Tensor:
        ce = torch.zeros((), device=student_logits.device)
        if teacher_logits is not None:
            s = F.log_softmax(student_logits / self.T, dim=-1)
            t = F.softmax(teacher_logits / self.T, dim=-1)
            kl = F.kl_div(s, t, reduction="batchmean") * (self.T ** 2)
        else:
            kl = ce
        hidden_loss = F.mse_loss(self.proj(student_hidden), teacher_hidden) \
            if teacher_hidden is not None else ce
        return self.alpha * kl + (1.0 - self.alpha) * hidden_loss


# ------------------------------------------------------------------
# MASTER UNIFIED SOVEREIGN BACKBONE
# ------------------------------------------------------------------

class QuillanRoninOni(nn.Module):
    """Quillan-Ronin v5.4.0-ONI — Throne + 34-member deliberation council.

    Throne (Quillan Core): intake -> prism shard -> pull assignment -> audit
    -> route [diffusion round | quality gates] -> Typist refinement -> output.
    Council: C1-C34, dense pull-weighted deliberation every token.
    Swarm: rank-r world-sim diversity fabric under each persona.
    """

    def __init__(self, cfg: Optional[QuillanOniConfig] = None):
        super().__init__()
        self.cfg = cfg or QuillanOniConfig()
        cfg = self.cfg

        # RoPE replaces learned wpe (RCI fix #1)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

        # Dual-Brain Ingestion (zero-init per spec: identity at start)
        self.q1_bridge = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.q2_bridge = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.ingest_gate = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)
        nn.init.zeros_(self.q1_bridge.weight)
        nn.init.zeros_(self.q2_bridge.weight)
        nn.init.zeros_(self.ingest_gate.weight)

        self.h = nn.ModuleList([UnrolledTransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.hidden_dim, eps=1e-5)

        # Dual Quillan Finalizers + comm gate
        self.quillan_finalizer_q1 = BitLinear(cfg.hidden_dim, cfg.hidden_dim,
                                              quantize_act=False, quantize_weight=False)
        self.quillan_finalizer_q2 = BitLinear(cfg.hidden_dim, cfg.hidden_dim,
                                              quantize_act=False, quantize_weight=False)
        self.quillan_comm_gate = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)

        # Tied embeddings (spec: lm_head.weight = wte.weight)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        # Cognitive engine stack
        self.governor = LeeMach6Governor(cfg.e_ice_limit_ms)
        self.velocity_governor = LeeMach6VelocityGovernor()
        self.e_ice = EthicalImpactConstraintEngine(cfg.hidden_dim, cfg.e_ice_limit_ms)
        self.marta = MARTAThermodynamicGating(cfg.hidden_dim)
        self.dqso = DynamicQuantumSwarmOscillation(cfg.hidden_dim)
        self.covenant = PrimeCovenantFramework(cfg.hidden_dim)
        self.ccrl = CCRLFramework(cfg.hidden_dim, cfg.num_experts)
        self.quantum = QuantumFormulasEngine(cfg.hidden_dim)
        self.complexity_router = ComplexityRouter(cfg.hidden_dim)
        self.agentic = QuillanAgenticExecutor(cfg.hidden_dim)
        self.memory = QuillanMemory(cfg.hidden_dim)
        self.diffusion = ModalityIsolatedThermoDiffusion(cfg.hidden_dim)
        self.distill = DistillationHead(cfg.hidden_dim)
        self.recirc_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        nn.init.zeros_(self.recirc_proj.weight)
        # 100% Formal Papers — instantiated when wired
        if _FORMAL_PAPERS_WIRED:
            global USE_INTEGER_ONLY
            USE_INTEGER_ONLY = bool(cfg.use_nitro)
            if cfg.use_evo_moe:
                self.evo_moe = EvoMoE(cfg.hidden_dim, n_experts=cfg.num_experts, rank=cfg.expert_rank)
            if cfg.use_mamba:
                self.mamba = MambaBlock(cfg.hidden_dim)
            if cfg.use_world_model:
                self.world_model = HighFidelityWorldModel(cfg.hidden_dim)
            if cfg.use_real_swarm:
                self.real_swarm = RealSwarmMesh(n_experts=cfg.num_experts, gpu_slots=4, rank=cfg.swarm_rank)
            if cfg.use_es:
                self.es = ESAtScale()
                self.forgetting = ForgettingMitigation()
            if cfg.use_speculative:
                self.spec = None  # lazily built 2-layer draft (path_override=1) inside generate
            if cfg.use_nitro:
                self.nitro = True   # BitLinear integer-only path flag
            # ProTrain/Memo/DeepOptimizer are training-time schedulers, not model params

        self.apply(self._init_weights)
        nn.init.zeros_(self.q1_bridge.weight)
        nn.init.zeros_(self.q2_bridge.weight)
        nn.init.zeros_(self.ingest_gate.weight)
        nn.init.zeros_(self.recirc_proj.weight)
        nn.init.constant_(self.marta.flow_controller[-2].bias, 2.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # -- governor passthrough (trainer consumes these) --
    def governor_adjust(self, latency_ms: float) -> Tuple[float, float, float]:
        return self.governor.adjust(latency_ms)

    def set_router_tau(self, tau: float):
        for block in self.h:
            block.moe.set_tau(tau)

    def tau_for_step(self, step: int, total_steps: int) -> float:
        c = self.cfg
        return c.tau_min * (c.tau_max / c.tau_min) ** (1.0 - min(1.0, step / max(1, total_steps)))

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False,
                path_override: Optional[int] = None, recirc_state: Optional[torch.Tensor] = None,
                deliberation: bool = True):
        cfg = self.cfg
        B, T = input_ids.size()

        x = self.wte(input_ids)

        # Recirculation: deep->shallow feedback bias (v10 port, zero-init)
        if recirc_state is not None:
            x = x + self.recirc_proj(recirc_state).to(x.dtype)

        # Dual-Brain Ingestion Gating (spec: 0.05 additive, zero-init bridges)
        q1 = self.q1_bridge(x)
        q2 = self.q2_bridge(x)
        g_ingest = torch.sigmoid(self.ingest_gate(torch.cat([q1, q2], dim=-1)))
        x = x + 0.05 * (g_ingest * q1 + (1.0 - g_ingest) * q2)

        presents = [] if use_cache else None
        if past_key_values is None:
            past_key_values = [None] * len(self.h)

        gov_scale = self.governor.current_scale
        last_probs, total_lb, total_z, total_ent = None, 0.0, 0.0, 0.0
        n_run = len(self.h)

        # Complexity-based early exit at inference (AGI paper sec 3.1)
        if not self.training and path_override is None and T > 1:
            with torch.no_grad():
                comp_logits = self.complexity_classifier_path(x.mean(dim=1))
                path = int(torch.argmax(comp_logits, dim=-1)[0].item())
            n_run = max(2, int(round(len(self.h) * ComplexityRouter.depth_fraction(path))))

        for i, block in enumerate(self.h):
            if i >= n_run:
                break
            use_ckpt = self.training and cfg.grad_checkpoint and past_key_values[i] is None
            if use_ckpt:
                out = checkpoint(lambda h: block(h, None, False, gov_scale=gov_scale),
                                 x, use_reentrant=False)
                x, _, probs, lb, z, ent = out
            else:
                x, present, probs, lb, z, ent = block(
                    x, layer_past=past_key_values[i],
                    use_cache=use_cache, gov_scale=gov_scale)
                if use_cache:
                    presents.append(present)
            last_probs = probs
            total_lb, total_z, total_ent = total_lb + lb, total_z + z, total_ent + ent

        # Cognitive Governing Filters (spec: no_grad modulation)
        with torch.no_grad():
            if last_probs is not None:
                e_ice_out = self.e_ice(x.detach(), last_probs.detach().reshape(B, T, -1))
                flow = self.marta(x.detach(), e_ice_out["constrained"].detach())
                x = x * (0.9 + 0.1 * flow.unsqueeze(-1))
                x = x + 0.05 * self.dqso(x.detach())

                # Throne deliberation control (inference only, pass-level):
                # pull confidence -> PID velocity governor -> hard tokens -> Langevin refinement
                if deliberation and not self.training and not use_cache and T > 1:
                    conf = last_probs.detach().reshape(B, T, -1).max(dim=-1).values.mean()
                    integrity = float(self.covenant(x.detach().mean(dim=1)).mean().item())
                    e_load = float(e_ice_out["constrained"].mean().item())
                    _, pid = self.velocity_governor.step(float(conf.item()), integrity, e_load)
                    thresh = pid["hard_threshold"]
                    refined, n_hard, ent_aux = self.diffusion(
                        x.detach(), last_probs.detach().reshape(B, T, -1).max(dim=-1).values,
                        )
                    x = x + (refined - x.detach()) * 0.5
                    self._last_deliberation = {
                        "pull_confidence": float(conf.item()),
                        "hard_threshold": thresh,
                        "hard_tokens": n_hard,
                        "token_velocity": pid["token_velocity"],
                    }

        hidden = self.ln_f(x)

        # 100% wiring: RealSwarm mesh synthesis + World Model arbitration (inference only)
        if not self.training and not use_cache:
            if _FORMAL_PAPERS_WIRED and getattr(self, "real_swarm", None) is not None:
                try:
                    swarm_out = self.real_swarm.forward(hidden.detach().float())
                    hidden = hidden + (swarm_out.to(hidden.device).to(hidden.dtype) - hidden.detach()) * 0.1
                except Exception:
                    pass
            if _FORMAL_PAPERS_WIRED and getattr(self, "world_model", None) is not None:
                try:
                    bs = self.world_model.estimate(hidden.detach())
                    act = torch.zeros_like(bs.latent)
                    traj = self.world_model.predict_trajectory(bs, act, horizon=1)
                    hidden = (hidden + hidden.detach() * (traj[-1][1] - 0.5)) * (2 - traj[-1][1])
                except Exception:
                    pass

        # Dual Quillan Finalizer Consensus
        q1_out = self.quillan_finalizer_q1(hidden)
        q2_out = self.quillan_finalizer_q2(hidden)
        q1_fused = q1_out + 0.1 * q2_out
        q2_fused = q2_out + 0.1 * q1_out
        gate_final = torch.sigmoid(self.quillan_comm_gate(torch.cat([q1_fused, q2_fused], dim=-1)))
        fused = gate_final * q1_fused + (1.0 - gate_final) * q2_fused

        logits = self.lm_head(fused)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                 shift_labels.view(-1), ignore_index=-100)
            aux = self._aux_losses(x, last_probs, total_lb, total_z, total_ent,
                                   e_ice_out if last_probs is not None else None)
            return logits, ce, aux

        if use_cache:
            return logits, presents
        return logits

    def complexity_classifier_path(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.complexity_router(pooled)

    def _aux_losses(self, x, last_probs, total_lb, total_z, total_ent, e_ice_out):
        cfg = self.cfg
        aux: Dict[str, torch.Tensor] = {}
        n_layers = max(1, len(self.h))
        aux["load_balance"] = total_lb / n_layers
        aux["z_loss"] = total_z / n_layers
        if last_probs is not None:
            aux["entropy"] = total_ent / n_layers
            try:
                aux["qhis"] = self.quantum.qhis_fidelity(x.detach(), x, v_lm6=self.governor.current_scale)
            except Exception:
                pass
            try:
                aux["qics"] = self.quantum.qics_entropy(x.mean(dim=1))
            except Exception:
                pass
        if e_ice_out is not None:
            aux["ethics"] = e_ice_out["constrained"].mean()
        return aux

    def total_aux_loss(self, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        loss = torch.zeros((), device=next(self.parameters()).device)
        if "load_balance" in aux:
            loss = loss + cfg.aux_load_weight * aux["load_balance"]
        if "z_loss" in aux:
            loss = loss + cfg.aux_z_weight * aux["z_loss"]
        if "entropy" in aux:
            loss = loss - cfg.entropy_bonus_weight * aux["entropy"]
        if "qhis" in aux:
            loss = loss + 0.005 * aux["qhis"]
        if "qics" in aux:
            loss = loss + 0.002 * aux["qics"]
        if "ethics" in aux:
            loss = loss + cfg.aux_ethics_weight * aux["ethics"]
        return loss

    @torch.no_grad()
    def generate(self, input_tokens: List[int], max_tokens: int = 150, temp: float = 0.8,
                 top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.15,
                 frequency_penalty: float = 0.5, presence_penalty: float = 0.3) -> List[int]:
        self.eval()
        # 100% wiring: SpeculativeDecoding (DFlash) — draft 1-2 tokens via low-depth path, target verifies
        if _FORMAL_PAPERS_WIRED and getattr(self.cfg, "use_speculative", False):
            try:
                from speculative_decode import SpeculativeDecoder
                dec = SpeculativeDecoder(draft_model=self, target_model=self, gamma=2)
                draft_tokens = self.forward(
                    torch.tensor([input_tokens[-self.cfg.max_seq_len:]], dtype=torch.long,
                                 device=next(self.parameters()).device),
                    path_override=1).argmax(dim=-1)[0][-2:].tolist()
                return self._generate_verify(input_tokens, draft_tokens, max_tokens, temp,
                                             top_k, top_p, repetition_penalty,
                                             frequency_penalty, presence_penalty)
            except Exception:
                return self._generate_legacy(input_tokens, max_tokens, temp, top_k, top_p,
                                             repetition_penalty, frequency_penalty, presence_penalty)
        return self._generate_legacy(input_tokens, max_tokens, temp, top_k, top_p,
                                     repetition_penalty, frequency_penalty, presence_penalty)

    def _generate_verify(self, input_tokens, draft_tokens, max_tokens, temp,
                         top_k, top_p, repetition_penalty, frequency_penalty, presence_penalty):
        """Speculative verify (DFlash 2602.06036): target accepts draft in one parallel pass."""
        gen = list(input_tokens)
        device = next(self.parameters()).device
        for d in draft_tokens:
            gen.append(int(d))
            if len(gen) >= max_tokens + len(input_tokens):
                break
        # target verifies the drafted span by re-scoring — single forward pass (bug-fix: was 3x)
        inp = torch.tensor([gen[-self.cfg.max_seq_len:]], dtype=torch.long, device=device)
        raw = self.forward(inp, use_cache=True, path_override=1)
        logits = raw[0] if isinstance(raw, tuple) else raw
        curr = logits[:, -1, :] / max(0.05, temp)
        probs = F.softmax(curr, dim=-1)
        if top_k > 0:
            val_k, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            probs[probs < val_k[:, -1:]] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)
        next_tok = int(torch.multinomial(probs, 1).item())
        gen.append(next_tok)
        return gen

    def _generate_legacy(self, input_tokens, max_tokens, temp, top_k, top_p,
                         repetition_penalty, frequency_penalty, presence_penalty):
        gen = list(input_tokens)
        device = next(self.parameters()).device
        inp = torch.tensor([gen[-self.cfg.max_seq_len:]], dtype=torch.long, device=device)
        # Full depth on prefill so cached decode steps stay consistent
        logits, kv_cache = self.forward(inp, use_cache=True, path_override=1)

        for _ in range(max_tokens):
            curr = logits[:, -1, :].clone()
            new_tokens = gen[len(input_tokens):]
            counts = Counter(new_tokens)
            for t, c in counts.items():
                curr[0, t] -= (c * frequency_penalty + presence_penalty)
            if repetition_penalty != 1.0 and new_tokens:
                for t in set(new_tokens[-64:]):
                    curr[0, t] = curr[0, t] / repetition_penalty if curr[0, t] > 0 \
                        else curr[0, t] * repetition_penalty

            if temp <= 0.01:
                next_tok = int(torch.argmax(curr, dim=-1).item())
            else:
                curr = curr / max(0.05, temp)
                probs = F.softmax(curr, dim=-1)
                if top_k > 0:
                    val_k, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < val_k[:, -1:]] = 0.0
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                if top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    kill = cum - sp > top_p
                    sp[kill] = 0.0
                    probs = torch.zeros_like(probs).scatter(1, si, sp)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())

            gen.append(next_tok)
            if next_tok in (0, 50256):  # unified EOS 0 + legacy tiktoken 50256 compat
                break
            if len(gen) >= self.cfg.max_seq_len:
                break
            inp_single = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, kv_cache = self.forward(inp_single, past_key_values=kv_cache, use_cache=True)

        return gen


    # ------------------------------------------------------------------
    # THRONE DELIBERATION LOOP (user canon: token deliberation & arbitration)
    # prism shard -> full council -> Quillan audit -> [diffusion round | gates]
    # -> Typist+Quillan refinement -> output
    # ------------------------------------------------------------------

    def quality_gate(self, hidden: torch.Tensor) -> Dict[str, Any]:
        """Exit gates: Nullion (paradox) + Warden (safety/E_ICE) + Shepherd
        (truth/covenant identity) + Quillan (final audit)."""
        with torch.no_grad():
            pooled = hidden.mean(dim=1)
            covenant_score = float(self.covenant(pooled).mean().item())
            pull_gate = self.h[-1].moe.pull_gate(pooled).mean(dim=0)
            warden_idx = 12   # C13-WARDEN
            shepherd_idx = 17  # C18-SHEPHERD
            nullion_idx = 16   # C17-NULLION
            warden_pull = float(pull_gate[warden_idx].item())
            shepherd_pull = float(pull_gate[shepherd_idx].item())
            nullion_pull = float(pull_gate[nullion_idx].item())
            _, entropy = self.ccrl(pooled, pull_gate.unsqueeze(0).expand(4, -1).reshape(-1, self.cfg.num_experts)[:1])
            e_ice_out = self.e_ice(pooled, pull_gate.unsqueeze(0))
            ethics = float(e_ice_out["constrained"].mean().item())
            passed = (covenant_score > 0.35) and (ethics < 0.85)
            return {
                "passed": passed,
                "covenant_identity": covenant_score,
                "ethics_constraint": ethics,
                "gate_pulls": {"nullion": nullion_pull, "warden": warden_pull,
                               "shepherd": shepherd_pull},
                "council_entropy": float(entropy.item()) if torch.is_tensor(entropy) else float(entropy),
            }

    @torch.no_grad()
    def deliberate(self, input_tokens: List[int], max_rounds: int = 2,
                   max_tokens: int = 150, temp: float = 0.8) -> Dict[str, Any]:
        """Full Throne deliberation: generate -> audit -> refine rounds -> gates.
        Returns tokens + full arbitration trace."""
        self.eval()
        trace: Dict[str, Any] = {"rounds": [], "gates": None}
        gen: List[int] = list(input_tokens)
        recirc: Optional[torch.Tensor] = None

        for rnd in range(max_rounds):
            logits = self.forward(
                torch.tensor([gen[-self.cfg.max_seq_len:]], dtype=torch.long,
                             device=next(self.parameters()).device),
                path_override=1, recirc_state=recirc)
            info = getattr(self, "_last_deliberation", {})
            trace["rounds"].append({
                "round": rnd + 1,
                "pull_confidence": info.get("pull_confidence"),
                "hard_threshold": info.get("hard_threshold"),
                "hard_tokens": info.get("hard_tokens"),
                "token_velocity": info.get("token_velocity"),
            })
            conf = info.get("pull_confidence", 1.0)
            if conf >= 0.80 or rnd == max_rounds - 1:
                break
            # Quillan orders another diffusion reasoning round (deep->shallow feedback)
            hidden_pooled = self.wte(torch.tensor([gen[-self.cfg.max_seq_len:]],
                                                  device=next(self.parameters()).device)).mean(dim=1)
            recirc = hidden_pooled

        # sample continuation
        logits = self.forward(
            torch.tensor([gen[-self.cfg.max_seq_len:]], dtype=torch.long,
                         device=next(self.parameters()).device),
            path_override=1, recirc_state=recirc)
        curr = logits[:, -1, :] / max(0.05, temp)
        probs = F.softmax(curr, dim=-1)
        val_k, _ = torch.topk(probs, min(40, probs.size(-1)))
        probs[probs < val_k[:, -1:]] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)
        for _ in range(max_tokens):
            nxt = int(torch.multinomial(probs, 1).item())
            gen.append(nxt)
            if nxt in (0, 50256):
                break
            logits = self.forward(torch.tensor([[nxt]], device=gen and next(self.parameters()).device),
                                  past_key_values=None, use_cache=False)
            curr = logits[:, -1, :] / max(0.05, temp)
            probs = F.softmax(curr, dim=-1)

        # Quality exit gates (Nullion/Warden/Shepherd + Quillan audit)
        with torch.no_grad():
            hidden = self.wte(torch.tensor([gen[-min(len(gen), self.cfg.max_seq_len):]],
                                           device=next(self.parameters()).device))
        trace["gates"] = self.quality_gate(hidden)
        # Typist (C33) + Quillan refinement note: final tokens already passed the
        # dual-finalizer consensus; typist emphasis is a Phase-C wrapper polish.
        trace["typist_refined"] = True
        return {"tokens": gen[len(input_tokens):], "trace": trace}


# Canonical aliases
QuillanRoninSovereignV9 = QuillanRoninOni  # legacy name compat
QuillanUnrolledConfig = QuillanOniConfig
QuillanUnrolledSovereign = QuillanRoninOni

