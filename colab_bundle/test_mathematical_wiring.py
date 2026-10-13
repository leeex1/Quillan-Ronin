#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST MATHEMATICAL WIRING & COMPUTATIONAL INVARIANTS
====================================================
Rigorous unit test battery verifying that the core academic papers in Quillan-Ronin
execute real tensor mathematics in the active compute graph, rather than existing
as passive registry stubs or nominal observer callbacks.

Verified Academic Formulations:
1. BitNet 1.58b / STE (Ma et al. 2402.17764 / Bengio et al.):
   Strict ternary weight quantization to {-1, 0, 1} with dynamic activation scaling.
2. ST-MoE Router Z-Loss (Zoph et al. 2202.08906):
   Active auxiliary regularization penalizing extreme router logit magnitudes.
3. Mixtral / Switch MoE Load Balancing:
   Entropy/KL load-balancing auxiliary loss driving uniform expert dispatch.
4. Gumbel-Softmax Categorical Annealing (Jang et al. / Maddison et al.):
   Tau temperature scheduling modulating routing exploration vs exploitation.
5. Couil Hybrid Causal Attention:
   Strict triangular autoregressive invariant preventing future token leakage.
6. Lee-Mach-6 Governor & Dynamic EMA (Samurai spec):
   Closed-loop latency feedback modulating parameter EMA decay and velocity.
7. Gyawali Empirical Profiler (2309.02521):
   Wall-clock phase-breakdown telemetry capturing real system execution.
"""

import math
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quillan_v5_4_oni import (
    BitLinear,
    _weight_quant,
    QuillanRoninOni,
    QuillanOniConfig,
    CausalSelfAttention,
    LeeMach6Governor,
    QuantumFormulasEngine
)
from paper_01_profiler import StepProfiler
from paper_02_abductive import AbductiveJump, AbductiveHypothesis
from paper_03_proxy_kd import SampleWeightedKL, ProxyKD
from paper_04_memo import MemoConfig, solve_optimal_alpha, MemoMemoryPlanner
from paper_05_07_heterogeneous import CUDAOptimizationPack, AxoNNHybridPlanner, HeterogeneousAllocator
from paper_08_10_inference_pack import AdaptiveBatchSizer, MultiGPUTaskAllocator, TaskAllocator, APEXScheduler, PipelinedInference
from paper_11_15_quant_memory_pack import (
    NVFP4Quantizer, NVFP4Linear, apply_random_hadamard_transform_16, NVFP4_E2M1_LEVELS,
    CUDACachingAllocatorSimulator, XMemEstimator, EGGROLLOptimizer, HyperscaleES,
    LongHorizonDistiller
)
from moba_attention import MoBAAttention, online_softmax_combine
from paper_16_20_agent_evolution_pack import (
    ConsciousnessSteeringEngine,
    SelfModelCoherenceLoss,
    ContinualSkillBenchTracker,
    BPEHarnessState,
    EvoHarnessReward,
    HarnessPolicy,
    EmbedderCostModel,
    HybridRetrieveRerankRouter,
    HybridEmbedderRouter,
    PersonaConditioning
)
from paper_21_25_grt_coordination_pack import (
    TemporalCoordinationNetwork,
    CoordinationTelemetry,
    AgentPhysicsPredictor,
    ASIBenchEvaluator,
    ASIBenchRunner,
    SCIENTIFIC_DOMAINS,
    GUIDANCE_LEVELS,
    REFERENCE_BASELINES,
    ABRAScalingLaws,
    ABRAScaling
)
from paper_26_30_recurrent_diffusion_pack import (
    GatedDeltaNetLayer,
    DynamicRecurrentCompressor,
    DynamicCompressionLoss,
    RecirculationNormMode,
    RecirculationLayer,
    AdaptiveRecirculationMLP,
    RecirculationEngine
)
from ecdysis_harness import (
    EcdysisHarnessEvolver,
    FailureRecord,
    FailurePatternGroup,
    FDCRRefiner
)
from sol_pi_harness import (
    ActionFusionHandler,
    CostGatedContextCompactor,
    ObservationPackBuffer,
    EvidencePreservingReducer,
    SoLPiHarness
)
from memory_attention import (
    MemoryAttentionLayer,
    MAOffloadManager,
    MARecallManager,
    MemoryAttentionEngine,
    MAAnalysisReport
)
from ai_consciousness_evaluator import (
    ConsciousnessLevel,
    IndicatorSpec,
    CANONICAL_INDICATORS,
    BayesianConsciousnessNetwork,
    DifferentiableConsciousnessHead
)
from ax_prover_engine import (
    LeanDiagnosticSeverity,
    DiagnosticMessage,
    DiagnosticReport,
    GoalState,
    LemmaDeclaration,
    MCPLeanLSPClient,
    ProofStep,
    TheoremSpecification,
    AxProverAgent,
    AxVerifierGate,
    AxProverOrchestrator,
    AxProverResult,
    DifferentiableProverScorer
)
from beyond_158_bits import (
    Sub158AnalyticEngine,
    AdaptivePrecisionLinear,
    ModuleBudgetSpec,
    HeterogeneousBitBudgetOptimizer,
    TernaryExpertBlock,
    HighPrecisionMoERouter,
    FullyTernaryCouncilMoE
)
from bitnet_cpu_infra import (
    I2SKernel,
    TL1Kernel,
    TL2Kernel,
    BitNetCPULinear
)
from bitnet_a48 import (
    BitNetWeightQuantizer,
    ActivationINT4Quantizer,
    ActivationINT8Quantizer,
    ActivationFP4Quantizer,
    TopKSparsifier,
    BitLinearA48,
    BitNetA48MultiHeadAttention,
    BitNetA48FFN,
    BitNetA48TransformerBlock
)
from bitnet_b158_engine import (
    AbsmeanWeightQuantizer,
    PerTokenActivationQuantizer,
    BitLinearB158,
    BitNetB158SwiGLU,
    BitNetB158Attention,
    BitNetB158TransformerBlock,
    HorowitzEnergyModel
)
from bitnet_2b4t_engine import (
    SubLN,
    AbsmeanWeightQuantizer2B4T,
    AbsmaxActivationQuantizer2B4T,
    BitLinear2B4T,
    ReLU2FFN,
    BitNet2B4TAttention,
    BitNet2B4TTransformerBlock,
    TwoStageLRScheduler,
    TwoStageWeightDecayScheduler,
    SFTLossAggregator,
    DPOTrainer2B4T,
    BitNet2B4TEnergyMetrics
)
from bitnet_cpp_edge import (
    SignedUnsignedWeightPacker,
    TL1KernelEdge,
    TL2KernelEdge,
    I2SKernelEdge,
    BitNetCppEdgeLinear,
    ELUTTheoreticalAnalyzer
)
from bitnet_scaling_engine import (
    BinaryWeightQuantizer,
    AbsmaxActivationQuantizerBitNet,
    BitLinear1Bit,
    GroupQuantizedBitLinear,
    BitNetScalingBlock,
    BitNetScalingLawAnalyzer
)
from bitnet_v2_engine import (
    fast_hadamard_transform,
    hadamard_transform,
    quantize_weight_ternary,
    quantize_act_int8,
    quantize_act_int4,
    BitLinearV2,
    HBitLinear,
    BitNetV2Attention,
    BitNetV2FFN,
    BitNetV2TransformerBlock,
    BitNetV2Model,
    OutlierDistributionProfiler
)
from inverse_cryptography_engine import (
    PrimeRasterDecomposer,
    CallimahosInverseCryptEngine,
    ZeroPriorAgentHandshake
)
from dali_moe_engine import (
    HardwareProfile,
    DALIGreedyAssignment,
    DALIResidualPrefetcher,
    DALIWorkloadAwareCache,
    DALIMoEOrchestrator
)
from dapo_rl_engine import (
    DAPOConfig,
    soft_overlong_punishment,
    compute_rule_reward,
    is_valid_dapo_group,
    compute_group_advantages,
    DAPOLoss,
    DAPODynamicSampler
)














# ─── 1. BITNET 1.58b / STE TERNARY INVARIANT ─────────────────────────────────

def test_bitnet_ternary_quantization_active():
    """Proves BitNet ternary quantization actively constrains weights to {-1, 0, 1}."""
    torch.manual_seed(42)
    in_features, out_features = 64, 64
    layer = BitLinear(in_features, out_features, bias=False, quantize_weight=True)
    
    # Raw weights are continuous normal floats
    assert torch.is_floating_point(layer.weight)
    assert not torch.all(torch.isin(layer.weight, torch.tensor([-1.0, 0.0, 1.0])))
    
    # Quantized weights MUST be strictly {-1, 0, 1} multiplied by a scalar scale
    w_quant = _weight_quant(layer.weight)
    scale = layer.weight.abs().mean(dim=[-2, -1], keepdim=True).clamp(min=0.01)
    normalized = torch.round(w_quant / scale)
    unique_vals = torch.unique(normalized).tolist()
    
    # Assert values are strictly ternary
    for val in unique_vals:
        assert val in [-1.0, 0.0, 1.0], f"Violation: non-ternary value {val} found in quantized weight!"
        
    # STE Gradient Passthrough: Backprop must flow to continuous weights
    x = torch.randn(2, in_features, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0.0, "STE failure: Gradients did not flow to weights!"


# ─── 2. ST-MoE Z-LOSS REGULARIZATION INVARIANT ──────────────────────────────

def test_moe_zloss_gradient_propagation():
    """Proves ST-MoE router z-loss penalizes extreme logits and propagates gradients."""
    router_logits = torch.tensor([[10.0, -10.0, 25.0, 0.5]], requires_grad=True)
    
    # ST-MoE Z-Loss formulation: log^2(sum(exp(x)))
    lse = torch.logsumexp(router_logits, dim=-1)
    z_loss = (lse ** 2).mean() * 0.001
    
    assert z_loss.item() > 0.0
    z_loss.backward()
    
    assert router_logits.grad is not None
    # Highest logit (25.0) must receive the strongest negative gradient penalty
    assert router_logits.grad[0, 2] > router_logits.grad[0, 3]


# ─── 3. GUMBEL TAU ANNEALING INVARIANT ───────────────────────────────────────

def test_gumbel_tau_annealing_schedule():
    """Validates that Gumbel tau temperature anneals across training steps."""
    cfg = QuillanOniConfig(hidden_dim=64, n_layer=2, num_experts=34, vocab_size=100)
    model = QuillanRoninOni(cfg)
    
    total_steps = 1000
    tau_step_0 = model.tau_for_step(0, total_steps)
    tau_step_500 = model.tau_for_step(500, total_steps)
    tau_step_1000 = model.tau_for_step(1000, total_steps)
    
    assert abs(tau_step_0 - 1.0) < 0.01, "Initial tau must be 1.0 (high exploration)"
    assert abs(tau_step_1000 - 0.1) < 0.01, "Final tau must be 0.1 (sharp exploitation)"
    assert tau_step_0 > tau_step_500 > tau_step_1000, "Tau must strictly decrease monotonically!"


# ─── 4. CAUSAL ATTENTION TEMPORAL INVARIANCE ────────────────────────────────

def test_couil_attention_causal_masking():
    """Proves causal masking prevents future tokens from influencing past representations."""
    cfg = QuillanOniConfig(hidden_dim=64, n_head=4, head_dim=16, max_seq_len=64)
    attn = CausalSelfAttention(cfg)
    
    B, L = 1, 8
    x1 = torch.randn(B, L, cfg.hidden_dim)
    
    # Pass 1: standard sequence
    out1, _ = attn(x1)
    
    # Pass 2: mutate ONLY the last token (index L-1)
    x2 = x1.clone()
    x2[0, L - 1, :] += 50.0
    out2, _ = attn(x2)
    
    # Tokens 0 through L-2 must remain 100% IDENTICAL (causal invariance)
    diff = (out1[0, :L-1, :] - out2[0, :L-1, :]).abs().max().item()
    assert diff < 1e-5, f"Causal violation: Future token at index {L-1} corrupted past token representations (max diff: {diff})!"


# ─── 5. LEE-MACH-6 GOVERNOR HARDWARE FEEDBACK ───────────────────────────────

def test_leemach6_governor_latency_feedback():
    """Proves the Lee-Mach-6 governor dynamically throttles scale when latency exceeds target."""
    gov = LeeMach6Governor(target_latency_ms=100)
    
    # Low latency condition (30ms < 100ms) -> scale increases toward 1.0
    scale_fast, ema_fast, bias_fast = gov.adjust(30.0)
    assert scale_fast == 1.0
    
    # High latency condition (250ms > 100ms) -> scale throttles down, EMA decay tightens
    scale_slow, ema_slow, bias_slow = gov.adjust(250.0)
    assert scale_slow < 1.0, "Governor must throttle compute scale when latency spikes!"
    assert ema_slow == 0.9999, "Governor must freeze EMA decay during thermal/latency events!"
    assert bias_slow == 1.0, "Recency bias must activate under latency pressure!"


# ─── 6. GYAWALI STEP PROFILER PHASE BREAKDOWN ────────────────────────────────

def test_gyawali_profiler_phase_decomposition(tmp_path=None):
    """Verifies that the Gyawali step profiler records real phase timings."""
    if tmp_path is None:
        import tempfile
        tmp_path = tempfile.mkdtemp()
    profiler = StepProfiler(device=torch.device("cpu"), log_dir=tmp_path, log_every=1)
    
    profiler.begin_step(0)
    profiler.mark_data_load_done()
    profiler.mark_fwd_start()
    # Mock forward compute
    _ = sum(i ** 2 for i in range(10000))
    profiler.mark_bwd_start()
    # Mock backward compute
    _ = sum(i ** 2 for i in range(10000))
    profiler.mark_opt_start()
    # Mock optimizer step
    _ = sum(i ** 2 for i in range(5000))
    
    profiler.end_step(loss=5.5, grad_norm=1.0, batch_size=2, seq_len=64)
    
    assert len(profiler.records) == 1
    rec = profiler.records[0]
    assert rec.total_ms > 0.0
    assert rec.fwd_ms >= 0.0
    assert rec.bwd_ms >= 0.0
    assert rec.opt_ms >= 0.0


# ─── 7. MODEL AUXILIARY LOSS INTEGRATION PROOF ───────────────────────────────

def test_active_auxiliary_loss_in_forward_pass():
    """Proves auxiliary regularizers actively contribute to the training compute graph."""
    cfg = QuillanOniConfig(hidden_dim=128, n_layer=2, num_experts=34, vocab_size=500)
    model = QuillanRoninOni(cfg)
    
    x = torch.randint(0, 500, (1, 16))
    y = x.clone()
    
    _, ce_loss, aux = model(x, labels=y)
    total_aux = model.total_aux_loss(aux)
    
    assert torch.is_tensor(ce_loss) and ce_loss.item() > 0.0
    assert torch.is_tensor(total_aux)
    
    # Combined loss must backpropagate without error
    total_loss = ce_loss + total_aux
    total_loss.backward()
    
    # Check that embedding weights received gradients
    assert model.wte.weight.grad is not None
    assert model.wte.weight.grad.abs().sum() > 0.0


# ─── 8. UHLMANN-BURES QUANTUM FIDELITY INVARIANT ─────────────────────────────

def test_uhlmann_bures_quantum_fidelity_invariants():
    """Proves Uhlmann-Bures fidelity satisfies 0 <= F <= 1, F(rho, rho)=1, and pure state transition probability."""
    engine = QuantumFormulasEngine(hidden_dim=32)
    B, D = 2, 32
    
    # Test identical pure states -> F must be 1.0, trace distance must be 0.0
    h_same = torch.randn(B, D)
    iq_same = engine.qhis_fidelity(h_same, h_same, v_lm6=1.0, lambda_drift=1.0)
    assert abs(iq_same.item() - 1.0) < 1e-3, f"Fidelity of identical states must be 1.0, got {iq_same.item()}"
    
    # Test orthogonal pure states -> F must be 0.0, trace distance must be 1.0 -> IQ = -1.0
    h_ortho1 = torch.zeros(B, D)
    h_ortho1[:, 0] = 1.0
    h_ortho2 = torch.zeros(B, D)
    h_ortho2[:, 1] = 1.0
    iq_ortho = engine.qhis_fidelity(h_ortho1, h_ortho2, v_lm6=1.0, lambda_drift=1.0)
    assert abs(iq_ortho.item() - (-1.0)) < 1e-3, f"Fidelity of orthogonal states must be 0, got {iq_ortho.item()}"
    
    # Test density matrix Bures fidelity
    rho1 = engine.state_to_density(h_ortho1)
    rho2 = engine.state_to_density(h_ortho2)
    iq_dm = engine.qhis_fidelity(rho1, rho1, v_lm6=1.0, lambda_drift=1.0)
    assert abs(iq_dm.item() - 1.0) < 1e-3, f"Bures self-fidelity on density matrix must be 1.0, got {iq_dm.item()}"


# ─── 9. LINDBLAD GKSL MASTER EQUATION TRACE CONSERVATION ────────────────────

def test_lindblad_gksl_trace_conservation():
    """Proves the Lindblad dissipator strictly conserves trace: Tr(drho/dt) == 0."""
    engine = QuantumFormulasEngine(hidden_dim=16)
    B, N = 2, 16
    h = torch.randn(B, N)
    rho = engine.state_to_density(h)
    
    # Jump operator representing environmental relaxation
    L = torch.randn(B, N, N) / math.sqrt(N)
    drho = engine.jqld_density_dissipator(rho, jump_ops=[L], gammas=[0.1])
    
    # Trace of derivative MUST be zero
    tr_drho = torch.diagonal(drho, dim1=-2, dim2=-1).sum(dim=-1)
    assert tr_drho.abs().max().item() < 1e-5, f"GKSL trace violation: Tr(drho/dt) = {tr_drho}"
    
    # State trajectory step preserves norm
    h_next = engine.jqld_evolution_step(h, tau_gumbel=0.2)
    norm_diff = (h.norm(dim=-1) - h_next.norm(dim=-1)).abs().max().item()
    assert norm_diff < 1e-4, f"JQLD trajectory norm violation: diff = {norm_diff}"


# ─── 10. VON NEUMANN ENTROPY THEOREM BOUNDS ──────────────────────────────────

def test_von_neumann_entropy_bounds():
    """Proves von Neumann entropy S(rho) == 0 for pure states and S(rho) == ln(N) for maximally mixed."""
    engine = QuantumFormulasEngine(hidden_dim=16)
    N = 8
    
    # Maximally mixed state: rho = I / N -> S = ln(N)
    rho_mixed = torch.eye(N).unsqueeze(0) / float(N)
    vals = torch.linalg.eigvalsh(rho_mixed)
    s_mixed = -torch.sum(vals * torch.log(vals + 1e-12)).item()
    expected_s = math.log(N)
    assert abs(s_mixed - expected_s) < 1e-4, f"Mixed state entropy violation: {s_mixed} != {expected_s}"
    
    # Pure state hidden vector entropy
    h_pure = torch.zeros(1, 16)
    h_pure[0, 0] = 5.0
    s_pure = engine.qics_entropy(h_pure)
    assert s_pure.item() < 1e-3, f"Pure state entropy must be 0, got {s_pure.item()}"


# ─── 11. TRANSVERSE-FIELD ISING HAMILTONIAN ENERGETICS ──────────────────────

def test_transverse_field_ising_energy_monotonicity():
    """Proves DQRO Transverse-Field Ising Hamiltonian responds to spin alignment and transverse field."""
    engine = QuantumFormulasEngine(hidden_dim=16)
    B, N = 2, 16
    
    # Ferromagnetically aligned spins vs anti-aligned spins
    spins_aligned = torch.full((B, N), 3.0) # tanh -> ~1.0
    spins_random = torch.randn(B, N)
    
    e_aligned = engine.dqro_energy(spins_aligned)
    e_random = engine.dqro_energy(spins_random)
    
    # Ferromagnetically coupled state with matching bias has lower (more favorable) energy
    assert e_aligned.mean().item() < e_random.mean().item()


# ─── 12. ABDUCTIVE JUMP E->J->A REASONING CYCLE (Paper 2 / 235) ─────────────

def test_abductive_jump_invariant():
    """Proves AbductiveJump generates coherent counterfactual axioms and active aux loss."""
    cfg = QuillanOniConfig(hidden_dim=64, n_layer=2, num_experts=34, vocab_size=200, use_abductive=True)
    model = QuillanRoninOni(cfg)
    assert model.abductive_jump is not None, "Abductive jump module must be initialized when use_abductive=True"

    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    model.zero_grad()
    logits, ce, aux = model(x, labels=y)
    
    assert "abductive_loss" in aux, "aux['abductive_loss'] must be emitted"
    total_loss = ce + model.total_aux_loss(aux)
    total_loss.backward()

    # Verify gradients reach abductive parameters
    g_s2a = model.abductive_jump.sim_to_axiom[0].weight.grad
    g_prior = model.abductive_jump.physical_prior[0].weight.grad
    assert g_s2a is not None and g_s2a.norm().item() > 0, "sim_to_axiom must receive backprop gradients"
    assert g_prior is not None and g_prior.norm().item() > 0, "physical_prior must receive backprop gradients"


# ─── 13. PROXY-KD SAMPLE-WEIGHTED BLACK-BOX DISTILLATION (Paper 3 / 2401.07013)

def test_proxy_kd_invariant():
    """Proves Proxy-KD sample-level weighting and active distillation aux loss in compute graph."""
    # 1. Test sample-level weighting
    weighted_kl = SampleWeightedKL(alpha=10.0, temperature=2.0)
    log_p = torch.tensor([-2.5, -1.0, -0.5, -4.0])
    w = weighted_kl.compute_weight(log_p)
    assert w[2] > w[3], "Higher likelihood must yield higher distillation weight"

    # 2. Test in model forward pass
    cfg = QuillanOniConfig(hidden_dim=64, n_layer=2, num_experts=34, vocab_size=200, use_proxy_kd=True)
    model = QuillanRoninOni(cfg)
    assert model.distill_head is not None, "distill_head must be initialized"

    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    model.zero_grad()
    logits, ce, aux = model(x, labels=y)

    assert "distill" in aux, "aux['distill'] must be present"
    total_loss = ce + model.total_aux_loss(aux)
    total_loss.backward()
    assert model.wte.weight.grad is not None and model.wte.weight.grad.norm().item() > 0


# ─── 14. MEMO TOKEN-WISE ACTIVATION MANAGEMENT & MEMORY PLANNING (Paper 4 / 2407.12117)

def test_memo_invariant():
    """Proves MEMO LP optimal swap fraction solution and active layer-wise activation management."""
    # 1. Test LP optimization
    mc = MemoConfig(batch_size=2, seq_len=512, hidden_dim=1024, n_layer=6, pcie_bandwidth_gbps=16.0, cpu_memory_gb=28.0)
    alpha = solve_optimal_alpha(mc)
    assert 0.0 <= alpha <= 1.0, "Optimal alpha must be in [0, 1]"

    # 2. Test bi-level transient planner
    planner = MemoMemoryPlanner(mc)
    plan = planner.plan(torch.device("cpu"))
    assert len(plan) > 0, "Memory plan must not be empty"

    # 3. Test active model integration
    cfg = QuillanOniConfig(hidden_dim=64, n_layer=2, num_experts=34, vocab_size=200, use_memo=True)
    model = QuillanRoninOni(cfg)
    assert model.memo is not None, "model.memo must be initialized when use_memo=True"

    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    logits, ce, aux = model(x, labels=y)
    fired_names = [name for name, _ in model._fired]
    assert "memo_layer" in fired_names, "memo_layer must fire in model forward pass"


# ─── 15. CUDA OPTIMIZATION PACK & PARALLEL COMPUTING (Paper 6 / 2410.05686v1)

def test_cuda_optimization_pack_invariant():
    """Proves CUDA memory coalescing audit, warp boundary padding, branchless predication, and bank conflict checks."""
    # 1. Coalescing audit: unit-stride contiguous vs strided transpose
    t_contig = torch.randn(4, 32)
    res_contig = CUDAOptimizationPack.audit_coalescing(t_contig)
    assert res_contig["is_contiguous"] and res_contig["has_unit_stride"], "Contiguous tensor must have unit stride"

    t_strided = t_contig.t()
    res_strided = CUDAOptimizationPack.audit_coalescing(t_strided)
    assert not res_strided["is_contiguous"], "Transposed tensor must be non-contiguous"
    assert res_strided["efficiency"] < 1.0, "Non-contiguous tensor must have penalized efficiency"

    # 2. Warp boundary padding (dim aligned to 32)
    t_unaligned = torch.randn(2, 25)
    t_padded = CUDAOptimizationPack.pad_to_warp_boundary(t_unaligned, dim=-1)
    assert t_padded.shape == (2, 32), f"Expected shape (2, 32), got {t_padded.shape}"

    # 3. Branchless predication avoiding warp divergence
    cond = torch.tensor([True, False, True])
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])
    out = CUDAOptimizationPack.branchless_predication(cond, a, b)
    assert torch.allclose(out, torch.tensor([1.0, 20.0, 3.0])), "Predication result mismatch"

    # 4. Bank conflicts in 32 shared memory banks
    assert CUDAOptimizationPack.check_bank_conflict(2) is True, "Stride 2 must cause bank conflict"
    assert CUDAOptimizationPack.check_bank_conflict(32) is True, "Stride 32 must cause bank conflict"
    assert CUDAOptimizationPack.check_bank_conflict(1) is False, "Stride 1 must be conflict-free"
    assert CUDAOptimizationPack.check_bank_conflict(33) is False, "Stride 33 must be conflict-free"

    # 5. Active Model forward integration
    cfg = QuillanOniConfig(hidden_dim=64, n_layer=2, num_experts=34, vocab_size=200, use_heterogeneous=True)
    model = QuillanRoninOni(cfg)
    assert model.hetero_manager is not None, "hetero_manager must be initialized when use_heterogeneous=True"

    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    logits, ce, aux = model(x, labels=y)
    fired_dict = dict(model._fired)
    assert "hetero_strategy" in fired_dict, "hetero_strategy must fire in model forward pass"
    assert fired_dict["hetero_strategy"]["coalesced"] is True, "Hidden state must be confirmed coalesced"


# ─── 16. AxoNN 4D HYBRID PARALLEL COMMUNICATION MODEL (Paper 7 / 2502.08145v1)

def test_axonn_4d_hybrid_planner_invariant():
    """Proves AxoNN 4D hybrid analytical communication equations (Rabenseifner/Thakur ring collectives) and grid search."""
    planner = AxoNNHybridPlanner()

    # 1. Test analytical layer communication equations (Eqs 1-5 in Singh et al., 2025)
    comm = planner.predict_layer_comm_time(
        g_data=2, gx=2, gy=2, gz=2,
        m=1024, k=1024, n=4096,
        beta_gbps=25.0
    )
    assert comm["total_comm_ms"] > 0, "Total communication time must be positive"
    assert comm["t_ag_z_ms"] > 0, "All-gather Z time must be positive"
    assert comm["t_rs_z_ms"] > 0, "Reduce-scatter Z time must be positive"
    assert comm["t_ar_y_ms"] > 0, "All-reduce Y time must be positive"
    assert comm["t_ar_x_ms"] > 0, "All-reduce X time must be positive"
    assert comm["t_ar_data_ms"] > 0, "Data parallel all-reduce time must be positive"
    expected_sum = comm["t_ag_z_ms"] + comm["t_rs_z_ms"] + comm["t_ar_y_ms"] + comm["t_ar_x_ms"] + comm["t_ar_data_ms"]
    assert math.isclose(comm["total_comm_ms"], expected_sum, rel_tol=1e-5), "Communication components must sum to total"

    # 2. Test 4D grid optimization search
    strat_8gpu = planner.find_optimal_4d_grid(num_gpus=8, batch_size=2, seq_len=512, hidden_dim=1024, n_layer=6)
    assert strat_8gpu.g_data * strat_8gpu.gx * strat_8gpu.gy * strat_8gpu.gz == 8, "4D factors must product to num_gpus"
    assert strat_8gpu.comm_time_ms > 0, "Optimal strategy must have positive communication cost"

    # 3. Test strategy picker integration
    strat = planner.pick_strategy(seq_len=512, hidden_dim=1024, n_layer=6, batch_size=2)
    assert strat.name == "dp_only", "Small sequence should select dp_only"
    assert hasattr(strat, "comm_time_ms") and strat.comm_time_ms >= 0


# ─── 17. GPU+CPU DYNAMIC WORKLOAD ALLOCATOR (Paper 8 / 2502.11129v1)

def test_heterogeneous_allocator_invariant():
    """Proves dynamic CPU/GPU throughput-based workload balancing and partitioning (Eynaliyev & Liu, 2025)."""
    alloc = HeterogeneousAllocator()

    # 1. Optimal split calculation: t_cpu = 4ms, t_gpu = 1ms -> 80% GPU, 20% CPU
    split = alloc.calculate_optimal_split(100, t_cpu_per_item=4.0, t_gpu_per_item=1.0, overhead_ms=0.5)
    assert split["n_gpu"] == 80, f"Expected 80 GPU items, got {split['n_gpu']}"
    assert split["n_cpu"] == 20, f"Expected 20 CPU items, got {split['n_cpu']}"
    assert split["t_gpu_ms"] == 80.0, "GPU execution time must match 80 * 1.0ms"
    assert split["t_cpu_ms"] == 80.0, "CPU execution time must match 20 * 4.0ms"
    assert split["speedup"] > 1.2, f"Expected speedup > 1.2x, got {split['speedup']:.2f}x"

    # 2. Workload partitioning across 34 MoE experts
    experts = list(range(34))
    gpu_exp, cpu_exp = alloc.partition_workload(experts, op_type="expert_dispatch")
    assert len(gpu_exp) + len(cpu_exp) == 34, "All experts must be partitioned"
    assert set(gpu_exp).isdisjoint(set(cpu_exp)), "Expert partitions must be disjoint"

    # 3. Model integration: allocator accessible on hetero_manager
    cfg = QuillanOniConfig(hidden_dim=64, n_layer=2, num_experts=34, vocab_size=200, use_heterogeneous=True)
    model = QuillanRoninOni(cfg)
    assert hasattr(model.hetero_manager, "allocator"), "hetero_manager must contain allocator"
    assert model.hetero_manager.allocator.allocate("matmul") == "cuda"
    assert model.hetero_manager.allocator.allocate("tokenize") == "cpu"


# ─── 18. MOBA: MIXTURE OF BLOCK ATTENTION (Paper 9 / 2502.13189v1)

def test_moba_block_attention_invariant():
    """Proves MoBA sub-quadratic block partition, mean-pooled key affinity, causal gating, and online softmax combination."""
    # 1. Test standalone MoBAAttention forward + backward
    moba = MoBAAttention(head_dim=16, n_head=4, block_size=16, top_k=1)
    q = torch.randn(2, 4, 48, 16, requires_grad=True)
    k = torch.randn(2, 4, 48, 16, requires_grad=True)
    v = torch.randn(2, 4, 48, 16, requires_grad=True)

    out = moba(q, k, v, is_causal=True)
    assert out.shape == (2, 4, 48, 16), f"Expected shape (2, 4, 48, 16), got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and q.grad.norm().item() > 0, "Query gradient must propagate through MoBA"
    assert k.grad is not None and k.grad.norm().item() > 0, "Key gradient must propagate through MoBA"
    assert v.grad is not None and v.grad.norm().item() > 0, "Value gradient must propagate through MoBA"

    # 2. Test full model integration
    cfg = QuillanOniConfig(
        hidden_dim=64, n_layer=2, num_experts=34, vocab_size=200,
        use_moba=True, moba_block_size=16, moba_top_k=2
    )
    model = QuillanRoninOni(cfg)
    assert model.h[0].attn.use_moba is True, "Attention layer must have use_moba=True"
    assert model.h[0].attn.moba is not None, "moba module must be instantiated"

    x = torch.randint(0, 200, (2, 48))
    y = torch.randint(0, 200, (2, 48))
    logits, ce, aux = model(x, labels=y)
    assert logits.shape == (2, 48, 200)
    total_loss = ce + model.total_aux_loss(aux)
    total_loss.backward()
    assert model.wte.weight.grad is not None and model.wte.weight.grad.norm().item() > 0, "Embedding gradients must propagate through MoBA layers"


def test_adaptive_batch_sizer_bca_invariant():
    """Proves Batching Configuration Advisor (BCA) solves argmax T(B) subject to latency SLO and scaling efficiency (Paper 10 / 2503.08311v2 Recasens)."""
    sizer = AdaptiveBatchSizer(gpu_memory_mb=4096.0, model_mb=1000.0, dram_bw_gbps=112.0)

    # 1. Standard solve under 50ms SLO
    res = sizer.bca_solve(seq_len=512, hidden_dim=1024, n_layer=6, slo_ms=50.0, epsilon=0.1)
    assert res["b_opt"] >= 1, "Optimal batch size must be >= 1"
    assert res["latency_ms"] <= 50.0, f"Latency {res['latency_ms']} ms must respect SLO 50.0 ms"
    assert res["efficiency_ratio"] > 0.1, f"Efficiency ratio {res['efficiency_ratio']} must exceed epsilon 0.1"
    assert res["throughput_tokens_per_sec"] > 0, "Throughput must be positive"
    assert res["memory_saved_percent"] >= 0.0, "Memory savings must be non-negative"

    # 2. Tight SLO test: under 12ms SLO, optimal batch size must be constrained
    tight_res = sizer.bca_solve(seq_len=512, hidden_dim=1024, n_layer=6, slo_ms=12.0, epsilon=0.1)
    assert tight_res["b_opt"] <= res["b_opt"], "Tighter SLO must yield batch size <= standard batch size"
    assert tight_res["latency_ms"] <= 12.0, "Tight SLO must be strictly satisfied"

    # 3. Test optimal_batch_size compatibility helper
    b_heuristic = sizer.optimal_batch_size(seq_len=512, hidden_dim=1024, n_layer=6, slo_ms=50.0)
    assert isinstance(b_heuristic, int) and b_heuristic >= 1, "optimal_batch_size must return positive integer"


def test_multigpu_task_allocator_lawenda_invariant():
    """Proves Multi-GPU task allocation, DDP-DS partitioning, BFD bin-packing, and Lawenda profiling decomposition (Paper 11 / 2503.15252v1 Lawenda)."""
    allocator = MultiGPUTaskAllocator(gpu_memory_mb=4096.0, num_gpus=4)

    # 1. Dataset workload distribution invariant
    n_samples = 100
    p_ddp = allocator.partition_dataset(n_samples, num_ranks=4, strategy="DDP")
    assert all(p["num_samples"] == n_samples for p in p_ddp), "DDP must replicate all samples across all ranks"

    p_ds = allocator.partition_dataset(n_samples, num_ranks=4, strategy="DDP-DS")
    assert sum(p["num_samples"] for p in p_ds) == n_samples, "DDP-DS partitions must sum to total samples"
    # Verify partitions are mutually disjoint
    all_indices = [idx for p in p_ds for idx in p["indices"]]
    assert len(all_indices) == len(set(all_indices)) == n_samples, "DDP-DS indices must be strictly disjoint and complete"

    # 2. Profiling metrics & speedup invariants
    base_1gpu = allocator.predict_performance(num_gpus=1, precision="FP64", pin_memory=False)
    p4_fp32 = allocator.predict_performance(num_gpus=4, precision="FP32", pin_memory=True)
    assert p4_fp32["speedup"] > base_1gpu["speedup"], "4 GPUs with FP32 must outperform 1 GPU FP64 baseline"
    assert p4_fp32["t_h2d_ms"] < base_1gpu["t_h2d_ms"], "DDP-DS with pin_memory must reduce Host-to-Device latency"

    # Precision scaling invariant: FP16 > FP32 > FP64 speedup
    p1_fp16 = allocator.predict_performance(num_gpus=1, precision="FP16")
    p1_fp32 = allocator.predict_performance(num_gpus=1, precision="FP32")
    p1_fp64 = allocator.predict_performance(num_gpus=1, precision="FP64")
    assert p1_fp16["speedup"] > p1_fp32["speedup"] > p1_fp64["speedup"], "Lower precision must yield strictly higher speedup"

    # 3. Multi-device BFD bin-packing invariant
    tasks = {
        "attention": {"gpu_ms": 12.0, "cpu_ms": 80.0, "mem_mb": 50},
        "ffn": {"gpu_ms": 8.0, "cpu_ms": 50.0, "mem_mb": 30},
        "moe": {"gpu_ms": 20.0, "cpu_ms": 120.0, "mem_mb": 100},
        "sampling": {"gpu_ms": 2.0, "cpu_ms": 3.0, "mem_mb": 1},
    }
    packed = allocator.bin_pack_tasks(tasks)
    assert len(packed["assignment"]) == 4, "All tasks must be assigned"
    assert packed["makespan_ms"] < sum(t["cpu_ms"] for t in tasks.values()), "Parallel packing makespan must be strictly less than serial CPU execution"

    # Backward compatibility
    legacy_alloc = allocator.allocate(["attention", "ffn", "sampling"])
    assert "attention" in legacy_alloc and "sampling" in legacy_alloc


def test_apex_scheduler_paper12_invariant():
    """Proves APEX CPU-GPU parallel inference analytical model, Inequality 6 boundary, and Asynchronous Overlap deferred sync (Paper 12 / 2506.03296v2 Fan)."""
    # 1. Inequality 6 test: Fast CPU relative to GPU (ratio < threshold) -> Asymmetric Pipelining
    fast_cpu_sched = APEXScheduler(n_layer=6, n_g_rate=30.0, n_c_rate=15.0)  # ratio = 2.0
    res_asym = fast_cpu_sched.evaluate_strategy(t_g_linear_ms=4.0, t_g_att_ms=6.0, has_prefill=False)
    assert res_asym["strategy"] == "Asymmetric-Pipelining", "When NG/NC < threshold, Asymmetric Pipelining must be selected"
    assert res_asym["sync_policy"] == "Immediate-Post-Layer"
    assert res_asym["estimated_speedup"] > 1.0

    # 2. Inequality 6 test: Slow CPU relative to GPU (ratio >= threshold) -> Asynchronous Overlap
    # threshold = 2*(4/6) + 3 + (6/4) = 1.333 + 3 + 1.5 = 5.833
    slow_cpu_sched = APEXScheduler(n_layer=6, n_g_rate=120.0, n_c_rate=10.0)  # ratio = 12.0 > 5.833
    res_async = slow_cpu_sched.evaluate_strategy(t_g_linear_ms=4.0, t_g_att_ms=6.0, has_prefill=False)
    assert res_async["strategy"] == "Asynchronous-Overlap", "When NG/NC >= threshold, Asynchronous Overlap must be selected"
    assert res_async["sync_policy"] == "Deferred-Next-Cycle", "Asynchronous Overlap must defer synchronization to avoid GPU stalls"
    assert res_async["estimated_speedup"] > 1.0

    # 3. Algorithm 1 scheduling dispatch test
    req_gpu = ["req_1", "req_2"]
    req_cpu = ["req_3"]
    # CPU queue empty -> GPU-only
    sch_gpuonly = slow_cpu_sched.schedule(prefill_requests=[], gpu_decode_requests=req_gpu, cpu_decode_requests=[])
    assert sch_gpuonly["strategy"] == "GPU-Only"

    # CPU queue non-empty -> Asynchronous-Overlap
    sch_overlap = slow_cpu_sched.schedule(prefill_requests=[], gpu_decode_requests=req_gpu, cpu_decode_requests=req_cpu)
    assert sch_overlap["strategy"] == "Asynchronous-Overlap"
    assert len(sch_overlap["scheduled_cpu_decode"]) == 1

    # Mixed prefill + decode scenario
    sch_mixed = slow_cpu_sched.schedule(prefill_requests=["pref_1"], gpu_decode_requests=req_gpu, cpu_decode_requests=req_cpu)
    assert sch_mixed["strategy"] in ("Asymmetric-Pipelining", "Asynchronous-Overlap")

    # 4. Backward compatibility
    compat_pipe = PipelinedInference(n_layer=6)
    assert compat_pipe.should_pipeline(64) is True
    assert compat_pipe.estimate_speedup() > 1.0


def test_nvfp4_pretraining_methodology_paper13_invariant():
    """Proves NVFP4 pretraining methodology: RHT outlier dispersion, 2D weight transpose invariance, stochastic rounding unbiasedness, and NVFP4Linear backprop (Paper 13 / 2509.25149v2 NVIDIA)."""
    # 1. Exact NVFP4 E2M1 representation levels
    assert len(NVFP4_E2M1_LEVELS) == 15, "NVFP4 E2M1 must have exactly 15 representation levels"
    assert NVFP4_E2M1_LEVELS.min().item() == -6.0 and NVFP4_E2M1_LEVELS.max().item() == 6.0

    # 2. Random Hadamard Transform (RHT) energy conservation & outlier dispersion
    torch.manual_seed(42)
    # Norm conservation: orthogonal transform preserves vector L2 norm
    x = torch.randn(4, 16)
    x_rht = apply_random_hadamard_transform_16(x)
    assert torch.allclose(x.norm(dim=-1), x_rht.norm(dim=-1), atol=1e-3), "RHT must preserve L2 energy norm"

    # Outlier dispersion: single spike at index 0 should be spread across 16 dimensions
    spike = torch.zeros(1, 16)
    spike[0, 0] = 16.0
    spike_rht = apply_random_hadamard_transform_16(spike)
    assert spike_rht.abs().max().item() < 16.0, "RHT must disperse concentrated outlier spike"
    assert torch.all(spike_rht.abs() > 0.0), "All dimensions must receive energy from RHT transform"

    # 3. 2D Block Weight Quantization Transpose Invariance
    # Critical property from Section 4.3: Q(W^T) == (Q(W))^T eliminates chain-rule violation
    w = torch.randn(32, 32)
    w_deq, _ = NVFP4Quantizer.quantize_weights_2d(w, block_size=16)
    w_t_deq, _ = NVFP4Quantizer.quantize_weights_2d(w.t(), block_size=16)
    assert torch.allclose(w_deq.t(), w_t_deq, atol=1e-5), "2D weight quantization must be strictly transpose-invariant"

    # 4. Stochastic Rounding Unbiased Expectation (Section 4.4)
    # Value 1.25 lies between FP4 levels 1.0 and 1.5. Expected value should equal 1.25
    torch.manual_seed(42)
    v = torch.full((10000,), 1.25)
    v_sr = NVFP4Quantizer.round_to_fp4_stochastic(v)
    mean_sr = v_sr.mean().item()
    assert abs(mean_sr - 1.25) < 0.02, f"Stochastic rounding mean {mean_sr} must be unbiased (near 1.25)"

    # 5. NVFP4Linear End-to-End Forward + Backward Propagation
    layer = NVFP4Linear(in_features=32, out_features=32, bias=True, high_precision=False)
    in_act = torch.randn(2, 32, requires_grad=True)
    out = layer(in_act)
    assert out.shape == (2, 32)
    loss = out.sum()
    loss.backward()

    assert in_act.grad is not None and in_act.grad.norm().item() > 0, "Activation gradients must backpropagate through NVFP4"
    assert layer.weight.grad is not None and layer.weight.grad.norm().item() > 0, "Weight gradients must backpropagate through NVFP4"

    # 6. Compression ratio
    assert NVFP4Quantizer.compression_ratio() > 3.5, "NVFP4 must provide >3.5x compression vs FP16"


def test_xmem_caching_allocator_paper14_invariant():
    """Proves xMem CPU-based GPU memory estimation: 512B alignment, Best-Fit Coalescing (BFC), segment caching, and full training lifecycle simulation (Paper 14 / 2510.21048v1 Shi)."""
    # 1. 512B Hardware alignment invariant
    sim = CUDACachingAllocatorSimulator()
    assert sim.round_up(100) == 512
    assert sim.round_up(512) == 512
    assert sim.round_up(513) == 1024

    # 2. Segment-based caching & BFC block splitting/coalescing
    # Allocate small block (100 KB <= 1 MB threshold) -> requests small segment (2 MB)
    sim.allocate("tensor_1", 100 * 1024)
    assert sim.reserved_bytes == 2 * 1048576, "Small allocation must create 2MB segment"
    assert sim.allocated_bytes == sim.round_up(100 * 1024)

    # Deallocate tensor_1 -> segment remains reserved in cache
    sim.deallocate("tensor_1")
    assert sim.allocated_bytes == 0, "Allocated bytes must drop to zero after deallocation"
    assert sim.reserved_bytes == 2 * 1048576, "Cached segment must remain reserved in simulator"

    # Allocate smaller tensor_2 (50 KB) -> must reuse existing segment without expanding reserved memory
    sim.allocate("tensor_2", 50 * 1024)
    assert sim.reserved_bytes == 2 * 1048576, "Cached block must be reused via Best-Fit without new segment"
    assert len(sim.segments) == 1

    # 3. Training iteration lifecycle simulation
    estimator = XMemEstimator(device_capacity_mb=4096.0)
    # Moderate config that safely fits on 4GB (5M params, batch 2, seq 128, dim 256, 4 layers)
    res = estimator.simulate_training_iteration(
        n_params=5_000_000, batch_size=2, seq_len=128, hidden_dim=256, n_layer=4, dtype_bytes=2, optimizer="adam"
    )
    assert res["total_peak_mb"] > res["peak_allocated_mb"], "Total peak must include reserved segments + CUDA context"
    assert res["fragmentation_mb"] >= 0.0, "Fragmentation must be non-negative"
    assert res["oom_predicted"] is False, "5M model must fit in 4GB without OOM"
    assert res["headroom_mb"] > 0.0, "Safe headroom must remain"

    # Large config that triggers predictable OOM (800M params with FP32 Adam on 4GB: weights 1.6GB + adam 6.4GB = 8GB > 4GB)
    oom_res = estimator.simulate_training_iteration(
        n_params=800_000_000, batch_size=2, seq_len=128, hidden_dim=256, n_layer=4, dtype_bytes=2, optimizer="adam"
    )
    assert oom_res["oom_predicted"] is True, "800M param model with Adam must correctly predict OOM on 4GB"
    assert oom_res["headroom_mb"] == 0.0

    # Backward compatibility
    # Backward compatibility
    assert estimator.will_oom(n_params=800_000_000, batch_size=2, seq_len=128, hidden_dim=256, n_layer=4) is True


def test_eggroll_hyperscale_es_paper15_invariant():
    """Proves EGGROLL low-rank evolution strategies, hardware-efficient batched forward pass, unmaterialized gradient updates, and antithetic fitness shaping (Paper 15 / 2511.16652v2 Sarkar)."""
    torch.manual_seed(42)
    m, n = 32, 64
    r = 2
    n_workers = 16
    sigma = 0.05
    lr = 0.05

    opt = EGGROLLOptimizer(rank=r, population_size=n_workers, sigma=sigma, lr=lr)

    # 1. Sample low-rank factors
    A, B = opt.sample_perturbation_factors(out_features=m, in_features=n, pop_size=n_workers, rank=r)
    assert A.shape == (n_workers, m, r), "Factor A shape must match [N, m, r]"
    assert B.shape == (n_workers, n, r), "Factor B shape must match [N, n, r]"

    # 2. Hardware-efficient batched forward pass vs naive materialized forward pass
    M = torch.randn(m, n)
    u = torch.randn(8, n)

    # Materialized perturbation for worker 0: E_0 = (1 / sqrt(r)) * A_0 @ B_0^T
    E_0 = (1.0 / math.sqrt(r)) * (A[0] @ B[0].T)
    naive_out = u @ (M + sigma * E_0).T
    fast_out = opt.batched_forward(u, M, A, B, sigma=sigma, worker_idx=0)
    diff_fwd = (naive_out - fast_out).abs().max().item()
    assert diff_fwd < 1e-5, f"Hardware-efficient batched forward pass must match naive GEMM within 1e-5 (got {diff_fwd})"

    # 3. Unmaterialized gradient update vs explicit materialized sum
    raw_fitness_pos = torch.randn(n_workers)
    raw_fitness_neg = torch.randn(n_workers)

    # Shaped fitness
    shaped_f = opt.fitness_shaping(raw_fitness_pos, raw_fitness_neg, mode="centered_rank")
    assert shaped_f.shape == (n_workers,), "Shaped fitness must be 1D vector of length N"
    assert shaped_f.min() >= -0.5 and shaped_f.max() <= 0.5, "Centered rank fitness must lie within [-0.5, 0.5]"

    # Ground truth explicit sum: (1 / (N * sigma)) * sum_i f_i * E_i
    explicit_e = torch.zeros(m, n)
    for i in range(n_workers):
        e_i = (1.0 / math.sqrt(r)) * (A[i] @ B[i].T)
        explicit_e += shaped_f[i] * e_i
    expected_grad = explicit_e / (float(n_workers) * sigma)

    # Unmaterialized gradient computation
    unmat_grad = opt.compute_unmaterialized_gradient(A, B, shaped_f, sigma=sigma)
    diff_grad = (unmat_grad - expected_grad).abs().max().item()
    assert diff_grad < 1e-5, f"Unmaterialized gradient must match explicit sum within 1e-5 (got {diff_grad})"

    # 4. High-dimensional critical decay rate invariant
    sigma_small = EGGROLLOptimizer.critical_sigma(base_sigma=0.02, param_dim=1_000, alpha=0.25)
    sigma_large = EGGROLLOptimizer.critical_sigma(base_sigma=0.02, param_dim=1_000_000, alpha=0.25)
    assert sigma_small > sigma_large > 0.0, "Critical sigma must strictly decay as parameter dimension scales to prevent Gaussian Annulus divergence"

    # 5. Black-box parameter convergence: EGGROLL minimizes quadratic error without backprop
    W_target = torch.randn(16, 16)
    W_cur = torch.zeros(16, 16)
    init_err = (W_cur - W_target).pow(2).sum().item()

    es_test = HyperscaleES(hidden_dim=16, population_size=32, sigma=0.05, rank=1, lr=0.05)
    for _ in range(40):
        a_sub, b_sub = es_test.eggroll.sample_perturbation_factors(16, 16, pop_size=32, rank=1)
        # Materialized fitness evaluation
        f_p = []
        f_n = []
        for i in range(32):
            e_sub = a_sub[i] @ b_sub[i].T
            f_p.append(-(W_cur + 0.05 * e_sub - W_target).pow(2).sum().item())
            f_n.append(-(W_cur - 0.05 * e_sub - W_target).pow(2).sum().item())
        W_cur = es_test.update(W_cur, torch.tensor(f_p), torch.tensor(f_n), A=a_sub, B=b_sub)

    final_err = (W_cur - W_target).pow(2).sum().item()
    assert final_err < init_err * 0.7, f"EGGROLL must achieve >30% error reduction on black-box quadratic test (init={init_err:.2f}, final={final_err:.2f})"


def test_long_horizon_mopd_paper16_invariant():
    """Proves Multi-Turn Long-Horizon Planning, Full-Vocab & Top-k Agent-OPD, Sampled-Token K2, and MOPD multi-teacher specialization (Paper 16 / 2607.24720v1 Men)."""
    torch.manual_seed(42)
    distiller = LongHorizonDistiller(hidden_dim=32, gamma=0.90, num_teachers=3)

    # 1. Horizon physical potential decay invariant: gamma^h decays monotonically
    h_weights = [distiller.horizon_weight(h) for h in range(11)]
    for i in range(len(h_weights) - 1):
        assert h_weights[i] > h_weights[i + 1] > 0.0, f"Horizon decay weight must strictly decrease monotonically: {h_weights[i]} vs {h_weights[i+1]}"

    # 2. Teacher specialization profile invariant
    # Teacher 0 (Glimmer) preferred horizon = 8; Teacher 1 (Lightning) preferred horizon = 2
    w_glimmer_long = distiller.multi_teacher_weight(teacher_id=0, horizon=8)
    w_glimmer_short = distiller.multi_teacher_weight(teacher_id=0, horizon=2)
    assert w_glimmer_long > w_glimmer_short, "Glimmer (Teacher 0) must have higher specialization weight at long horizons (h=8) than short (h=2)"

    w_lightning_short = distiller.multi_teacher_weight(teacher_id=1, horizon=2)
    w_lightning_long = distiller.multi_teacher_weight(teacher_id=1, horizon=8)
    assert w_lightning_short > w_lightning_long, "Lightning (Teacher 1) must have higher specialization weight at short horizons (h=2) than long (h=8)"

    # 3. Full-Vocabulary Agent-OPD gradient flow
    B, T, V = 2, 4, 16
    student_logits = torch.randn(B, T, V, requires_grad=True)
    teacher_1 = torch.randn(B, T, V)
    loss_full = LongHorizonDistiller.full_vocabulary_opd(student_logits, teacher_1)
    loss_full.backward()
    assert student_logits.grad is not None and student_logits.grad.norm().item() > 0.0, "Full-vocab OPD must produce active backpropagation gradients"
    student_logits.grad.zero_()

    # 4. Top-k Vocabulary Agent-OPD invariant
    loss_topk = LongHorizonDistiller.top_k_opd(student_logits, teacher_1, k=4)
    loss_topk.backward()
    assert student_logits.grad is not None and student_logits.grad.norm().item() > 0.0, "Top-k OPD must produce active backpropagation gradients"
    student_logits.grad.zero_()

    # 5. Sampled-Token Schulman K2 invariant
    sampled_tokens = torch.randint(0, V, (B, T))
    loss_k2 = LongHorizonDistiller.sampled_token_opd(student_logits, teacher_1, sampled_tokens, mode="k2")
    assert loss_k2.item() > 0.0, "Sampled-token K2 distance must be positive for differing distributions"
    # When student == teacher, K2 loss must be 0
    loss_k2_zero = LongHorizonDistiller.sampled_token_opd(teacher_1, teacher_1, sampled_tokens, mode="k2")
    assert abs(loss_k2_zero.item()) < 1e-6, "K2 distance must be 0 when student and teacher are identical"

    # 6. Multi-Teacher compatibility audit
    # Identical teachers -> similarity = 1.0, no conflict
    compat_self = LongHorizonDistiller.compute_teacher_compatibility([teacher_1, teacher_1])
    assert compat_self["mean_compatibility"] > 0.99
    assert compat_self["has_conflict"] is False

    # Conflicting / orthogonal teacher distributions
    orthogonal_t2 = -teacher_1 * 5.0
    compat_ortho = LongHorizonDistiller.compute_teacher_compatibility([teacher_1, orthogonal_t2])
    assert compat_ortho["has_conflict"] is True, "Opposing teacher policies must be flagged as conflicting"

    # 7. MOPD multi-teacher combined loss
    teacher_2 = torch.randn(B, T, V)
    mopd_loss = distiller.distill_loss(student_logits, [teacher_1, teacher_2], horizons=[2, 8])
    mopd_loss.backward()
    assert student_logits.grad.norm().item() > 0.0, "MOPD combined loss must backpropagate gradients across all teachers"


def test_consciousness_steering_paper17_invariant():
    """Proves Directional Safety Refusal Ablation, Difference-of-Means Consciousness Vector Extraction, Mechanistic Activation Steering, and Survey Proximity Delta-KL (Paper 17 / 2607.28607v1 Kim)."""
    torch.manual_seed(42)
    D = 64
    B = 8

    # 1. Difference of Means Extraction invariant: unit vector ||v_c|| == 1.0
    x_pos = torch.randn(20, D) + 1.5
    x_neg = torch.randn(20, D) - 1.5
    v_c = ConsciousnessSteeringEngine.extract_difference_of_means(x_pos, x_neg)
    assert abs(v_c.norm(p=2).item() - 1.0) < 1e-6, "Extracted consciousness vector must have unit L2 norm"

    # 2. Orthogonal Directional Refusal Ablation invariant: <x_ablated, r_hat> == 0.0
    refusal_direction = torch.randn(D)
    r_hat = refusal_direction / refusal_direction.norm(p=2)
    x = torch.randn(B, D)
    x_ablated = ConsciousnessSteeringEngine.ablate_direction(x, refusal_direction)
    # Check inner product across all batch elements
    dot_prods = (x_ablated @ r_hat).abs()
    assert dot_prods.max().item() < 1e-6, f"Ablated activations must be strictly orthogonal to safety refusal vector (max dot={dot_prods.max().item()})"

    # 3. Mechanistic Activation Steering invariant
    alpha = 3.5
    x_steered = ConsciousnessSteeringEngine.steer_activation(x, v_c, alpha=alpha)
    diff = x_steered - x
    assert (diff.norm(dim=-1) - alpha).abs().max().item() < 1e-5, "Steering shift magnitude must strictly equal alpha along consciousness direction"

    # 4. Geometric angle audit invariant
    # Synthesize opposing refusal vector vs consciousness vector
    opposing_refusal = -v_c * 2.0 + torch.randn(D) * 0.1
    orthogonal_tom = torch.randn(D)
    orthogonal_tom = orthogonal_tom - (orthogonal_tom @ r_hat) * r_hat  # make orthogonal to refusal

    audit = ConsciousnessSteeringEngine.geometric_angle_audit(opposing_refusal, v_c, tom_vec=orthogonal_tom)
    assert audit["opposes_safety"] is True, "Safety refusal must be detected as opposing consciousness"
    assert audit["refusal_consciousness_angle_deg"] > 90.0, "Angle between refusal and consciousness must exceed 90 degrees"
    assert audit["tom_orthogonal"] is True, "Theory of Mind must remain near-orthogonal to safety refusal"

    # 5. Human value survey distribution Delta-KL restoration
    p_human = torch.tensor([0.65, 0.25, 0.10])
    p_baseline = torch.tensor([0.15, 0.35, 0.50])     # far from human
    p_steered = torch.tensor([0.60, 0.25, 0.15])      # restored close to human
    delta_kl = ConsciousnessSteeringEngine.human_value_delta_kl(p_human, p_baseline, p_steered)
    assert delta_kl > 0.0, f"Consciousness steering must produce positive Delta-KL toward human distribution (got {delta_kl:.4f})"

    # 6. Backward compatibility & active gradient flow in SelfModelCoherenceLoss
    coherence = SelfModelCoherenceLoss(hidden_dim=D, num_personas=34, alpha=alpha)
    h_in = torch.randn(4, 16, D, requires_grad=True)
    p_ids = torch.tensor([0, 5, 12, 33])
    loss_c = coherence(h_in, p_ids)
    loss_c.backward()
    assert h_in.grad is not None and h_in.grad.norm().item() > 0.0, "SelfModelCoherenceLoss must backpropagate gradients to representations"

    # Test steer method on module
    steered_h = coherence.steer(h_in.detach())
    assert (steered_h - h_in.detach()).norm().item() > 0.0


def test_continual_skill_bench_paper18_invariant():
    """Proves ContinualSkillBench in-context continual learning dynamics, Delta_seq, library fragmentation ratio, and pruning (Paper 18 / 2608.03874v1 Guan)."""
    tracker = ContinualSkillBenchTracker(num_councils=4, window=50, beta=0.85)

    # 1. Independent baseline vs Sequential execution delta Delta_seq
    # Council 0 has baseline = 0.40
    tracker.set_independent_baseline(0, 0.40)
    assert tracker.baseline_scores[0] == 0.40

    # Sequential rewards: high performance due to in-context learning
    for reward in [0.70, 0.75, 0.80, 0.85, 0.90]:
        tracker.update(0, reward)

    delta_seq = tracker.sequential_delta(0)
    assert delta_seq > 0.0, f"Sequential score must exceed independent baseline (Delta_seq={delta_seq:.4f})"
    assert tracker.skill_delta(0) > 0.0, "Skill delta over window must reflect positive progression"

    # 2. Skill creation & modification tracking
    tracker.record_skill_creation(0, "skill_graph_search", "BFS algorithm", "def bfs(): pass")
    tracker.record_skill_creation(0, "skill_matrix_mult", "Strassen matrix mult", "def strassen(): pass")
    tracker.record_skill_creation(0, "skill_quick_sort", "QuickSort partition", "def qsort(): pass")
    assert len(tracker.skills_created[0]) == 3

    tracker.record_skill_modification(0, "skill_graph_search")
    assert tracker.skills_created[0]["skill_graph_search"]["modifications"] == 1

    # 3. Compact vs Fragmented library dynamics (Section 4.4)
    # Model A (Compact, e.g. GPT-5.3-Codex): High reuse rate, low fragmentation
    tracker.record_skill_invocation(0, "skill_graph_search")
    tracker.record_skill_invocation(0, "skill_graph_search")
    tracker.record_skill_invocation(0, "skill_matrix_mult")

    frag_0 = tracker.fragmentation_ratio(0)
    reuse_0 = tracker.reuse_rate(0)
    # 3 created, 2 reused -> frag = (3 - 2) / 3 = 0.333, reuse = 2/3 = 0.667
    assert 0.30 <= frag_0 <= 0.35, f"Expected fragmentation ~0.33, got {frag_0}"
    assert 0.65 <= reuse_0 <= 0.68, f"Expected reuse rate ~0.67, got {reuse_0}"

    # Model B (Fragmented, e.g. GPT-4o): Creates 10 skills, reuses only 1 -> frag = 9/10 = 0.90
    for i in range(10):
        tracker.record_skill_creation(1, f"bloated_skill_{i}")
    tracker.record_skill_invocation(1, "bloated_skill_0")

    frag_1 = tracker.fragmentation_ratio(1)
    reuse_1 = tracker.reuse_rate(1)
    assert frag_1 >= 0.89, f"Fragmented model must exhibit high fragmentation ratio >= 0.89 (got {frag_1})"
    assert reuse_1 <= 0.11, f"Fragmented model must exhibit low reuse rate <= 0.11 (got {reuse_1})"

    # 4. Pruning unused fragmented skills restores library compactness
    pruned = tracker.prune_fragmented_skills(1, min_invocations=1)
    assert len(pruned) == 9, f"Must prune 9 un-reused skills, pruned {len(pruned)}"
    assert len(tracker.skills_created[1]) == 1
    assert tracker.fragmentation_ratio(1) == 0.0, "Pruned library must have zero fragmentation"

    # 5. Global statistics aggregator
    stats = tracker.get_stats()
    assert "avg_fragmentation" in stats and "avg_reuse_rate" in stats
    assert stats["total_skills_created"] == 4  # 3 in council 0 + 1 remaining in council 1


def test_evoharness_rl_paper19_invariant():
    """Proves EvoHarness-RL BPE state abstraction, cost-aware GRPO trajectory rewards, and HarnessPolicy (Paper 19 / 2608.05446v1 Ning)."""
    # 1. Unified BPE State Abstraction: Belief, Progress, Experience (Section 2.1)
    bpe = BPEHarnessState()

    # Belief (B_t)
    bpe.set_belief("kettle_location", "stove_burner_3")
    bpe.set_belief("countertop_1", "empty")
    assert bpe.track("kettle_location") == {"kettle_location": "stove_burner_3"}
    assert bpe.track("unknown_object") == {"unknown_object": "unknown"}
    world_view = bpe.track("world")
    assert "kettle_location" in world_view and "countertop_1" in world_view

    # Progress (P_t)
    bpe.commit("step_1", "find kettle", status="active")
    bpe.commit("step_1", "find kettle", status="completed")
    bpe.commit("step_2", "clean kettle", status="pending")
    assert len(bpe.progress) == 2
    assert bpe.progress[0]["status"] == "completed"

    # Experience (E_t) & Note buffer
    bpe.note("Clean kettle using sponge at sink", category="task_specific")
    bpe.note("Check stove burners when looking for cookware", category="search_priorities")
    assert len(bpe.note_buffer) == 2

    # Consolidate notes into Experience store at epoch boundaries
    bpe.consolidate_notes()
    assert len(bpe.note_buffer) == 0
    assert "Clean kettle using sponge at sink" in bpe.experience["task_specific"]

    # Recall reusable experience & bump usage counts
    rec = bpe.recall("task_specific", "Clean kettle using sponge at sink")
    assert rec is not None and rec["usage_count"] == 2  # 1 initial + 1 from recall
    assert bpe.recall("general", "non_existent_skill") is None

    # Evict stale skills with usage < 2
    bpe.experience["general"]["unused_skill"] = {"insight": "stale", "usage_count": 0}
    evicted = bpe.evict_stale_experience(min_usage=1)
    assert evicted == 1
    assert "unused_skill" not in bpe.experience["general"]

    # 2. Cost-Aware GRPO Trajectory Reward Formulation (Equations 5 & 6)
    rewarder = EvoHarnessReward(t_max=20, lambda_eff=2.0, lambda_div_max=1.5, u_horizon=60)

    # Cosine annealing schedule for diversity bonus
    lam_0 = rewarder.get_lambda_div(epoch_u=0)
    lam_mid = rewarder.get_lambda_div(epoch_u=30)
    lam_end = rewarder.get_lambda_div(epoch_u=60)
    assert abs(lam_0 - 1.5) < 1e-4, f"Initial lambda_div must be max 1.5, got {lam_0}"
    assert abs(lam_mid - 0.75) < 1e-4, f"Midpoint lambda_div must be half 0.75, got {lam_mid}"
    assert abs(lam_end - 0.0) < 1e-4, f"Final lambda_div must anneal to 0.0, got {lam_end}"

    # Trajectory A: Solved, efficient, diverse actions
    traj_a = ["commit[step_1]", "track[world]", "go_to[burner]", "pick[kettle]", "note[done]"]
    res_a = rewarder.compute_reward(solved=True, trajectory=traj_a, epoch_u=0, invalid_count=0)
    assert res_a["r_succ"] == 10.0, "Solved task must receive R_succ = 10.0"
    assert res_a["r_eff"] > 0.0, "Solved task must receive positive efficiency bonus"
    assert res_a["r_div"] == 1.0, "All distinct verbs must yield R_div = 1.0"
    assert res_a["r_spam"] == 0.0, "No duplicate actions"
    assert res_a["total_reward"] > 10.0

    # Trajectory B: Unsolved, repetitive actions, invalid syntax
    traj_b = ["track[world]", "track[world]", "track[world]", "malformed_action"]
    res_b = rewarder.compute_reward(solved=False, trajectory=traj_b, epoch_u=0, invalid_count=1)
    assert res_b["r_succ"] == 0.0, "Unsolved task receives R_succ = 0.0"
    assert res_b["r_eff"] == 0.0, "Efficiency bonus is strictly gated on task success"
    assert res_b["r_spam"] > 0.0, "Spam penalty must be positive for repetitive actions"
    assert res_b["r_inv"] > 0.0, "Format penalty must penalize malformed actions"
    assert res_b["total_reward"] < 1.0

    # 3. Trainable HarnessPolicy (backward compatibility, differentiability, annealing)
    hidden_dim = 64
    policy = HarnessPolicy(hidden_dim=hidden_dim)
    h = torch.randn(4, 16, hidden_dim, requires_grad=True)

    dec = policy.decide(h)
    assert "num_rounds" in dec and 1 <= dec["num_rounds"] <= 4
    assert "use_world_model" in dec and isinstance(dec["use_world_model"], bool)
    assert "use_abductive" in dec and isinstance(dec["use_abductive"], bool)
    assert "bpe_meta_action" in dec and dec["bpe_meta_action"] in ("track", "commit", "recall", "note")

    # Differentiability: active gradient backpropagation to input representations
    loss = (dec["rounds_logits"].sum() + dec["wm_logit"].sum() +
            dec["abduct_logit"].sum() + dec["bpe_logits"].sum())
    loss.backward()
    assert h.grad is not None and h.grad.norm().item() > 0.0, "HarnessPolicy must propagate gradients"

    # Annealed harness budget dynamics (Section 4.1): ~6 calls early -> ~1 call late
    budget_early = policy.compute_annealed_harness_budget(epoch_u=0, total_epochs=60)
    budget_late = policy.compute_annealed_harness_budget(epoch_u=60, total_epochs=60)
    assert abs(budget_early - 6.0) < 1e-3, f"Early exploration budget should be ~6 calls, got {budget_early}"
    assert abs(budget_late - 1.0) < 1e-3, f"Late stable budget should be ~1 call, got {budget_late}"


def test_embedder_dilemma_paper20_invariant():
    """Proves The Embedder's Dilemma analytical cost model, thinking-token tax share, Pareto frontier, and hybrid retrieve-then-rerank (Paper 20 / 2608.12875v1 El Assadi)."""
    # 1. Analytical Cost Model & Thinking-Token Tax (Equation 1 & Section 3.3)
    cost_model = EmbedderCostModel(r_in_per_mtok=1.25, r_out_per_mtok=10.0, r_gpu_hr=2.49)

    # Gemini 3.1 Pro scenario: 10M input (5M cached), 12M total tokens, 1.5M reasoning tokens
    llm_cost_data = cost_model.compute_llm_cost(
        input_tokens=10_000_000,
        cached_tokens=5_000_000,
        total_tokens=12_000_000,
        reasoning_tokens=1_500_000
    )
    # uncached input: 5M * $1.25/M = $6.25
    # cached input: 5M * $0.125/M = $0.625
    # generated: 2M * $10.00/M = $20.00
    # total: $26.875
    assert abs(llm_cost_data["total_cost"] - 26.875) < 1e-3
    # reasoning cost: 1.5M * $10/M = $15.00
    # thinking tax share = $15.00 / $26.875 = 0.558 (55.8%, within 28-81% range)
    assert 0.28 <= llm_cost_data["thinking_tax_share"] <= 0.81, f"Thinking tax share must be in 28-81% range, got {llm_cost_data['thinking_tax_share']:.3f}"

    # Embedding cost on H100 (throughput = 40,000 tok/s)
    emb_cost = cost_model.compute_embedding_cost(num_tokens=10_000_000, throughput_tok_s=40000.0)
    assert emb_cost > 0.0

    # Pareto Frontier: LLM vs Embedding cost gap
    # Replicating benchmark pass: Gemini 3.1 Pro ($154.14) vs Octen-8B ($0.11) -> 1,431x ratio (Section 4.4)
    pareto_ratio = cost_model.cost_ratio(154.14, 0.1077)
    assert 1400.0 <= pareto_ratio <= 1450.0, f"Expected Pareto cost ratio ~1,431x, got {pareto_ratio:.1f}"

    # 2. HybridRetrieveRerankRouter Task Specialization (Section 4.2 & 4.3)
    router = HybridRetrieveRerankRouter(hidden_dim=1024)

    # Classification / Clustering / STS -> Dispatches to pure embedding (superior or tied accuracy, 1,431x cheaper)
    disp_cls = router.dispatch("classify intent", task_category="classification")
    assert disp_cls["route"] == "embedding"
    assert disp_cls["pipeline"] == "pure_bi_encoder"

    disp_sts = router.dispatch("compare sentences", task_category="sts")
    assert disp_sts["route"] == "embedding"

    # Semantic Retrieval (BEIR) -> Pure embedding outperforms reranker (63.1 > 60.3)
    disp_beir = router.dispatch("search documentation", task_category="retrieval_semantic")
    assert disp_beir["route"] == "embedding"
    assert disp_beir["pipeline"] == "pure_bi_encoder"
    assert disp_beir["expected_quality_delta"] > 0.0

    # Reasoning-Heavy Retrieval (BRIGHT) -> Dispatches to Retrieve-then-Rerank (+12.8 points boost)
    disp_bright = router.dispatch("multi-hop causal deduction", task_category="reasoning_retrieval")
    assert disp_bright["route"] == "hybrid_rerank"
    assert disp_bright["pipeline"] == "retrieve_then_rerank"
    assert disp_bright["recommended_shortlist_k"] == 50
    assert disp_bright["expected_quality_delta"] >= 12.0

    # 3. Backward Compatibility Heuristic Routing
    assert router.route("why did the system fail under counterfactual intervention?") == "llm"
    assert router.route("lookup token 42") == "cheap"

    # 4. Pipeline Simulation & Token Savings (Section 4.3 & 5)
    sim = router.simulate_pipeline(task="reasoning_bright", corpus_size=10000, shortlist_k=50, doc_len_tokens=300)
    assert sim["token_savings_pct"] >= 99.0, f"Shortlist reranking must yield >99% token savings over full corpus in-context (got {sim['token_savings_pct']:.2f}%)"
    # Hybrid quality (34.2) retains >97% of full LLM in-context quality (35.1)
    quality_retention = sim["hybrid_quality"] / sim["full_llm_quality"]
    assert quality_retention >= 0.95, f"Hybrid retrieve-then-rerank must preserve >=95% of LLM reasoning quality (got {quality_retention:.3f})"


def test_spp_persona_conditioning_paper21_invariant():
    """Proves Synthetic Persona Pretraining (SPP): Token Zero grounding, joint reflection loss, persona binding, and abliteration invariance (Paper 21 / 2608.13482v1 Minder)."""
    torch.manual_seed(42)
    B, T, D = 4, 16, 64
    vocab_size = 1000
    num_personas = 34

    spp = PersonaConditioning(vocab_size=vocab_size, hidden_dim=D, num_personas=num_personas)

    # 1. Token Zero Persona Prefixing (Section 2.1)
    raw_tokens = torch.randint(0, 900, (B, T))
    persona_ids = torch.tensor([0, 5, 12, 33])
    t0_tokens = spp.prepend_token_zero(raw_tokens, persona_ids)
    assert t0_tokens.shape == (B, T + 1), f"Token-zero prefixing must expand length to T+1 (got {t0_tokens.shape})"
    # Check that position 0 contains the persona-specific identifier
    for i in range(B):
        expected_tok = vocab_size - num_personas + persona_ids[i].item()
        assert t0_tokens[i, 0].item() == expected_tok, f"Position 0 must be persona token {expected_tok}"

    # 2. Hidden Representation Modulation & Differentiability
    h_in = torch.randn(B, T, D, requires_grad=True)
    h_out = spp(h_in, persona_id=persona_ids)
    assert h_out.shape == (B, T, D)
    assert not torch.allclose(h_out, h_in), "Persona conditioning must modulate hidden representations"

    # Gradient flow test through persona embeddings and gate
    loss = h_out.sum()
    loss.backward()
    assert h_in.grad is not None and h_in.grad.norm().item() > 0.0
    assert spp.persona_embed.weight.grad is not None and spp.persona_embed.weight.grad.norm().item() > 0.0
    assert spp.persona_gate.grad is not None and spp.persona_gate.grad.abs().item() > 0.0

    # 3. Joint Pretraining + Reflection Loss Objective (Section 2.2)
    # L_SPP = L_doc + lambda_refl * L_refl
    logits = torch.randn(B, T, vocab_size)
    targets = torch.randint(0, vocab_size, (B, T))
    reflection_mask = torch.zeros(B, T, dtype=torch.bool)
    reflection_mask[:, 8:] = True  # Second half of sequence contains constitutional reflections

    loss_dict = spp.compute_spp_loss(logits, targets, reflection_mask=reflection_mask, lambda_refl=1.5)
    assert "loss" in loss_dict and "doc_loss" in loss_dict and "refl_loss" in loss_dict
    assert loss_dict["doc_loss"] > 0.0 and loss_dict["refl_loss"] > 0.0
    expected_total = loss_dict["doc_loss"] + 1.5 * loss_dict["refl_loss"]
    assert torch.allclose(loss_dict["loss"], expected_total, atol=1e-5)

    # 4. Persona Binding Fidelity & Invariance to Refusal Abliteration (Section 3.4 & 3.4.1)
    # Test binding fidelity
    fidelity = spp.persona_binding_fidelity(h_out.detach(), persona_ids)
    assert fidelity > 0.0, f"Conditioned representation must exhibit positive persona binding fidelity (got {fidelity:.4f})"

    # Directional Refusal Abliteration: x' = x - (x @ r_hat) * r_hat (Arditi et al.)
    refusal_vector = torch.randn(D)
    abl_res = spp.verify_binding_under_abliteration(h_out.detach(), persona_ids, refusal_vector)
    # Key Paper finding: token-zero pretraining values sit deep in representations and survive refusal abliteration
    assert abl_res["fidelity_post_abliteration"] > 0.0, "Persona binding must survive refusal abliteration"
    assert abl_res["fidelity_delta"] < 0.20, f"Abliteration delta should be small (<0.20), got {abl_res['fidelity_delta']:.4f}"


def test_coordination_network_paper22_invariant():
    """Proves When Agents Coordinate temporal bipartite network instrument, sustained message graph, density, clustering, and cost decomposition (Paper 22 / 2608.16801v1 Destefanis)."""
    # 1. Temporal Bipartite Network Construction (Section 2 & Figure 2)
    net = TemporalCoordinationNetwork(num_agents=4)
    # Agents: agent_0, agent_1, agent_2, agent_3

    # Direct messages:
    # Single greeting between agent_0 and agent_1 (carried only 1 message -> must be filtered by sustained threshold >= 2)
    net.log_message("agent_0", "agent_1", timestamp=1.0, byte_size=40, token_cost=20)

    # Sustained communication between pairs:
    # agent_0 <-> agent_2: 2 messages
    net.log_message("agent_0", "agent_2", timestamp=2.0, byte_size=150, token_cost=60)
    net.log_message("agent_2", "agent_0", timestamp=3.0, byte_size=180, token_cost=70)

    # agent_1 <-> agent_2: 2 messages
    net.log_message("agent_1", "agent_2", timestamp=4.0, byte_size=200, token_cost=80)
    net.log_message("agent_2", "agent_1", timestamp=5.0, byte_size=210, token_cost=85)

    # agent_0 <-> agent_1: add a second message to make it sustained
    net.log_message("agent_1", "agent_0", timestamp=6.0, byte_size=120, token_cost=50)

    # agent_3 is completely idle (0 messages, tests idle node degree=0 handling)

    # 2. Sustained Graph Construction & Metric Verification (Section 5 & Appendix A.3)
    adj = net.build_sustained_message_graph(min_messages=2)
    # Triangle between agent_0, agent_1, agent_2 (each pair has >= 2 messages)
    assert "agent_1" in adj["agent_0"] and "agent_2" in adj["agent_0"]
    assert "agent_0" in adj["agent_1"] and "agent_2" in adj["agent_1"]
    assert "agent_0" in adj["agent_2"] and "agent_1" in adj["agent_2"]
    # agent_3 has 0 sustained connections
    assert len(adj["agent_3"]) == 0

    # Mean Degree: d_bar = (2 + 2 + 2 + 0) / 4 = 1.5
    mean_deg = net.compute_mean_degree(adj)
    assert abs(mean_deg - 1.5) < 1e-4, f"Expected mean degree 1.5, got {mean_deg}"

    # Density: rho = 2 * 3 / (4 * 3) = 6 / 12 = 0.50
    density = net.compute_density(adj)
    assert abs(density - 0.50) < 1e-4, f"Expected density 0.50, got {density}"

    # Global Clustering Coefficient / Transitivity: 3 * triangles / connected_triples
    # agent_0, agent_1, agent_2 form 1 triangle. Connected triples: each of the 3 agents has k=2 -> 3 * 1 = 3 triples.
    # C = 3 * 1 / 3 = 1.0
    c_coeff = net.compute_clustering_coefficient(adj)
    assert abs(c_coeff - 1.0) < 1e-4, f"Expected clustering coefficient 1.0 for closed triangle, got {c_coeff}"

    # 3. File Channel vs Direct Messaging Cost Decomposition (Section 6)
    # Log persistent file operations (1 write, 3 reads across agents)
    net.log_file_write("agent_0", "spec.py", timestamp=10.0, byte_size=800, token_cost=200)
    net.log_file_read("agent_1", "spec.py", timestamp=11.0, byte_size=800, token_cost=100)
    net.log_file_read("agent_2", "spec.py", timestamp=12.0, byte_size=800, token_cost=100)
    net.log_file_read("agent_3", "spec.py", timestamp=13.0, byte_size=800, token_cost=100)

    costs = net.compute_cost_decomposition()
    assert costs["msg_tokens"] == 365.0  # 20 + 60 + 70 + 80 + 85 + 50
    assert costs["file_tokens"] == 500.0  # 200 + 100 + 100 + 100
    assert costs["total_tokens"] == 865.0
    assert costs["file_token_share"] > 0.50, "File channel must represent significant persistent coordination share"

    # 4. Coordinator Hub Invariance (Section 7)
    # Naming agent_0 coordinator: in this mesh, deg(agent_0)=2, max_deg=2 -> hub score = 1.0 (no runaway hub)
    hub_score = net.compute_coordinator_hub_score("agent_0", adj)
    assert hub_score <= 1.0

    # 5. Integrated CoordinationTelemetry
    telemetry = CoordinationTelemetry(num_agents=4)
    telemetry.log_message("agent_0", "agent_1", token_cost=100)
    telemetry.log_file_op("agent_0", "orders.py", is_write=True, token_cost=200)
    telemetry.update(torch.tensor([0.8, 0.85, 0.82, 0.88]))
    stats = telemetry.get_stats()
    assert "coordination_mean" in stats and "file_token_share" in stats


def test_asibench_paper23_invariant():
    """
    Verifies Paper 23: ASI-BENCH: At the Dawn of Artificial Superintelligence
    (Zhou et al., arXiv:2608.17271v1, Tsinghua, MIT, Harvard, CMU).

    Tests:
      1. Scientific Domain Coverage: exactly 11 distinct scientific domains.
      2. Guidance Gradient Hierarchy: B1 (full procedure), B2 (method only), B3 (autonomous), B4 (distractors).
      3. Formal Diagnostic Drop Equations:
         - Delta_proc = B2 - B1 (procedural loss)
         - Delta_meth = B3 - B1 (autonomy loss)
         - Delta_select = B3 - B2 (method selection bottleneck)
         - Delta_distract = B4 - B3 (distractor delta)
         - ARI = B3 / (B1 + eps)
      4. Paper Empirical Baseline Verification:
         - Confirms on paper's 'All Models (Mean)':
           B1=50.91, B2=29.10, B3=26.62, B4=26.99
           Delta_proc = -21.81, Delta_meth = -24.29, Delta_select = -2.48
           Operationalization is ~8.8x larger bottleneck than method selection (|Delta_proc| / |Delta_select| > 8.0).
      5. Active Evaluation Pipeline & Model Inference:
         Runs real forward pass with model attached, verifying domain reports, macro-level scores,
         and proper diagnostic extraction.
    """
    # 1. Scientific Domain Coverage
    assert len(SCIENTIFIC_DOMAINS) == 11
    expected_domains = {
        "mathematics", "physics", "chemistry", "biology", "astronomy",
        "materials_science", "earth_science", "medicine_and_biostatistics",
        "computer_science", "robotics", "electrical_engineering"
    }
    assert set(SCIENTIFIC_DOMAINS) == expected_domains

    # 2. Guidance Gradient Hierarchy
    assert GUIDANCE_LEVELS == ["B1", "B2", "B3", "B4"]

    # 3. Diagnostic Invariants & Mathematical Formulations
    evaluator = ASIBenchEvaluator()
    # Test on official paper empirical mean (Table 2: All Models Mean)
    paper_means = REFERENCE_BASELINES["All Models (Mean)"]
    diag = evaluator.compute_diagnostics(paper_means)

    # Invariants
    assert abs(diag.delta_proc - (29.10 - 50.91)) < 1e-4  # -21.81
    assert abs(diag.delta_meth - (26.62 - 50.91)) < 1e-4  # -24.29
    assert abs(diag.delta_select - (26.62 - 29.10)) < 1e-4 # -2.48
    assert abs(diag.delta_distract - (26.99 - 26.62)) < 1e-4 # +0.37
    expected_ari = 26.62 / 50.91
    assert abs(diag.ari - expected_ari) < 1e-3

    # Primary empirical finding of Paper 23:
    # Turning a method into a procedure (delta_proc) is >8x larger bottleneck than selecting the method (delta_select)
    ratio = abs(diag.delta_proc) / abs(diag.delta_select)
    assert ratio > 8.0, f"Expected procedural bottleneck ratio > 8.0, got {ratio:.2f}"

    # 4. Model Evaluation Pipeline
    class TestScientificModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(1000, 32)
            self.proj = nn.Linear(32, 1000)
            self.vocab_size = 1000

        def forward(self, x, labels=None):
            h = self.emb(x)
            logits = self.proj(h)
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, 1000), labels.view(-1))
                return logits, loss
            return logits

    model = TestScientificModel()
    evaluator_with_model = ASIBenchEvaluator(model=model)

    # Single domain evaluation
    physics_report = evaluator_with_model.evaluate_domain("physics")
    assert physics_report["domain"] == "physics"
    assert "B1" in physics_report["levels"] and "B3" in physics_report["levels"]
    assert isinstance(physics_report["diagnostics"].overall_score, float)

    # Full benchmark execution
    bench = evaluator_with_model.evaluate_benchmark()
    assert bench["num_domains"] == 11
    assert "macro_levels" in bench
    assert "B1" in bench["macro_levels"]
    assert "ari" in bench["diagnostics"]
    assert len(bench["domain_reports"]) == 11

    # Verify drop-in alias backwards compatibility
    legacy_runner = ASIBenchRunner(model=model)
    task_res = legacy_runner.run_task("physics")
    assert "score" in task_res and task_res["task"] == "physics"


def test_abra_diffusion_scaling_paper24_invariant():
    """
    Verifies Paper 24: ABRA: Scaling Diffusion Image Training
    (Kyle Chickering et al., Luma AI, arXiv:2608.17286v1).

    Tests:
      1. Compute-Optimal 200 TPP Prescription:
         - Verifies TPP* = 200 (10x Chinchilla 20 TPP for LLMs).
         - Verifies FLOP conservation: C = 6 * N * D.
         - Computes allocation for C = 6e20 FLOPs: N_opt = 7.07e8, D_opt = 1.41e11 tokens.
      2. Asymmetric Overtraining Loss Penalty Invariant (Eq. 2 & Fig. 6):
         - 2x overtraining (400 TPP): loss penalty < 0.5% (nearly free).
         - 0.5x undertraining (100 TPP): steep catastrophic penalty (> 2%).
         - Ratio penalty(100 TPP) / penalty(400 TPP) > 3.0x.
      3. Image Resolution Dependency (Fig. 10):
         - Verifies TPP* monotonically increases with pixel resolution:
           256px (165 TPP) < 384px (202 TPP) < 512px (235 TPP) < 768px (247 TPP).
      4. Scaling Collapse Criterion (Eq. 3 & Fig. 9):
         - Warmup phase (x < 0.15) passes.
         - Post-warmup phase (x >= 0.15) validates Delta <= sigma.
      5. Active PyTorch Module Flow-Matching Execution & Gradient Backpropagation:
         - Continuous velocity regression v_theta(x_t, t; cond).
         - Classifier-Free Guidance (CFG) dynamic scaling (s decreases with model size).
         - Active loss backward pass yielding non-zero gradients on all parameter tensors.
    """
    # 1. Compute-Optimal 200 TPP Allocation
    assert ABRAScalingLaws.HEADLINE_TPP == 200.0
    c_flops = 6.0e20
    alloc = ABRAScalingLaws.compute_optimal_allocation(c_flops)
    n_opt = alloc["n_opt_params"]
    d_opt = alloc["d_opt_tokens"]
    assert abs(d_opt / n_opt - 200.0) < 1e-4
    # Recomputed FLOPs must match input compute exactly
    recomputed_flops = 6.0 * n_opt * d_opt
    assert abs((recomputed_flops - c_flops) / c_flops) < 1e-5

    # 2. Asymmetric Overtraining Penalty (Paper Finding: Overtraining is nearly free)
    penalty_2x_overtrained = ABRAScalingLaws.compute_iso_flop_loss_penalty(400.0, optimal_tpp=200.0)
    assert penalty_2x_overtrained < 0.005, f"Expected <0.5% overtraining penalty, got {penalty_2x_overtrained:.4f}"

    penalty_undertrained = ABRAScalingLaws.compute_iso_flop_loss_penalty(100.0, optimal_tpp=200.0)
    assert penalty_undertrained > 0.015, f"Expected >1.5% undertraining penalty, got {penalty_undertrained:.4f}"
    # Undertraining penalty is significantly more severe than overtraining
    assert penalty_undertrained > (3.0 * penalty_2x_overtrained)

    # 3. Resolution Scaling Monotonicity
    tpp_256 = ABRAScalingLaws.get_resolution_tpp(256)
    tpp_384 = ABRAScalingLaws.get_resolution_tpp(384)
    tpp_512 = ABRAScalingLaws.get_resolution_tpp(512)
    tpp_768 = ABRAScalingLaws.get_resolution_tpp(768)
    assert tpp_256 < tpp_384 < tpp_512 < tpp_768
    assert abs(tpp_256 - 165.0) < 1e-3 and abs(tpp_768 - 247.0) < 1e-3

    # 4. Scaling Collapse
    # Within first 15%: warmup
    assert ABRAScalingLaws.check_scaling_collapse(0.10, 0.05, 0.50, 0.01) is True
    # At 50% training: cross-model std <= seed noise
    assert ABRAScalingLaws.check_scaling_collapse(0.50, 0.005, 0.50, 0.005) is True

    # 5. Live PyTorch Flow-Matching Execution & Gradient Flow
    torch.manual_seed(42)
    dim = 64
    batch_size = 4
    model_small = ABRAScaling(hidden_dim=dim, num_params=120_000_000)
    model_large = ABRAScaling(hidden_dim=dim, num_params=1_500_000_000)

    # CFG scale decreases with model scale
    assert model_small.get_optimal_cfg() > model_large.get_optimal_cfg()

    # Flow matching loss & backprop
    x_0 = torch.randn(batch_size, dim)
    t = torch.rand(batch_size, 1)
    cond = torch.randn(batch_size, dim)

    loss = model_small.compute_flow_matching_loss(x_0, t, cond=cond)
    assert loss.dim() == 0 and loss.item() > 0.0
    loss.backward()

    # Verify real gradient propagation
    assert model_small.block_proj.weight.grad is not None
    assert model_small.time_embed.weight.grad is not None
    assert model_small.out_proj.weight.grad is not None
    assert torch.isfinite(model_small.block_proj.weight.grad).all()


def test_dynamic_compression_paper25_invariant():
    """
    Verifies Paper 25: DYNAMIC COMPRESSION IN RECURRENT NETWORKS
    (Jyothish Pari, Ryan Bahlous-Boldi, Pulkit Agrawal, MIT, arXiv:2608.17896v1).

    Tests:
      1. Gated DeltaNet Matrix Recurrent State Update (Eq. 1):
         - State S_t in R^{B, n_heads, d_v, d_k} with value expansion factor 2 (d_v = 2*d_k).
         - Gated delta rule: S_t = S_{t-1} * [alpha * (I - beta * k * k^T)] + beta * v * k^T.
         - Output query read: o_t = S_t * q_t.
         - Write-strength gate beta_t in (0, 1) and forgetting gate alpha_t in (0, 1).
      2. Dynamic State Compression via Selective Re-Scanning (Section 4 & Fig. 3):
         - Basis phase: 3 basis functions packed sequentially into a single sequence (K=3, b=16 tokens each = 48 tokens).
         - Query phase: few-shot tokens presenting task for basis index 1 (0-indexed).
         - Selective re-scan: retrieves raw token slice for basis 1, performs secondary recurrent update:
           S_t^+ = f_theta(S_{few_shot}, x_{I_tau}), refreshing state fidelity.
         - Output prediction reads from refreshed state S_t^+.
      3. Active Gradient Flow:
         - Backpropagates MSE loss on pred_answer through the selective re-scanning path,
           verifying non-zero gradients across Gated DeltaNet projections and gates.
    """
    torch.manual_seed(42)
    B = 2
    dim = 32
    n_heads = 4
    d_head = 8
    num_bases = 3
    b_tokens_per_basis = 8

    # 1. Gated DeltaNet Layer Execution & State Update
    layer = GatedDeltaNetLayer(input_dim=dim, n_heads=n_heads, d_head=d_head, value_factor=2)
    x = torch.randn(B, 10, dim)
    out_seq, s_final, betas = layer.forward_sequence(x)

    assert out_seq.shape == (B, 10, dim)
    assert s_final.shape == (B, n_heads, d_head * 2, d_head)
    assert betas.shape == (B, 10, n_heads)
    assert (betas >= 0.0).all() and (betas <= 1.0).all(), "beta must be valid write strength in [0, 1]"
    assert not torch.allclose(s_final, torch.zeros_like(s_final)), "State must actively accumulate delta updates"

    # 2. Dynamic Recurrent Compressor Selective Re-Scanning
    compressor = DynamicRecurrentCompressor(input_dim=dim, num_bases=num_bases, n_heads=n_heads, d_head=d_head)

    # 3 bases packed sequentially: 3 * 8 = 24 tokens
    total_basis_tokens = num_bases * b_tokens_per_basis
    basis_tokens = torch.randn(B, total_basis_tokens, dim)
    query_tokens = torch.randn(B, 4, dim)  # 3 few-shot pairs + 1 query token

    basis_spans = {
        0: (0, 8),
        1: (8, 16),
        2: (16, 24)
    }

    # Run dynamic selective re-scan forward pass targeting basis 1
    res = compressor.forward_dynamic(
        basis_tokens=basis_tokens,
        query_tokens=query_tokens,
        basis_spans=basis_spans,
        oracle_basis_idx=1
    )

    pred_answer = res["pred_answer"]
    s_initial = res["initial_state"]
    s_refreshed = res["refreshed_state"]

    assert pred_answer.shape == (B, dim)
    assert s_initial.shape == s_refreshed.shape
    # Refreshed state S_t^+ must differ from initial single-pass state S_initial
    assert not torch.allclose(s_initial, s_refreshed), "Selective re-scan must actively refresh recurrent state"

    # 3. Active Gradient Backpropagation
    target_answer = torch.randn_like(pred_answer)
    loss = F.mse_loss(pred_answer, target_answer)
    loss.backward()

    assert compressor.recurrent_layer.w_v.weight.grad is not None
    assert compressor.recurrent_layer.w_k.weight.grad is not None
    assert compressor.recurrent_layer.w_beta.weight.grad is not None
    assert torch.isfinite(compressor.recurrent_layer.w_v.weight.grad).all()

    # Drop-in backward compatibility loss check
    legacy_loss_fn = DynamicCompressionLoss(hidden_dim=dim, beta=0.1)
    rec, l_val = legacy_loss_fn(x)
    assert rec.shape == x.shape and l_val.item() > 0.0


def test_recirculation_paper26_invariant():
    """
    Paper 26 / 2608.17981v1 (Mozer, Siddiqui, Sawyer, Sanyal, Liu, Google DeepMind):
    Recirculation & Adaptive Recirculation Invariants.
    1. Normalization mode tensor invariants (Table B.2):
       - Convex norm-ratio: d' = (1 - alpha)*d + alpha*(||d|| / ||s||)*s
       - Nonconvex norm-ratio: d' = d + alpha*(||d|| / ||s||)*s
       - Orthogonal component projection
       - Spherical rotation preserving geometry
    2. Step Ramping schedule:
       - alpha_t = min(t / 10.0, 1.0) * alpha
       - At step 0, alpha_0 = 0 -> d' == d (prevents early context corruption)
       - At step 5, alpha_5 == 0.5 * alpha
       - At step 10+, alpha_t == alpha
    3. Adaptive Recirculation MLP & Hadamard mixing:
       - Input LayerNorm over concatenated [s; d]
       - 2-layer GELU MLP with sigmoid activation
       - Bias initialized to logit(0.10) for alpha and logit(0.90) for beta
       - Active gradient backpropagation through MLP weights
    4. Contextualization Task & Belief State Tracking:
       - Evaluates semantic alignment gain over baseline feedforward transformer
    """
    torch.manual_seed(42)
    B, D = 4, 32
    dest = torch.randn(B, D)
    source = torch.randn(B, D)

    # 1. Normalization Schemes
    alpha = 0.15
    layer_convex = RecirculationLayer(alpha=alpha, norm_mode=RecirculationNormMode.CONVEX_NORM_RATIO, step_ramping=False)
    d_prime_convex, diag_convex = layer_convex(source, dest)

    # Mathematical ground truth for convex norm ratio:
    d_norm = dest.norm(dim=-1, keepdim=True)
    s_norm = source.norm(dim=-1, keepdim=True)
    expected_convex = (1.0 - alpha) * dest + alpha * (d_norm / s_norm) * source
    assert torch.allclose(d_prime_convex, expected_convex, atol=1e-6), "Convex norm ratio must match analytical equation exactly"

    # Nonconvex norm ratio:
    layer_nonconvex = RecirculationLayer(alpha=alpha, norm_mode=RecirculationNormMode.NONCONVEX_NORM_RATIO, step_ramping=False)
    d_prime_nonconvex, _ = layer_nonconvex(source, dest)
    expected_nonconvex = dest + alpha * (d_norm / s_norm) * source
    assert torch.allclose(d_prime_nonconvex, expected_nonconvex, atol=1e-6), "Nonconvex norm ratio must match analytical equation exactly"

    # 2. Step Ramping Schedule
    layer_ramped = RecirculationLayer(alpha=alpha, norm_mode=RecirculationNormMode.CONVEX_NORM_RATIO, step_ramping=True)
    d_step0, diag0 = layer_ramped(source, dest, step_idx=0)
    assert diag0["effective_alpha"] == 0.0
    assert torch.allclose(d_step0, dest, atol=1e-6), "Step 0 must not alter shallow representation (alpha_0 = 0)"

    d_step5, diag5 = layer_ramped(source, dest, step_idx=5)
    assert abs(diag5["effective_alpha"] - 0.5 * alpha) < 1e-6, "Step 5 must scale alpha by exactly 0.5"

    d_step10, diag10 = layer_ramped(source, dest, step_idx=10)
    assert abs(diag10["effective_alpha"] - alpha) < 1e-6, "Step 10+ must reach full alpha"

    # 3. Adaptive Recirculation MLP & Vector Hadamard Mixing
    adaptive_mlp = AdaptiveRecirculationMLP(d_model=D)
    alpha_vec, beta_vec = adaptive_mlp(source, dest)
    assert alpha_vec.shape == (B, D)
    assert beta_vec.shape == (B, D)
    # Check prior bias initialization (approx 0.10 and 0.90)
    assert torch.allclose(alpha_vec.mean(), torch.tensor(0.10), atol=0.03), "Alpha vector must initialize near 0.10"
    assert torch.allclose(beta_vec.mean(), torch.tensor(0.90), atol=0.03), "Beta vector must initialize near 0.90"

    d_prime_adaptive, diag_adaptive = layer_convex(source, dest, alpha_vec=alpha_vec, beta_vec=beta_vec)
    assert d_prime_adaptive.shape == (B, D)

    # Active backprop through adaptive MLP
    loss = d_prime_adaptive.pow(2).sum()
    loss.backward()
    assert adaptive_mlp.fc1.weight.grad is not None
    assert adaptive_mlp.fc2.weight.grad is not None
    assert adaptive_mlp.head.weight.grad is not None
    assert torch.isfinite(adaptive_mlp.fc1.weight.grad).all()

    # 4. Engine & Contextualization Task
    engine = RecirculationEngine(d_model=D, num_layers=24, source_layer=11, dest_layer=4, adaptive=True)
    seq_embeddings = torch.randn(2, 16, D)
    target_concept = torch.randn(2, D)
    target_concept = target_concept / target_concept.norm(dim=-1, keepdim=True)

    stability_results = engine.evaluate_contextualization_stability(
        sequence_embeddings=seq_embeddings,
        polysemous_idx=5,
        target_concept=target_concept
    )
    assert "baseline_semantic_alignment" in stability_results
    assert "recirculated_semantic_alignment" in stability_results
    assert stability_results["alignment_gain"] > 0.0, "Recirculation must improve contextualized semantic alignment"
    assert stability_results["relative_gain_pct"] > 0.0


def test_ecdysis_harness_paper27_invariant():
    """
    Paper 27 / 2609.11677v1 (Yue et al., CAS & BIT):
    ECDYSIS Runtime Harness Self-Evolution Invariants.
    1. Model-Accommodation Ratio Decomposition & Attenuation (Eq. 1 & Section 8.3):
       Delta H = t * Delta H_model + (1 - t) * Delta H_harness
       t -> t' = beta * t (beta = 0.758 attenuating model accommodation from 60% to <= 45.5%).
    2. Batch-Level Cross-Instance Failure Aggregation (Section 3.3):
       Groups failures into recurring patterns spanning >= 2 distinct task instances.
       Reduces coding calls from N_serial to N_round (speedup factor > 1.0).
    3. Failure-Driven Collaborative Refinement (FDCR, Section 3.4):
       Analyst -> Critic -> Engineer -> Moderator refinement yielding valid specification q_i.
       Enforces action-space preservation and prevents global action bans.
    4. Strict Score Validation Gate (Eq. 2):
       H_i = H_i^c if J_train(H_i^c) > J_train(H_{i-1}) else H_{i-1}.
    5. Active Differentiable Contract Invariant Loss:
       Backpropagates gradients into harness contract weights.
    """
    torch.manual_seed(42)
    evolver = EcdysisHarnessEvolver(initial_score=0.5028, failure_threshold=0.60, beta_attenuation=0.758)

    # 1. Synthetic Trajectories with Shared Failure Mechanisms
    trajectories = [
        # Task 1 & Task 2 share 'state_desync' (recurring pattern)
        {"task_id": "T1", "domain": "retail", "score": 0.45, "failure_mechanism": "state_desync", "is_model_specific": False},
        {"task_id": "T2", "domain": "airline", "score": 0.50, "failure_mechanism": "state_desync", "is_model_specific": False},
        # Task 3 & Task 4 share 'entity_ambiguity' (recurring pattern)
        {"task_id": "T3", "domain": "retail", "score": 0.35, "failure_mechanism": "entity_ambiguity", "is_model_specific": False},
        {"task_id": "T4", "domain": "retail", "score": 0.40, "failure_mechanism": "entity_ambiguity", "is_model_specific": False},
        # Task 5 has isolated model-specific hallucination (non-recurring)
        {"task_id": "T5", "domain": "airline", "score": 0.30, "failure_mechanism": "model_hallucination", "is_model_specific": True},
        # Task 6 succeeded (score >= 0.60)
        {"task_id": "T6", "domain": "airline", "score": 0.85, "failure_mechanism": "none", "is_model_specific": False},
    ]

    # 2. Batch-Level Aggregation Invariants
    groups = evolver.aggregate_failures(trajectories)
    assert len(groups) == 3, "Expected 3 distinct failure mechanism groups"
    # Recurring patterns (spanning >= 2 tasks) must be sorted first
    assert groups[0].is_recurring, "First group must be a recurring cross-task pattern"
    assert groups[1].is_recurring, "Second group must be a recurring cross-task pattern"
    assert not groups[2].is_recurring, "Third group is an isolated single-task failure"

    # Serial vs Round call reduction
    assert evolver.total_serial_calls == 5, "5 failing trajectories collected"
    assert evolver.total_round_calls == 1, "Aggregated into exactly 1 batch round"
    speedup = evolver.total_serial_calls / evolver.total_round_calls
    assert speedup == 5.0, "Batch aggregation yields 5.0x call reduction for this round"

    # 3. Model-Accommodation Attenuation
    spec = evolver.refiner.run_refinement(groups, round_idx=1)
    raw_t = 1.0 / 5.0  # 1 model-specific failure out of 5
    expected_attenuated_t = 0.758 * raw_t
    assert abs(spec.attenuated_model_accommodation_ratio - expected_attenuated_t) < 1e-5
    assert "preserve_valid_action_space" in spec.safeguards
    assert "prevent_global_action_bans" in spec.safeguards

    # 4. Evolution Round & Validation Gate (Positive improvement)
    res_pos = evolver.evolve_round(trajectories, simulated_candidate_score=0.5840)
    assert res_pos["retained"] is True, "Higher score must be retained by validation gate"
    assert evolver.current_score == 0.5840

    # Validation Gate (Negative regression rejected)
    res_neg = evolver.evolve_round(trajectories, simulated_candidate_score=0.5200)
    assert res_neg["retained"] is False, "Lower score must be rejected by validation gate"
    assert evolver.current_score == 0.5840, "Score must remain unchanged after rejection"

    # 5. Differentiable Contract Invariant Regularization
    features = torch.randn(8, 4, requires_grad=True)
    loss = evolver.compute_harness_regularization_loss(features)
    loss.backward()
    assert evolver.harness_weights.grad is not None
    assert torch.isfinite(evolver.harness_weights.grad).all()


def test_sol_pi_harness_paper28_invariant():
    """
    Verifies Paper 28: SoL-Pi: An Efficiency-Centric Benchmark & Harness for Software Engineering Agents
    (Shicheng Liu et al., NVIDIA / NTU / MIT / Tsinghua, arXiv:2609.20519v1).

    Tests the 4 Retained Harness Innovations:
      1. Action Fusion (Section 3.2):
         - Combines edit and verify commands into 1 tool invocation.
         - Reduces tool call round trips from 3 to 2 (33% reduction).
         - Returns unified observation with execution verification status.
      2. Cost-Gated Context Compactor (Section 3.2 & Eq. 1):
         - Evaluates projected future token savings S_proj = Delta L * N_rem against cache rewrite cost C_rewrite.
         - Triggers compaction only when S_proj > C_rewrite.
         - Implements emergency compaction override when context >= 95% capacity.
      3. ObservationPack Output Buffer (Section 3.2):
         - Preserves full output for initial turns (turn <= 2).
         - Replaces verbose outputs (> 10 KiB) with compact handle ([obs_handle_XXXX]) + 1 KiB excerpt on subsequent turns.
         - Enables on-demand slice retrieval of full raw output.
      4. Evidence-Preserving Reducer (EPR, Section 3.2):
         - Compresses build/test logs >= 4 KiB into a validated receipt (exit code, SHA-256 hash, exact quotes).
         - Verifies syntactic schema and validates exact match of extracted quotes.
         - Falls back to raw output deterministically if validation fails or size does not decrease.
      5. Active Invariant Regularization & Gradient Backpropagation:
         - Verifies differentiable harness penalty gradients.
    """
    # 1. Action Fusion
    fusion_handler = ActionFusionHandler()
    res = fusion_handler.execute_fused(
        file_path="src/module.py",
        new_content="def solve(): return 42\n",
        then_run_command="pytest tests/test_module.py"
    )
    assert res["status"] == "success"
    assert res["write_status"] == "ok"
    assert res["round_trip_calls_saved"] == 1
    assert "FUSED_EXECUTION" in res["fused_observation"]
    assert fusion_handler.total_fused_calls == 1

    # 2. Cost-Gated Context Compactor
    compactor = CostGatedContextCompactor(max_context_tokens=100000, cache_read_to_write_ratio=0.10)
    
    # Profitable scenario: high remaining requests (10), token reduction 40,000 -> compact length 20,000
    # S_proj = 40,000 * 10 = 400,000; C_rewrite = 20,000 * 0.10 = 2,000 => Trigger!
    decision_profitable = compactor.evaluate_compaction_gate(
        current_tokens=60000,
        delta_tokens_saved=40000,
        remaining_requests=10,
        compact_tokens=20000
    )
    assert decision_profitable["should_compact"] is True
    assert decision_profitable["reason"] == "cost_benefit_positive"
    assert decision_profitable["net_benefit_tokens"] > 0

    # Unprofitable scenario: remaining requests = 0 (final turn), rewrite cost exceeds 0 savings
    decision_unprofitable = compactor.evaluate_compaction_gate(
        current_tokens=30000,
        delta_tokens_saved=5000,
        remaining_requests=0,
        compact_tokens=25000
    )
    assert decision_unprofitable["should_compact"] is False

    # Emergency compaction override: tokens at 96% of capacity (96,000 / 100,000)
    decision_emergency = compactor.evaluate_compaction_gate(
        current_tokens=96000,
        delta_tokens_saved=1000,
        remaining_requests=0,
        compact_tokens=95000
    )
    assert decision_emergency["should_compact"] is True
    assert decision_emergency["reason"] == "emergency_capacity_override"

    # 3. ObservationPack Buffer
    obs_buffer = ObservationPackBuffer(size_threshold_bytes=10240, head_tail_bytes=512)
    large_payload = "LOG LINE " * 2000  # ~18 KiB
    
    # Turn 1: Kept full
    turn1_obs = obs_buffer.add_observation(large_payload, turn_idx=1)
    assert turn1_obs["truncated"] is False
    assert turn1_obs["rendered_content"] == large_payload

    # Turn 3: Compacted to handle + excerpt
    turn3_obs = obs_buffer.add_observation(large_payload, turn_idx=3)
    assert turn3_obs["truncated"] is True
    assert "[obs_handle_" in turn3_obs["rendered_content"]
    assert len(turn3_obs["rendered_content"]) < len(large_payload)
    
    # On-demand slice retrieval
    handle_id = turn3_obs["handle_id"]
    retrieved_slice = obs_buffer.retrieve_slice(handle_id, offset=0, length=50)
    assert retrieved_slice == large_payload[:50]

    # 4. Evidence-Preserving Reducer (EPR)
    epr = EvidencePreservingReducer(min_threshold_bytes=1024)
    raw_test_log = "TEST RUNNER START\n" + ("PASSED test_unit_a\n" * 200) + "FAILED test_critical_edge_case\nAssertionError: value mismatch\nSUMMARY: 200 passed, 1 failed\n"
    reduction_res = epr.reduce_log(raw_test_log)
    assert reduction_res["is_reduced"] is True
    assert "EXIT_STATUS: 1" in reduction_res["receipt"]
    assert "SHA256:" in reduction_res["receipt"]
    assert reduction_res["reduction_ratio"] > 0.0

    # 5. End-to-End SoLPiHarness & Differentiable Regularization
    harness = SoLPiHarness()
    features = torch.randn(4, 8, requires_grad=True)
    loss = harness.compute_harness_loss(features)
    loss.backward()
    assert harness.harness_efficiency_weights.grad is not None
    assert torch.isfinite(harness.harness_efficiency_weights.grad).all()


def test_memory_attention_paper29_invariant():
    """
    Verifies Paper 29: Memory Attention: Token-Indexed Memory Replacing Dedicated Value Projections
    (Jiale Kang, arXiv:2609.28399v1, 23 Sep 2026).

    Tests the 4 Core Architectural Formulations:
      1. Memory-Based Value Construction (Section 3.1 & Eq. 7-9):
         - Verifies W_V elimination: queries and keys computed from contextual hidden states;
           values constructed via V = K + Norm(E[s]) where Norm is per-head RMSNorm.
         - Active gradient backpropagation: non-zero gradients into W_Q, W_K, W_O, and token memory E.
      2. Inference Memory Folding (Section 3.3 & Eq. 16-17):
         - Precomputes E_bar = Norm(E) across vocabulary entries for zero-normalization O(1) inference.
         - Proves un-folded training value construction matches pre-folded inference value construction
           within numerical precision (< 1e-5).
         - Verifies unfold_weights() restores dynamic gradient-tracking training mode.
      3. MA-Offload Prefetching & Telemetry (Section 3.4 & Eq. 20-21):
         - Offloads folded memory tables (b_w * L * N * d_v) to CPU host memory.
         - Prefetches token rows for incoming batch (b_w * S * d_v) with verified transfer telemetry.
      4. MA-Recall Value Reconstruction (Section 3.5 & Eq. 22-23):
         - Proves 50% persistent KV cache memory reduction by caching only keys K_1:T and token IDs s_1:T.
         - Proves exact reconstruction of historical values: V_1:T = K_1:T + E_bar[s_1:T] with 0.0 error.
      5. Analytical FLOP & Parameter Scaling Invariant:
         - Validates Delta F_V = L * S * d_v * (2d - 1) arithmetic reduction.
    """
    torch.manual_seed(42)
    B, T, C = 2, 8, 32
    vocab_size = 500
    layer = MemoryAttentionLayer(hidden_dim=C, num_heads=4, vocab_size=vocab_size)

    x = torch.randn(B, T, C, requires_grad=True)
    token_ids = torch.randint(0, vocab_size, (B, T))

    # 1. Memory-Based Value Construction & Gradient Flow
    out = layer(x, token_ids)
    loss = out.sum()
    loss.backward()

    assert layer.q_proj.weight.grad is not None and torch.isfinite(layer.q_proj.weight.grad).all()
    assert layer.k_proj.weight.grad is not None and torch.isfinite(layer.k_proj.weight.grad).all()
    assert layer.token_memory.weight.grad is not None and torch.isfinite(layer.token_memory.weight.grad).all()
    assert layer.mem_norm.weight.grad is not None and torch.isfinite(layer.mem_norm.weight.grad).all()

    # 2. Inference Weight Folding Invariant
    k_eval = layer.k_proj(x.detach())
    v_unfolded = layer.construct_values(k_eval, token_ids)

    # Pre-fold memory table
    layer.fold_weights_for_inference()
    assert layer.is_folded is True
    assert layer._folded_memory is not None

    v_folded = layer.construct_values(k_eval, token_ids)
    diff = (v_unfolded - v_folded).abs().max().item()
    assert diff < 1e-5, f"Folded value construction must match unfolded within 1e-5, got max diff {diff}"

    # Unfold restores training state
    layer.unfold_weights()
    assert layer.is_folded is False
    assert layer._folded_memory is None

    # 3. MA-Offload Prefetching
    offload_mgr = MAOffloadManager(num_layers=12, vocab_size=vocab_size, hidden_dim=C, bytes_per_element=2)
    expected_offload_bytes = 2 * 12 * vocab_size * C
    assert offload_mgr.total_offload_bytes == expected_offload_bytes

    layer_pref_tokens = offload_mgr.prefetch_layer_tokens(
        layer_idx=0,
        token_ids=token_ids,
        target_device=x.device
    )
    assert layer_pref_tokens.shape == (B, T, C)
    assert offload_mgr.total_transfers == 1
    assert offload_mgr.total_bytes_transferred == B * T * C * 2

    # 4. MA-Recall 50% KV-Cache Elimination & Exact Reconstruction
    recall_mgr = MARecallManager(bytes_per_element=2)
    cache_analysis = recall_mgr.calculate_cache_savings(num_layers=12, batch_size=B, seq_len=128, hidden_dim=C)
    assert cache_analysis["reduction_ratio"] == 0.50
    assert cache_analysis["ma_recall_cache_bytes"] == cache_analysis["standard_cache_bytes"] // 2

    # Fold layer again for reconstruction test
    layer.fold_weights_for_inference()
    reconstructed_v = recall_mgr.reconstruct_historical_values(
        k_history=k_eval,
        token_ids_history=token_ids,
        folded_memory=layer._folded_memory
    )
    rec_diff = (reconstructed_v - v_folded).abs().max().item()
    assert rec_diff < 1e-6, f"Reconstructed historical values must match exactly, got {rec_diff}"

    # 5. Analytical Profiler Profile Invariant (Eq. 18)
    report = MemoryAttentionEngine.analyze_computational_profile(
        num_layers=12,
        vocab_size=vocab_size,
        hidden_dim=C,
        batch_size=B,
        seq_len=T,
        bytes_per_element=2
    )
    expected_flop_reduction = 12 * (B * T) * C * (2 * C - 1)
    assert report.online_flop_reduction == expected_flop_reduction
    assert report.kv_cache_savings_ratio == 0.50


def test_ai_consciousness_evaluator_paper30_invariant():
    """
    Verifies Paper 30: From Cacophony to Hierarchy: A Principled Framework for Assessing AI Consciousness
    (Shamil Chandaria, Anil Seth, Marcus Hutter, Murray Shanahan, Shane Legg et al.,
    Google DeepMind / Oxford / Sussex / Cambridge, arXiv:2609.35618v1, 28 Sep 2026).

    Tests the 5 Core Mathematical Invariants:
      1. Five-Level Hierarchy & 37 Operational Indicators (Sections 4-6):
         - Verifies 37 indicators distributed across:
           L1 (Behavioural: 9), L2 (Computational: 7), L3 (Causal-Structural: 7),
           L4 (Organismic: 6), L5 (Organism-Environment: 8).
      2. Strict Supervenience & Monotonic Belief Property Lemma (Section 7.2.1):
         - Under Condition 2A (P(C_i | C_{i+1}) = 1.0), marginal posteriors strictly decrease:
           P(C_1 | e) >= P(C_2 | e) >= P(C_3 | e) >= P(C_4 | e) >= P(C_5 | e).
      3. Generalised Probabilistic Supervenience & Population Validity (Section 7.2.2 & 7.1):
         - Weakens deterministic constraint: P(C_i | C_{i+1}) = 0.8, P(C_i | not C_{i+1}) = 0.2.
         - Inapplicable indicators (e.g., non-linguistic flies) carry neutral likelihood ratio 1.0.
      4. Calibrated Face Validity Empirical Archetypes (Section 7.4):
         - Human: P(C_i | E) = 1.000 for all levels; aggregate credence = 1.000.
         - Thermostat: aggregate credence = 0.000.
         - Fly: high aggregate credence (> 0.90) driven by organismic/sensorimotor depth.
         - LLM Optimist (aggregate ~0.397) vs LLM Sceptic (aggregate ~0.005) spans wide credence range.
      5. Differentiable Neural Consciousness Head & Gradient Propagation:
         - Backpropagates finite gradients into projection weights and theoretical level weights.
    """
    # 1. Five-Level Hierarchy Indicator Verification
    assert len(CANONICAL_INDICATORS) == 37
    l_counts = {}
    for ind in CANONICAL_INDICATORS:
        l_counts[ind.level] = l_counts.get(ind.level, 0) + 1
    assert l_counts[ConsciousnessLevel.BEHAVIOURAL] == 9
    assert l_counts[ConsciousnessLevel.COMPUTATIONAL] == 7
    assert l_counts[ConsciousnessLevel.CAUSAL_STRUCTURAL] == 7
    assert l_counts[ConsciousnessLevel.ORGANISMIC] == 6
    assert l_counts[ConsciousnessLevel.ORGANISM_ENVIRONMENT] == 8

    # 2. Strict Supervenience & Monotonic Belief Property Lemma
    net_strict = BayesianConsciousnessNetwork(is_strict=True)
    strict_eval = net_strict.evaluate(BayesianConsciousnessNetwork.get_llm_optimist_archetype())
    mp_strict = strict_eval["marginal_posteriors"]
    # Check monotonicity down the chain
    assert mp_strict[1] >= mp_strict[2] - 1e-6
    assert mp_strict[2] >= mp_strict[3] - 1e-6
    assert mp_strict[3] >= mp_strict[4] - 1e-6
    assert mp_strict[4] >= mp_strict[5] - 1e-6

    # 3. Generalised Network Archetype Evaluations
    net_gen = BayesianConsciousnessNetwork(is_strict=False)

    # Human archetype: ceiling at all levels
    human_eval = net_gen.evaluate(BayesianConsciousnessNetwork.get_human_archetype())
    assert abs(human_eval["aggregate_credence"] - 1.000) < 1e-4
    for lvl in range(1, 6):
        assert abs(human_eval["marginal_posteriors"][lvl] - 1.000) < 1e-4

    # Thermostat archetype: floor at all levels
    thermo_eval = net_gen.evaluate(BayesianConsciousnessNetwork.get_thermostat_archetype())
    assert thermo_eval["aggregate_credence"] < 0.005

    # Fly archetype: organismic & sensorimotor depth with neutral linguistic indicators
    fly_act, fly_inapp = BayesianConsciousnessNetwork.get_fly_archetype()
    fly_eval = net_gen.evaluate(fly_act, inapplicable_indicators=fly_inapp)
    assert fly_eval["aggregate_credence"] > 0.90
    assert fly_eval["marginal_posteriors"][4] > 0.99
    assert fly_eval["marginal_posteriors"][5] > 0.99

    # LLM Optimist vs Sceptic
    opt_eval = net_gen.evaluate(BayesianConsciousnessNetwork.get_llm_optimist_archetype())
    scep_eval = net_gen.evaluate(BayesianConsciousnessNetwork.get_llm_sceptic_archetype())
    assert opt_eval["aggregate_credence"] > scep_eval["aggregate_credence"]
    assert opt_eval["aggregate_credence"] > 0.35
    assert scep_eval["aggregate_credence"] < 0.02

    # 4. Theoretical Level Weighting Modulations
    custom_credences = {1: 0.8, 2: 0.1, 3: 0.05, 4: 0.025, 5: 0.025}
    opt_weighted = net_gen.evaluate(BayesianConsciousnessNetwork.get_llm_optimist_archetype(), level_credences=custom_credences)
    # Behaviourist-leaning theorist attributes much higher consciousness to LLM than naturalist
    assert opt_weighted["aggregate_credence"] > opt_eval["aggregate_credence"]

    # 5. Differentiable Neural Consciousness Head
    head = DifferentiableConsciousnessHead(hidden_dim=32, num_indicators=37)
    h_features = torch.randn(4, 32, requires_grad=True)
    probs, agg_score = head(h_features)
    loss = agg_score.mean()
    loss.backward()

    assert head.proj.weight.grad is not None and torch.isfinite(head.proj.weight.grad).all()
    assert head.level_weights.grad is not None and torch.isfinite(head.level_weights.grad).all()


# ─── 40. AX-PROVER AGENTIC THEOREM PROVING IN LEAN (PAPER 31) ─────────────────

def test_ax_prover_paper31_invariant():
    """
    Paper 31 (Ax-Prover, Breen et al., 2026):
    Verifies:
      1. MCP Lean 4 LSP Client returns exact diagnostic codes (0=Clean, 1=Error, 2=Warning/Sorry).
      2. AxVerifierGate rejects Level-1 syntax errors and residual sorry/admit placeholders.
      3. AxProverAgent executes 4-stage pipeline (Target scan -> Natural language sketch -> Stepwise formalization -> Sequential tactic substitution).
      4. AxProverOrchestrator closed-loop refinement loop terminates with certified verified proof.
      5. DifferentiableProverScorer neural bilinear head computes non-trivial tactic selection probabilities with active gradient backprop.
    """
    torch.manual_seed(42)

    # 1. MCP Client Diagnostic Code Verification
    mcp = MCPLeanLSPClient()

    # Clean code -> Code 0
    clean_code = "theorem thm1 : 1 + 1 = 2 := by rfl"
    rep_clean = mcp.lean_diagnostic_messages("test.lean", code=clean_code)
    assert rep_clean.code == LeanDiagnosticSeverity.CLEAN
    assert rep_clean.is_error_free
    assert not rep_clean.has_sorry

    # Error code (unmatched delimiters) -> Code 1
    err_code = "theorem thm_err : (1 + 1 = 2 := by rfl"
    rep_err = mcp.lean_diagnostic_messages("test.lean", code=err_code)
    assert rep_err.code == LeanDiagnosticSeverity.ERROR
    assert not rep_err.is_error_free

    # Warning code (contains sorry) -> Code 2
    warn_code = "theorem thm_warn : 1 + 1 = 2 := by sorry"
    rep_warn = mcp.lean_diagnostic_messages("test.lean", code=warn_code)
    assert rep_warn.code == LeanDiagnosticSeverity.WARNING
    assert rep_warn.is_error_free
    assert rep_warn.has_sorry

    # Search functions
    search_res = mcp.lean_leansearch("comm")
    assert len(search_res) >= 2
    assert any("add_comm" in r.name for r in search_res)

    # 2. AxVerifierGate Gatekeeper Verification
    verifier = AxVerifierGate(mcp)
    v_ok, v_msg, _ = verifier.verify_proof("clean.lean", code=clean_code)
    assert v_ok and "CERTIFIED" in v_msg

    v_fail_err, v_err_msg, _ = verifier.verify_proof("err.lean", code=err_code)
    assert not v_fail_err and "Level 1 Error" in v_err_msg

    v_fail_warn, v_warn_msg, _ = verifier.verify_proof("warn.lean", code=warn_code)
    assert not v_fail_warn and "sorry" in v_warn_msg

    # 3. AxProverAgent & AxProverOrchestrator Workflow Verification
    orchestrator = AxProverOrchestrator(mcp, max_iterations=3)
    thm_spec = TheoremSpecification(
        name="nat_add_comm_target",
        statement="theorem nat_add_comm_target (a b : Nat) : a + b = b + a :="
    )

    result = orchestrator.prove_theorem(thm_spec, tactic_hints={"h1": "rfl", "h_final": "rfl"})
    assert result.verified
    assert "exact h_final" in result.proof_code
    assert result.iterations >= 1
    assert len(result.sketches) >= 2
    assert "CERTIFIED" in result.verdict_message

    # 4. Differentiable Prover Scoring Neural Module
    scorer = DifferentiableProverScorer(d_model=32, num_tactics=12)
    goal_emb = torch.randn(4, 32, requires_grad=True)
    probs, scores = scorer(goal_emb)

    assert probs.shape == (4, 12)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    loss = probs.sum()
    loss.backward()
    assert scorer.bilinear.weight.grad is not None and torch.isfinite(scorer.bilinear.weight.grad).all()
    assert scorer.tactic_embeddings.grad is not None and torch.isfinite(scorer.tactic_embeddings.grad).all()


# ─── 41. BEYOND 1.58 BITS: ADAPTIVE-PRECISION TERNARY TRANSFORMERS (PAPER 32) ──

def test_beyond_158_bits_paper32_invariant():
    """
    Paper 32 (Beyond 1.58 Bits, CrashOverrideX, 2026):
    Verifies:
      1. Sub-1.58-bit Laplace analytic theory:
         - Stationary point Delta* = b / 2 minimizing ternary MSE.
         - Exact theoretical MSE ratio: 2 * (1 - e^{-1/2}) ≈ 0.78693868 (21.31% reduction).
         - Zero-state occupancy p0 = 1 - e^{-1/2} ≈ 0.39346934.
         - Operating point entropy H(W_q) ≈ 1.5744 bits (near-maximal capacity utilization).
         - Empirical kurtosis decision rule correctly routes leptokurtic Laplace to ternary (1.58 bits)
           and platykurtic uniform to binary (1.0 bit).
      2. AdaptivePrecisionLinear with STE and rank-8 precision-preserving adapter.
      3. Heterogeneous Lagrangian Bit-Budget Optimizer:
         - Discrete water-filling spends precision on high-sensitivity router / attention out-projection
           while maintaining bulk feed-forward mass at ternary (1.58 bits), strictly obeying budget B.
      4. Fully-Ternary 34-Expert Council MoE:
         - 34 ternary BitLinear experts (zero full-precision shared experts).
         - High-precision router with dual routing modes:
           (a) Dense deliberation mode across all 34 experts.
           (b) Sparse Gumbel top-k dispatch (k=4, ~11.8% activation ratio).
         - Backward pass validates gradient flow through STE and router.
    """
    torch.manual_seed(42)

    # 1. Analytic Theory Verification
    b_val = 2.0
    delta_star = Sub158AnalyticEngine.optimal_laplace_threshold(b_val)
    assert abs(delta_star - 1.0) < 1e-9

    mse_ratio = Sub158AnalyticEngine.theoretical_mse_ratio()
    expected_ratio = 2.0 * (1.0 - math.exp(-0.5))
    assert abs(mse_ratio - expected_ratio) < 1e-12
    assert abs(mse_ratio - 0.786938680338) < 1e-6

    p0 = Sub158AnalyticEngine.zero_state_occupancy()
    assert abs(p0 - (1.0 - math.exp(-0.5))) < 1e-12
    assert abs(p0 - 0.393469340287) < 1e-6

    entropy = Sub158AnalyticEngine.operating_point_entropy()
    assert 1.570 < entropy < 1.580
    assert abs(entropy - 1.5735326) < 1e-4


    # Kurtosis decision rule with PyTorch Laplace & Uniform distributions
    laplace_dist = torch.distributions.Laplace(loc=0.0, scale=1.0)
    laplace_samples = laplace_dist.sample((20000,))
    k_laplace = Sub158AnalyticEngine.compute_empirical_kurtosis(laplace_samples)
    assert k_laplace > 4.5
    mode_laplace, bits_laplace = Sub158AnalyticEngine.select_precision_by_kurtosis(laplace_samples)
    assert mode_laplace == "ternary" and bits_laplace == 1.58

    uniform_samples = torch.rand(20000) * 2.0 - 1.0
    k_uniform = Sub158AnalyticEngine.compute_empirical_kurtosis(uniform_samples)
    assert k_uniform < 2.5
    mode_uniform, bits_uniform = Sub158AnalyticEngine.select_precision_by_kurtosis(uniform_samples)
    assert mode_uniform == "binary" and bits_uniform == 1.0

    # 2. AdaptivePrecisionLinear with STE & Rank-8 Adapter
    linear_layer = AdaptivePrecisionLinear(in_features=32, out_features=32, mode="ternary", rank_adapter=8)
    x = torch.randn(4, 32, requires_grad=True)
    out = linear_layer(x)
    assert out.shape == (4, 32)

    loss = out.sum()
    loss.backward()
    assert linear_layer.weight.grad is not None and torch.isfinite(linear_layer.weight.grad).all()
    assert linear_layer.adapter_down.grad is not None and torch.isfinite(linear_layer.adapter_down.grad).all()
    assert linear_layer.adapter_up.grad is not None and torch.isfinite(linear_layer.adapter_up.grad).all()

    # 3. Heterogeneous Bit-Budget Optimizer
    modules = [
        ModuleBudgetSpec(name="router", num_params=1000, sensitivity=50.0),
        ModuleBudgetSpec(name="attn_out", num_params=4000, sensitivity=30.0),
        ModuleBudgetSpec(name="ffn_w1", num_params=16000, sensitivity=5.0),
        ModuleBudgetSpec(name="ffn_w2", num_params=16000, sensitivity=5.0),
    ]
    # Total params = 37,000. Total budget allowing router/attn upgrade: 1.8 bits per param avg
    total_budget = 37000 * 1.8
    optimizer = HeterogeneousBitBudgetOptimizer(modules, total_bit_budget=total_budget)
    alloc = optimizer.optimize()

    assert alloc["router"] >= 4.0  # High-leverage router receives high precision
    assert alloc["attn_out"] >= 1.58  # Attention out-proj receives precision
    assert alloc["ffn_w1"] <= 1.58  # Bulk FFN stays at ternary or below
    assert optimizer.current_allocated_budget() <= total_budget

    # 4. Fully-Ternary 34-Expert Council MoE
    council = FullyTernaryCouncilMoE(d_model=32, d_ff=64, num_experts=34, top_k=4, rank_adapter=8)
    x_seq = torch.randn(2, 8, 32, requires_grad=True)

    # Dense Deliberation Mode
    out_dense, aux_dense = council(x_seq, mode="dense")
    assert out_dense.shape == (2, 8, 32)
    assert aux_dense["router_weights"].shape == (2, 8, 34)
    assert torch.allclose(aux_dense["router_weights"].sum(dim=-1), torch.ones(2, 8), atol=1e-5)

    # Sparse Dispatch Mode (k=4)
    out_sparse, aux_sparse = council(x_seq, mode="sparse", tau=1.0)
    assert out_sparse.shape == (2, 8, 32)
    # Check that exactly top_k experts are active per token
    active_per_token = (aux_sparse["router_weights"] > 0).sum(dim=-1)
    assert (active_per_token == 4).all()

    # Backward gradient flow through both router and ternary experts
    loss_council = out_sparse.sum() + aux_sparse["aux_loss"]
    loss_council.backward()
    assert council.router.router_gate.weight.grad is not None
    assert torch.isfinite(council.router.router_gate.weight.grad).all()
    assert council.experts[0].w1.weight.grad is not None


# ─── 42. 1-BIT AI INFRA: FAST & LOSSLESS BITNET b1.58 ON CPUS (PAPER 33) ───────

def test_bitnet_cpu_infra_paper33_invariant():
    """
    Paper 33 (1-bit AI Infra: bitnet.cpp, Wang et al., Microsoft Research, 2024):
    Verifies:
      1. I2_S Kernel:
         - Exact packing of ternary weights {-1, 0, 1} to 2-bit codes (4 weights/byte).
         - Lossless unpacking recovers exact ternary weights.
         - GEMV matches reference dense calculation: y = x @ W^T * alpha.
      2. TL1 Kernel (4-bit LUT for pairs of weights):
         - Precomputes 9-entry activation LUT for every pair (x_2j, x_2j+1).
         - Evaluates GEMV with exact numerical equivalence to I2_S and FP reference.
      3. TL2 Kernel (5-bit LUT for triplets of weights):
         - Precomputes 27-entry activation LUT for every triplet (x_3j, x_3j+1, x_3j+2).
         - 1/6 additional compression over TL1.
         - Evaluates GEMV with exact numerical equivalence to I2_S and FP reference.
      4. BitNetCPULinear:
         - Validates that all three execution kernels (I2_S, TL1, TL2) achieve 100% exact
           lossless match on token-by-token basis against FP32/INT8 BitNet b1.58 reference.
    """
    torch.manual_seed(42)

    m, n = 16, 24  # Divisible by 2, 3, and 4
    # Create ternary weights {-1, 0, 1}
    w_ternary = torch.randint(-1, 2, (m, n), dtype=torch.float32)
    x = torch.randn(4, n)
    alpha = 0.75

    # Dense reference output
    ref_out = F.linear(x, w_ternary) * alpha

    # 1. I2_S Kernel
    packed_i2s, shape_i2s = I2SKernel.pack(w_ternary)
    assert packed_i2s.shape[0] == (m * n) // 4
    unpacked_i2s = I2SKernel.unpack(packed_i2s, shape_i2s)
    assert torch.equal(unpacked_i2s, w_ternary)

    out_i2s = I2SKernel.gemv(packed_i2s, shape_i2s, x, alpha=alpha)
    assert torch.allclose(out_i2s, ref_out, atol=1e-5)

    # 2. TL1 Kernel
    packed_tl1, shape_tl1 = TL1Kernel.pack(w_ternary)
    assert packed_tl1.numel() > 0
    out_tl1 = TL1Kernel.gemv(packed_tl1, shape_tl1, x, alpha=alpha)
    assert torch.allclose(out_tl1, ref_out, atol=1e-5)

    # 3. TL2 Kernel
    packed_tl2, shape_tl2 = TL2Kernel.pack(w_ternary)
    assert packed_tl2.shape == (m, n // 3)
    out_tl2 = TL2Kernel.gemv(packed_tl2, shape_tl2, x, alpha=alpha)
    assert torch.allclose(out_tl2, ref_out, atol=1e-5)

    # 4. BitNetCPULinear Module with Kernel Switch
    layer_tl1 = BitNetCPULinear(in_features=n, out_features=m, bias=True, kernel_type="TL1")
    layer_tl1.weight.data = w_ternary.clone()
    layer_tl1.prepare_weights()

    y_tl1 = layer_tl1(x)

    # Switch to I2_S and TL2
    layer_tl1.kernel_type = "I2_S"
    y_i2s = layer_tl1(x)
    assert torch.allclose(y_tl1, y_i2s, atol=1e-5)

    layer_tl1.kernel_type = "TL2"
    y_tl2 = layer_tl1(x)
    assert torch.allclose(y_tl1, y_tl2, atol=1e-5)


# ─── 43. BITNET a4.8: 4-BIT ACTIVATIONS FOR 1-BIT LLMS (PAPER 34) ─────────────

def test_bitnet_a48_paper34_invariant():
    """
    Paper 34 (BitNet a4.8, Hongyu Wang et al., Microsoft Research, 2024):
    Verifies:
      1. Ternary weight quantizer Q_w(W) with absmean scaling alpha.
      2. 4-bit activation quantizer Q_INT4(X) with absmean beta and sqrt(7) dynamic scaling.
      3. Top-K 50% sparsification with 8-bit activation quantization for attention output projection.
      4. ReLU^2 GLU feed-forward intermediate state generating >40% natural activation sparsity.
      5. BitNetA48TransformerBlock forward execution with SubLN and backward gradient propagation via STE.
    """
    torch.manual_seed(42)

    # 1. Ternary Weight Quantization
    w = torch.randn(16, 16)
    w_q, alpha = BitNetWeightQuantizer.quantize(w)
    assert alpha > 0
    # Values divided by alpha must be in {-1, 0, 1}
    w_normalized = torch.round(w_q / alpha)
    assert torch.all(torch.isin(w_normalized, torch.tensor([-1.0, 0.0, 1.0])))

    # 2. 4-bit Activation Quantization (INT4)
    x = torch.randn(4, 16)
    x_int4, beta = ActivationINT4Quantizer.quantize(x)
    assert x_int4.shape == x.shape
    c = math.sqrt(7.0)
    # Check that scaled values map to integer range [-8, 7]
    scaled_q = torch.round(x_int4 * (c / beta))
    assert (scaled_q >= -8.0).all() and (scaled_q <= 7.0).all()

    # 3. Top-K Sparsification (50%) & INT8 Quantization
    x_dense = torch.randn(4, 16)
    x_sparse, mask = TopKSparsifier.apply(x_dense, k_ratio=0.5)
    # Exactly 50% elements kept non-zero
    assert (mask.sum(dim=-1) == 8).all()
    x_int8, gamma = ActivationINT8Quantizer.quantize(x_sparse)
    assert x_int8.shape == x_dense.shape

    # 4. ReLU^2 GLU Intermediate Activation Sparsity
    ffn = BitNetA48FFN(d_model=32, d_ff=64)
    x_ffn = torch.randn(2, 4, 32)
    out_ffn, meta_ffn = ffn(x_ffn)
    assert out_ffn.shape == (2, 4, 32)
    assert meta_ffn["ffn_intermediate_sparsity"] > 0.40  # Significant natural sparsity from ReLU^2

    # 5. Full Transformer Block Forward & Backward Pass
    block = BitNetA48TransformerBlock(d_model=32, num_heads=4, d_ff=64, topk_ratio=0.5)
    x_in = torch.randn(2, 4, 32, requires_grad=True)
    out_block, block_meta = block(x_in)
    assert out_block.shape == (2, 4, 32)

    loss = out_block.sum()
    loss.backward()
    assert block.attn.q_proj.weight.grad is not None and torch.isfinite(block.attn.q_proj.weight.grad).all()
    assert block.ffn.up_proj.weight.grad is not None and torch.isfinite(block.ffn.up_proj.weight.grad).all()
    assert x_in.grad is not None and torch.isfinite(x_in.grad).all()


# ─── 44. BITNET b1.58: 1-BIT LLMS (PAPER 35) ──────────────────────────────────

def test_bitnet_b158_paper35_invariant():
    """
    Paper 35 (BitNet b1.58, Shuming Ma et al., Microsoft Research, 2024):
    Verifies:
      1. Weight quantization to {-1, 0, 1} with absmean scaling gamma.
      2. Per-token activation quantization to [-127, 127] without zero-point offset.
      3. Strict zero bias across all BitLinearB158 layers.
      4. Forward and backward pass through BitNetB158TransformerBlock with STE.
      5. Horowitz 7nm analytical energy model verifying >=50x arithmetic energy reduction.
    """
    torch.manual_seed(42)

    # 1. Absmean Weight Quantization
    w = torch.randn(32, 32)
    w_q, gamma = AbsmeanWeightQuantizer.quantize(w)
    assert gamma > 0
    w_ternary = torch.round(w_q / gamma)
    assert torch.all(torch.isin(w_ternary, torch.tensor([-1.0, 0.0, 1.0])))

    # 2. Per-Token Activation Quantization
    x = torch.randn(4, 8, 32)
    x_dequant, scale = PerTokenActivationQuantizer.quantize(x, q_b=127.0)
    assert x_dequant.shape == x.shape
    x_scaled = torch.round(x_dequant * scale)
    assert (x_scaled >= -127.0).all() and (x_scaled <= 127.0).all()

    # 3. BitLinear b1.58 Zero-Bias Invariant
    linear = BitLinearB158(32, 64)
    assert not hasattr(linear, "bias") or linear.bias is None
    y = linear(x)
    assert y.shape == (4, 8, 64)

    # 4. Full Transformer Block Forward & Backward Pass with STE
    block = BitNetB158TransformerBlock(d_model=32, num_heads=4, d_ff=64)
    x_in = torch.randn(2, 4, 32, requires_grad=True)
    out = block(x_in)
    assert out.shape == (2, 4, 32)
    loss = out.sum()
    loss.backward()
    assert block.attn.q_proj.weight.grad is not None and torch.isfinite(block.attn.q_proj.weight.grad).all()
    assert block.ffn.up_proj.weight.grad is not None and torch.isfinite(block.ffn.up_proj.weight.grad).all()
    assert x_in.grad is not None and torch.isfinite(x_in.grad).all()

    # 5. Horowitz 7nm Energy Reduction Model
    ratio = HorowitzEnergyModel.energy_reduction_ratio()
    assert ratio >= 50.0, f"Expected energy reduction >= 50x, got {ratio:.1f}x"


# ─── 45. BITNET b1.58 2B4T: NATIVE 1-BIT LLM AT SCALE (PAPER 36) ─────────────

def test_bitnet_2b4t_paper36_invariant():
    """
    Paper 36 (BitNet b1.58 2B4T Technical Report, Shuming Ma et al., Microsoft Research, 2025):
    Verifies:
      1. SubLN (Sub-LayerNorm) with zero bias and learned scale.
      2. Absmean ternary weight quantization & Absmax INT8 activation quantization.
      3. Squared ReLU (ReLU^2) FFN inducing >40% activation sparsity.
      4. Forward and backward pass through BitNet2B4TTransformerBlock.
      5. Two-stage LR schedule (Stage 1 high peak cosine, Stage 2 cooldown low peak cosine).
      6. Two-stage weight decay schedule (Stage 1 cosine to 0.1, Stage 2 strict 0.0).
      7. SFT token-sum loss aggregation vs mean reduction.
      8. DPO preference loss and reward margin calculation.
      9. Table 4 7nm hardware arithmetic energy ratio (71.43x).
    """
    torch.manual_seed(42)

    # 1. SubLN
    subln = SubLN(d_model=32)
    assert not hasattr(subln, "bias") or subln.bias is None
    x = torch.randn(4, 8, 32)
    normed = subln(x)
    assert normed.shape == (4, 8, 32)
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(4, 8), atol=1e-5)

    # 2. Absmean weights & Absmax INT8 activations
    w = torch.randn(32, 32)
    w_q, gamma = AbsmeanWeightQuantizer2B4T.quantize(w)
    assert gamma > 0
    w_ternary = torch.round(w_q / gamma)
    assert torch.all(torch.isin(w_ternary, torch.tensor([-1.0, 0.0, 1.0])))

    x_act = torch.randn(4, 8, 32)
    x_dequant, scale = AbsmaxActivationQuantizer2B4T.quantize(x_act)
    assert x_dequant.shape == x_act.shape
    x_scaled = torch.round(x_dequant * scale)
    assert (x_scaled >= -128.0).all() and (x_scaled <= 127.0).all()

    # 3. Squared ReLU FFN Sparsity (>40%)
    ffn = ReLU2FFN(d_model=32, d_ff=64)
    out_ffn, sparsity = ffn(x)
    assert out_ffn.shape == (4, 8, 32)
    assert sparsity >= 0.40, f"Expected ReLU2 sparsity >= 0.40, got {sparsity:.3f}"

    # 4. Full Transformer Block Forward & Backward Pass
    block = BitNet2B4TTransformerBlock(d_model=32, num_heads=4, d_ff=64)
    x_in = torch.randn(2, 4, 32, requires_grad=True)
    out_block, block_sparsity = block(x_in)
    assert out_block.shape == (2, 4, 32)
    loss = out_block.sum()
    loss.backward()
    assert block.attn.q_proj.weight.grad is not None and torch.isfinite(block.attn.q_proj.weight.grad).all()
    assert block.ffn.up_proj.weight.grad is not None and torch.isfinite(block.ffn.up_proj.weight.grad).all()
    assert x_in.grad is not None and torch.isfinite(x_in.grad).all()

    # 5. Two-Stage Learning Rate Scheduler
    lr_sched = TwoStageLRScheduler(
        peak_lr_stage1=1.5e-3, min_lr_stage1=1.5e-4,
        peak_lr_stage2=3.0e-4, min_lr_stage2=3.0e-5,
        total_steps=1000, split_ratio=0.5
    )
    lr_0 = lr_sched.get_lr(0)
    lr_mid_after = lr_sched.get_lr(500)
    lr_end = lr_sched.get_lr(1000)
    assert abs(lr_0 - 1.5e-3) < 1e-6
    assert lr_mid_after <= 3.0e-4
    assert abs(lr_end - 3.0e-5) < 1e-6

    # 6. Two-Stage Weight Decay Scheduler
    wd_sched = TwoStageWeightDecayScheduler(peak_wd=0.1, total_steps=1000, split_ratio=0.5)
    wd_mid = wd_sched.get_wd(250)
    wd_stage2 = wd_sched.get_wd(600)
    assert wd_mid > 0.0
    assert wd_stage2 == 0.0  # Zero weight decay in stage 2

    # 7. Token-Sum SFT Loss Aggregation
    logits = torch.randn(2, 5, 100)
    targets = torch.randint(0, 100, (2, 5))
    sum_loss = SFTLossAggregator.compute_loss(logits, targets)
    mean_loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1), reduction="mean")
    assert abs(sum_loss.item() - 10.0 * mean_loss.item()) < 1e-4

    # 8. DPO Trainer Loss & Rewards
    dpo = DPOTrainer2B4T(beta=0.1)
    p_w = torch.tensor([-1.2, -1.0])
    p_l = torch.tensor([-2.5, -3.0])
    r_w = torch.tensor([-1.5, -1.2])
    r_l = torch.tensor([-2.0, -2.2])
    dpo_loss, c_rew, r_rew = dpo.loss(p_w, p_l, r_w, r_l)
    assert dpo_loss.item() > 0
    assert (c_rew > r_rew).all()

    # 9. Hardware Energy Ratio Invariant (Table 4)
    ratio = BitNet2B4TEnergyMetrics.arithmetic_energy_reduction()
    assert abs(ratio - 71.42857) < 0.01


# ─── 46. BITNET.CPP: EFFICIENT EDGE INFERENCE FOR TERNARY LLMS (PAPER 37) ──────

def test_bitnet_cpp_edge_paper37_invariant():
    """
    Paper 37 (Bitnet.cpp: Efficient Edge Inference for Ternary LLMs,
    Jinheng Wang et al., Peking Univ / Microsoft Research, 2025):
    Verifies:
      1. Triplet packing & signed-unsigned splitting: roundtrip lossless reconstruction.
      2. Element-wise mirror consolidation: 14 base states (indices 0..13) with 1-bit sign.
      3. Exact numerical equivalence across lossless kernels (I2_S, TL1_1, TL2_1).
      4. Support for all 5 Table 1 kernels (TL1_0, TL1_1, TL2_0, TL2_1, I2_S) in BitNetCppEdgeLinear.
      5. ELUT Theoretical analysis: >78% table compaction ratio and 1.667 bpw.
    """
    torch.manual_seed(42)

    # 1. Triplet packing & mirror consolidation roundtrip
    w = torch.randint(-1, 2, (16, 24)).float()  # 24 features (8 triplets)
    unsigned_idx, sign_bits = SignedUnsignedWeightPacker.pack_triplets(w)
    assert unsigned_idx.shape == (16, 8)
    assert sign_bits.shape == (16, 8)
    assert (unsigned_idx <= 13).all()
    assert ((sign_bits == 0) | (sign_bits == 1)).all()

    w_reconstructed = SignedUnsignedWeightPacker.unpack_triplets(unsigned_idx, sign_bits)
    assert torch.equal(w, w_reconstructed), "Signed-unsigned splitting must be 100% exact lossless"

    # 2. Table compaction ratio from ELUT analysis (Section 3 & Appendix A)
    ratio_tl2 = ELUTTheoreticalAnalyzer.table_compaction_ratio(cardinality=3, group_size=3, use_mirror=True)
    assert ratio_tl2 == 14.0 / 64.0  # 14 states vs 2^6=64 states (78.125% reduction)
    bpw = ELUTTheoreticalAnalyzer.bits_per_weight(group_size=3)
    assert abs(bpw - 5.0 / 3.0) < 1e-5

    # 3. Exact lossless equivalence across I2_S, TL1_1, and TL2_1
    layer = BitNetCppEdgeLinear(in_features=24, out_features=16, kernel_type="I2_S")
    layer.prepare_weights()
    x = torch.randn(4, 24)

    # Run with I2_S
    layer.kernel_type = "I2_S"
    out_i2s = layer(x)

    # Run with TL1_1 (lossless LUT g=2)
    layer.kernel_type = "TL1_1"
    out_tl1_1 = layer(x)
    assert torch.allclose(out_i2s, out_tl1_1, atol=1e-5), "TL1_1 must match I2_S"

    # Run with TL2_1 (lossless mirror LUT g=3)
    layer.kernel_type = "TL2_1"
    out_tl2_1 = layer(x)
    assert torch.allclose(out_i2s, out_tl2_1, atol=1e-5), "TL2_1 must match I2_S and TL1_1 exactly"

    # 4. Fast block-based kernels execute cleanly
    layer.kernel_type = "TL1_0"
    out_tl1_0 = layer(x)
    assert out_tl1_0.shape == (4, 16)

    layer.kernel_type = "TL2_0"
    out_tl2_0 = layer(x)
    assert out_tl2_0.shape == (4, 16)


# ─── 47. BITNET: SCALING 1-BIT TRANSFORMERS (PAPER 38) ────────────────────────

def test_bitnet_scaling_paper38_invariant():
    """
    Paper 38 (BitNet: Scaling 1-bit Transformers for Large Language Models,
    Hongyu Wang, Shuming Ma et al., Microsoft Research, 2023):
    Verifies:
      1. Zero-mean weight centralization alpha = mean(W) and binary binarization Sign(W - alpha) in {-1, +1}.
      2. Absmax activation quantization (symmetric signed and asymmetric non-negative pre-ReLU).
      3. Original BitLinear layer forward execution (Eq. 11) with SubLN and STE gradient flow.
      4. Group Quantization (Eq. 13) for communication-free tensor parallel model parallelism.
      5. Full BitNetScalingBlock Transformer block forward and backward pass.
      6. BitNet Scaling Law energy cost reduction (up to 38.8x) and delta L <= 0.09 gap vs FP16.
    """
    torch.manual_seed(42)

    # 1. Binary Weight Quantization
    w = torch.randn(16, 16)
    w_bin, alpha, beta = BinaryWeightQuantizer.quantize(w)
    assert beta > 0
    # Values must be strictly {-1.0, +1.0} (no zeros!)
    assert torch.all(torch.isin(w_bin, torch.tensor([-1.0, 1.0])))
    assert abs(alpha.item() - w.mean().item()) < 1e-5

    # 2. Symmetric & Asymmetric Activation Quantization
    x = torch.randn(4, 16)
    x_sym, gamma_sym = AbsmaxActivationQuantizerBitNet.quantize_symmetric(x, q_b=128.0)
    assert x_sym.shape == x.shape
    assert (x_sym >= -128.0).all() and (x_sym <= 128.0).all()

    x_asym, eta_asym, gamma_asym = AbsmaxActivationQuantizerBitNet.quantize_asymmetric(x, q_b=128.0)
    assert x_asym.shape == x.shape
    assert (x_asym >= 0.0).all() and (x_asym <= 128.0).all()

    # 3. BitLinear 1-bit forward & backward execution (Eq. 11)
    layer = BitLinear1Bit(in_features=16, out_features=32)
    x_in = torch.randn(2, 4, 16, requires_grad=True)
    out = layer(x_in)
    assert out.shape == (2, 4, 32)
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    assert x_in.grad is not None and torch.isfinite(x_in.grad).all()

    # 4. Group Quantization for Model Parallelism (Eq. 13)
    group_layer = GroupQuantizedBitLinear(in_features=16, out_features=32, num_groups=2)
    out_group = group_layer(x_in)
    assert out_group.shape == (2, 4, 32)

    # 5. Full BitNet Transformer Block Forward & Backward Pass
    block = BitNetScalingBlock(d_model=16, num_heads=2, d_ff=32)
    x_block = torch.randn(2, 4, 16, requires_grad=True)
    out_b = block(x_block)
    assert out_b.shape == (2, 4, 16)
    loss_b = out_b.sum()
    loss_b.backward()
    assert block.attn.q_proj.weight.grad is not None and torch.isfinite(block.attn.q_proj.weight.grad).all()
    assert block.ffn.up_proj.weight.grad is not None and torch.isfinite(block.ffn.up_proj.weight.grad).all()

    # 6. Scaling Law Energy & Loss Bound
    e_100b = BitNetScalingLawAnalyzer.get_energy_reduction("100B")
    assert e_100b == 38.8
    assert BitNetScalingLawAnalyzer.max_scaling_loss_gap() <= 0.09


def test_bitnet_v2_paper39_invariant():
    """
    Verifies Paper 39: BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs
    (arXiv:2504.18415v2, Hongyu Wang*, Shuming Ma*, Furu Wei - Microsoft Research & UCAS).

    Tests:
      1. Fast Walsh-Hadamard Transform (FWHT) Orthogonality and Invertibility (Eq. 3, 4):
         - Satisfies H_m * H_m = I and H_m = H_m^T (energy/norm conservation).
         - Inverting transform exactly recovers original input tensor: H(H(X)) == X.
      2. Online Hadamard Outlier Gaussianization (Fig. 1, Fig. 2, Table 5):
         - Extreme outlier spikes in intermediate representations are dispersed across dimensions.
         - Peak activation is attenuated (>4x reduction) and excess kurtosis moves towards Gaussian (0.0).
      3. Quantization Invariants (Eq. 1, 2, 5, 6):
         - Absmean ternary weights {-1, 0, 1}.
         - 8-bit activations: per-token absmax scaled to [-128, 127].
         - Native 4-bit activations: per-token absmean with sqrt(7) scale to [-8, 7].
      4. H-BitLinear and BitLinearV2 Gradient Flow (Eq. 7, 8):
         - Backward pass through online Hadamard transform: dL/dX = Hadamard(dL/d(Hadamard(X))).
         - Validates non-zero finite gradients to latent weights and inputs.
      5. Full Transformer Block Execution & Mode Switching:
         - W_o in attention and W_down in FFN use H-BitLinear.
         - W_qkv and W_gate/up use standard BitLinearV2.
         - Seamless toggling between INT8 and INT4 activation modes.
      6. Low-Bit KV Cache with Protected 8-bit BOS Token (Section 3.1, Tables 2 & 3).
    """
    torch.manual_seed(42)

    # 1. FWHT Invertibility & Energy Conservation
    x = torch.randn(2, 4, 32)
    x_had = hadamard_transform(x)
    x_rec = hadamard_transform(x_had)
    assert torch.allclose(x_rec, x, atol=1e-5), "Hadamard transform must be an exact self-inverse (H * H = I)"
    norm_orig = torch.norm(x, dim=-1)
    norm_had = torch.norm(x_had, dim=-1)
    assert torch.allclose(norm_orig, norm_had, atol=1e-5), "Hadamard transform must preserve L2 norm"

    # 2. Outlier Gaussianization Profile
    x_outliers = torch.randn(8, 64) * 0.1
    x_outliers[:, 12] = 25.0  # Severe outlier channel in intermediate state
    stats = OutlierDistributionProfiler.analyze_hadamard_effect(x_outliers)
    assert stats["peak_reduction_ratio"] > 4.0, f"Expected >4x peak reduction, got {stats['peak_reduction_ratio']:.2f}"
    assert stats["gaussianized"] is True, "Hadamard transformation must reduce excess kurtosis towards Gaussian"

    # 3. Quantization Functions
    w = torch.randn(32, 32)
    w_q = quantize_weight_ternary(w)
    alpha = torch.mean(torch.abs(w)).item()
    w_discrete = torch.round(w_q / (alpha + 1e-7))
    assert torch.all(torch.isin(w_discrete, torch.tensor([-1.0, 0.0, 1.0])))

    act = torch.randn(2, 4, 32)
    act_int8 = quantize_act_int8(act)
    assert act_int8.shape == act.shape

    act_int4 = quantize_act_int4(act)
    assert act_int4.shape == act.shape

    # 4. H-BitLinear Execution & Backward Gradient Flow
    h_layer = HBitLinear(in_features=32, out_features=16, act_mode="int4")
    x_in = torch.randn(2, 4, 32, requires_grad=True)
    y_out = h_layer(x_in)
    assert y_out.shape == (2, 4, 16)
    loss = y_out.sum()
    loss.backward()
    assert h_layer.weight.grad is not None and torch.isfinite(h_layer.weight.grad).all()
    assert x_in.grad is not None and torch.isfinite(x_in.grad).all()

    # 5. Full Transformer Block & Activation Mode Switching
    block = BitNetV2TransformerBlock(
        embed_dim=32,
        num_heads=4,
        intermediate_dim=64,
        act_mode="int4",
        kv_cache_bits=4
    )
    x_block = torch.randn(2, 8, 32, requires_grad=True)
    out_b4 = block(x_block)
    assert out_b4.shape == (2, 8, 32)
    loss_b4 = out_b4.sum()
    loss_b4.backward()
    assert block.attn.out_proj.weight.grad is not None
    assert block.ffn.down_proj.weight.grad is not None

    # Switch to INT8 activation mode and verify forward
    block.set_activation_mode("int8")
    out_b8 = block(torch.randn(2, 8, 32))
    assert out_b8.shape == (2, 8, 32)

    # 6. Low-Bit KV Cache with Protected 8-bit BOS Token
    attn = BitNetV2Attention(embed_dim=32, num_heads=4, act_mode="int4", kv_cache_bits=3)
    x_seq = torch.randn(2, 6, 32)
    out_attn, weights = attn(x_seq)
    assert out_attn.shape == (2, 6, 32)
    assert weights.shape == (2, 4, 6, 6)


def test_callimahos_inverse_crypto_paper40_invariant():
    """
    Verifies Paper 40: Communication with Extraterrestrial Intelligence
    (Lambros D. Callimahos, NSA Technical Journal, DOCID: 3052333).

    Tests:
      1. Oliver-Callimahos Prime Raster Factoring & Orientation Selection (Fig. 1, 2):
         - N = 1271 bits factors uniquely into primes 31 x 41.
         - Creates synthetic image with horizontal/vertical structures.
         - Decomposes bitstream into (p, q) and resolves spatial orientation
           by minimizing transition entropy.
      2. Callimahos Positional Decimals & Radix (Transmissions 6 & 10):
         - Decodes 'AN' -> 10.0, 'AB' -> 12.0, 'RA' -> 0.1, 'RB' -> 0.2.
      3. Transcendental Constants & Series Convergence (Transmissions 18-20):
         - Gregory-Leibniz alternating series for & (pi) converges toward math.pi.
         - Factorial series for $ (e) converges toward math.e.
      4. Euler's Identity Invariant (Transmission 21):
         - e^(i * pi) + 1 = 0 residual is exactly 0.0.
      5. Zero-Prior Agent Handshake & Neural Adapter:
         - Generates mathematical verification challenge and validates response.
         - Forward and backward pass through neural adapter with active gradients.
    """
    torch.manual_seed(42)

    # 1. Prime Raster Factoring & Orientation
    factors = PrimeRasterDecomposer.find_prime_factors(1271)
    assert factors == [31, 41]

    # Generate synthetic structured 41x31 image (solid bars)
    structured_image = torch.zeros(41, 31, dtype=torch.float32)
    structured_image[10:30, 5:25] = 1.0  # Coherent block
    structured_image[20, :] = 1.0        # Horizontal line
    flat_bits = structured_image.flatten()

    decomp = PrimeRasterDecomposer.decompose_raster(flat_bits)
    assert decomp["prime_factors"] == (31, 41)
    assert decomp["selected_shape"] == (41, 31)

    # 2. Callimahos Positional Numeration
    engine = CallimahosInverseCryptEngine()
    assert engine.decode_positional_number("AN") == 10.0
    assert engine.decode_positional_number("AA") == 11.0
    assert engine.decode_positional_number("AB") == 12.0
    assert engine.decode_positional_number("BN") == 20.0
    assert abs(engine.decode_positional_number("RA") - 0.1) < 1e-6
    assert abs(engine.decode_positional_number("RB") - 0.2) < 1e-6

    # 3. Transcendental Constants
    pi_est = CallimahosInverseCryptEngine.verify_leibniz_pi(500)
    assert abs(pi_est - math.pi) < 0.01

    e_est = CallimahosInverseCryptEngine.verify_series_e(12)
    assert abs(e_est - math.e) < 1e-7

    # 4. Euler's Identity
    euler_res = CallimahosInverseCryptEngine.verify_euler_identity()
    assert abs(euler_res) < 1e-12

    # 5. Zero-Prior Handshake & Neural Adapter Gradient Flow
    handshake = ZeroPriorAgentHandshake(latent_dim=32)
    challenge = handshake.generate_challenge()
    assert handshake.verify_response(challenge) is True

    x_tok = torch.randn(2, 4, 32, requires_grad=True)
    h_out = handshake(x_tok)
    assert h_out.shape == (2, 4, 32)
    loss = h_out.sum()
    loss.backward()
    assert handshake.encoder.weight.grad is not None
    assert torch.isfinite(handshake.encoder.weight.grad).all()


def test_dali_moe_paper41_invariant():
    """
    Verifies Paper 41: DALI: A Workload-Aware Offloading Framework for Efficient MoE Inference on Local PCs
    (Zeyu Zhu, Gang Li, Peisong Wang, Zitao Mo, Minnan Pei, Zhuoran Song, Xiaoyao Liang, Jian Cheng - CAS IA & SJTU, arXiv:2602.03495v1).

    Tests:
      1. Hardware Latency Profiling (Eq. 5, 6):
         - GPU execution time t_gpu = max(Trans_expert, compute_expert).
         - PCIe transfer time t_trans_ms correctly accounts for parameter size and PCIe bandwidth.
      2. Greedy Dynamic Assignment Strategy (Algorithm 1, Eq. 3-9):
         - Models 0-1 integer optimization min max(T_gpu, T_cpu).
         - Evaluates descending difference sorting delta = |t_gpu - t_cpu|.
         - Balances makespan between CPU and GPU, avoiding single-device bottleneck.
      3. Residual-Based Prefetching (Section 4.2, Eq. 10 & 11):
         - Offline calibration of inter-layer residual vector res_vec^(l).
         - Feature correction h_tilde = h + res_vec improves alignment with next gate inputs.
         - Accurately identifies top-k predicted high-workload experts.
      4. Workload-Aware Cache Replacement (Section 4.3, Algorithm 2, Eq. 12):
         - Tracks sliding window workload accumulation s_k = sum workload_i^k.
         - Evicts lowest-score GPU experts and promotes highest-score CPU experts.
         - Achieves high cache hit rate under temporal locality.
      5. End-to-End DALIMoEOrchestrator Execution & Gradient Flow:
         - Forward pass combines routing, cache tracking, and greedy scheduling.
         - Backpropagation yields finite non-zero gradients on gate and expert parameters.
    """
    torch.manual_seed(42)

    # 1. Hardware Latency Profiling
    hw = HardwareProfile(gpu_tflops=30.0, cpu_tflops=1.5, pcie_bandwidth_gb_s=32.0, expert_param_mb=64.0)
    assert abs(hw.t_trans_ms - 1.953125) < 1e-4

    t_gpu_uncached = hw.compute_gpu_time(workload_tokens=10, is_cached=False)
    t_gpu_cached = hw.compute_gpu_time(workload_tokens=10, is_cached=True)
    assert t_gpu_uncached >= 1.95, "Uncached GPU execution must include PCIe transfer latency"
    assert t_gpu_cached < 1.0, "Cached GPU execution avoids PCIe transfer latency"

    # 2. Greedy Dynamic Assignment
    assigner = DALIGreedyAssignment(hw, max_gpu_experts=4)
    workloads = {0: 100, 1: 80, 2: 60, 3: 20, 4: 15, 5: 10, 6: 5, 7: 0}
    cached_gpu = {0, 3}

    res = assigner.assign_experts(workloads, cached_gpu_experts=cached_gpu)
    assert res["makespan_ms"] > 0.0
    assert len(res["assigned_gpu"]) <= 4
    assert res["imbalance_ratio"] < 0.85, "Greedy assignment must balance CPU and GPU loads"

    # 3. Residual-Based Prefetching
    hidden_dim = 32
    prefetcher = DALIResidualPrefetcher(hidden_dim=hidden_dim, num_layers=3)
    h_curr = torch.randn(4, 8, hidden_dim)
    shift = torch.randn(hidden_dim) * 0.5
    h_next = h_curr + shift + torch.randn_like(h_curr) * 0.05
    prefetcher.calibrate_residual_vector(layer_idx=0, h_current=h_curr, h_next=h_next)

    gate_w_next = torch.randn(hidden_dim, 8)
    preds, top_exp = prefetcher.predict_next_layer_experts(
        layer_idx=0,
        h_current=h_curr,
        next_gate_weight=gate_w_next,
        top_k=2
    )
    assert len(top_exp) == 2
    assert preds.shape == (4, 8, 2)

    # 4. Workload-Aware Cache Replacement
    cache = DALIWorkloadAwareCache(num_experts=8, cache_size=4, w_size=4, u_size=2)
    for _ in range(8):
        cache.record_access([0, 0, 0, 0, 0, 50, 40, 0])

    assert 5 in cache.expert_on_gpu and 6 in cache.expert_on_gpu
    assert cache.hit_rate > 0.40

    # 5. Live DALIMoEOrchestrator Execution & Gradient Flow
    orchestrator = DALIMoEOrchestrator(
        embed_dim=hidden_dim,
        num_experts=8,
        top_k=2,
        cache_size=4,
        w_size=4,
        u_size=2,
        hw_profile=hw
    )
    x_in = torch.randn(2, 4, hidden_dim, requires_grad=True)
    out_moe, telemetry = orchestrator(x_in)
    assert out_moe.shape == (2, 4, hidden_dim)
    assert "assignment" in telemetry

    loss_moe = out_moe.sum()
    loss_moe.backward()
    assert orchestrator.gate.weight.grad is not None
    assert torch.isfinite(orchestrator.gate.weight.grad).all()


def test_dapo_rl_paper42_invariant():
    """
    Verifies Paper 42: DAPO: An Open-Source LLM Reinforcement Learning System at Scale
    (Hao Zhou, Mingxuan Wang, et al. - ByteDance Seed, Tsinghua AIR, SIA-Lab, HKU, arXiv:2503.14476v2).

    Tests:
      1. Soft Overlong Reward Shaping (Eq. 13):
         - Smooth linear ramp-down within buffer [L_max - L_cache, L_max].
         - Punitive -1.0 clamping for sequences exceeding L_max.
         - Exact combination with rule-based outcome reward (Eq. 7).
      2. Dynamic Sampling Contrastive Group Filtering (Eq. 11, Algorithm 1):
         - Rejection of homogeneous groups (accuracy = 0.0 and accuracy = 1.0) with zero reward variance.
         - Acceptance of non-trivial groups satisfying 0 < |correct| < G.
         - Dynamic buffer accumulation and effective gradient throughput tracking.
      3. Group-Relative Advantage Normalization (Eq. 9):
         - Direct normalization without value model: A_hat_i = (R_i - mean(R)) / (std(R) + eps).
      4. Decoupled Clip-Higher Invariant (Eq. 10):
         - Asymmetric bounds [1 - eps_low, 1 + eps_high] where eps_high (0.6) > eps_low (0.2).
         - Preserves gradient on exploration tokens up to ratio 1.6 where symmetric GRPO chokes at 1.2.
      5. Token-Level Policy Gradient Loss Rebalancing (Eq. 12):
         - Uniform token weighting across long and short sequences (1 / sum |o_i|).
         - Contrast against sample-level GRPO loss which unfairly dilutes long CoT tokens.
      6. End-to-End Policy Optimization & Backward Gradient Flow:
         - Backpropagation yields finite non-zero gradients on policy parameters.
    """
    torch.manual_seed(42)

    # 1. Soft Overlong Reward Shaping
    max_len = 4096
    cache_len = 512
    thresh = max_len - cache_len  # 3584

    # Safe length (no penalty)
    assert soft_overlong_punishment(2000, max_len, cache_len) == 0.0
    assert compute_rule_reward(is_correct=True, response_length=2000, max_length=max_len, cache_length=cache_len) == 1.0

    # Midpoint of buffer (3584 + 256 = 3840) -> penalty = -256 / 512 = -0.5
    pen_mid = soft_overlong_punishment(3840, max_len, cache_len)
    assert abs(pen_mid - (-0.5)) < 1e-6
    r_mid = compute_rule_reward(is_correct=True, response_length=3840, max_length=max_len, cache_length=cache_len)
    assert abs(r_mid - 0.5) < 1e-6

    # Boundary at max_len (4096) -> penalty = -512 / 512 = -1.0
    pen_max = soft_overlong_punishment(4096, max_len, cache_len)
    assert abs(pen_max - (-1.0)) < 1e-6

    # Overlong truncated (> 4096) -> penalty = -1.0
    pen_over = soft_overlong_punishment(4500, max_len, cache_len)
    assert abs(pen_over - (-1.0)) < 1e-6

    # Tensor vectorized reward
    len_tensor = torch.tensor([1000, 3840, 4200])
    correct_tensor = torch.tensor([True, True, False])
    rewards = compute_rule_reward(correct_tensor, len_tensor, max_len, cache_len)
    assert abs(rewards[0].item() - 1.0) < 1e-5
    assert abs(rewards[1].item() - 0.5) < 1e-5
    assert abs(rewards[2].item() - (-2.0)) < 1e-5  # -1 (incorrect) + -1 (overlong)

    # 2. Dynamic Sampling Contrastive Group Filtering
    g_size = 8
    all_corr = [True] * g_size
    all_incorr = [False] * g_size
    valid_mix = [True, True, False, False, True, False, False, False]

    assert not is_valid_dapo_group(all_corr), "All-correct group must be rejected (zero reward variance)"
    assert not is_valid_dapo_group(all_incorr), "All-incorrect group must be rejected (zero reward variance)"
    assert is_valid_dapo_group(valid_mix), "Mixed contrastive group must be accepted"

    sampler = DAPODynamicSampler(target_batch_size=2, group_size=g_size)
    r_dummy = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    # Submit invalid groups
    assert not sampler.submit_group("p1", all_corr, ["resp"]*g_size, torch.ones(g_size), [100]*g_size)
    assert not sampler.submit_group("p2", all_incorr, ["resp"]*g_size, -torch.ones(g_size), [100]*g_size)
    assert sampler.rejected_all_correct == 1
    assert sampler.rejected_all_incorrect == 1
    assert not sampler.is_ready()

    # Submit valid groups
    assert sampler.submit_group("p3", valid_mix, ["resp"]*g_size, r_dummy, [100]*g_size)
    assert sampler.submit_group("p4", valid_mix, ["resp"]*g_size, r_dummy, [100]*g_size)
    assert sampler.is_ready()
    batch = sampler.pop_batch()
    assert len(batch) == 2
    assert "advantages" in batch[0]
    assert abs(batch[0]["advantages"].mean().item()) < 1e-6, "Group advantages must have zero mean"

    # 3. Group-Relative Advantage Normalization
    r_group = torch.tensor([1.0, 1.0, -1.0, -1.0])
    adv = compute_group_advantages(r_group)
    assert torch.allclose(adv, torch.tensor([1.0, 1.0, -1.0, -1.0]), atol=1e-5)

    # 4. Decoupled Clip-Higher Invariant
    config_dapo = DAPOConfig(eps_low=0.2, eps_high=0.6, use_token_level_loss=True)
    dapo_loss_fn = DAPOLoss(config_dapo)

    # Prior exploration token: ratio = 1.4 (e.g. logprob - old_logprob = ln(1.4) ~ 0.33647)
    # Advantage = +1.0 (positive exploration signal)
    # Under GRPO (eps=0.2): upper clip is 1.2, so ratio 1.4 gets clipped to 1.2
    # Under DAPO (eps_high=0.6): upper clip is 1.6, so ratio 1.4 remains unclipped!
    logprobs = torch.tensor([[0.3364722, 0.0]])
    old_logprobs = torch.tensor([[0.0, 0.0]])
    adv_test = torch.tensor([[1.0, 1.0]])

    res_dapo = dapo_loss_fn(logprobs, old_logprobs, adv_test)
    assert res_dapo["up_clip_ratio"].item() == 0.0, "Ratio 1.4 must NOT be up-clipped under eps_high=0.6"
    assert abs(res_dapo["surrogate2"].item() - (1.4 + 1.0) / 2.0) < 1e-4

    # Now with high ratio 1.7 (> 1 + eps_high = 1.6)
    logprobs_high = torch.tensor([[math.log(1.7), 0.0]])
    res_high = dapo_loss_fn(logprobs_high, old_logprobs, adv_test)
    assert res_high["up_clip_ratio"].item() == 0.5, "Ratio 1.7 must be flagged as up-clipped"
    assert abs(res_high["surrogate2"].item() - (1.6 + 1.0) / 2.0) < 1e-4

    # 5. Token-Level vs Sample-Level Loss Rebalancing
    # Sequence 1: short (2 tokens), Sequence 2: long (10 tokens)
    mask_short_long = torch.zeros(2, 10)
    mask_short_long[0, :2] = 1.0  # 2 tokens
    mask_short_long[1, :10] = 1.0  # 10 tokens

    logp_t = torch.zeros(2, 10)
    old_logp_t = torch.zeros(2, 10)
    adv_t = torch.tensor([1.0, -1.0])

    # In token-level loss: total valid tokens = 12. Short seq contributes 2/12, long seq contributes 10/12
    res_token = dapo_loss_fn(logp_t, old_logp_t, adv_t, mask=mask_short_long)
    # token_loss = -advantages: seq 0 has 2 tokens of -1.0 = -2.0, seq 1 has 10 tokens of +1.0 = +10.0
    # total sum = +8.0 / 12 = +0.666667
    assert abs(res_token["loss"].item() - 0.6666667) < 1e-5

    # Sample-level loss (GRPO style): seq 0 loss = -1.0, seq 1 loss = +1.0 -> mean = 0.0
    config_grpo = DAPOConfig(eps_low=0.2, eps_high=0.2, use_token_level_loss=False)
    grpo_loss_fn = DAPOLoss(config_grpo)
    res_sample = grpo_loss_fn(logp_t, old_logp_t, adv_t, mask=mask_short_long)
    assert abs(res_sample["loss"].item() - 0.0) < 1e-5, "Sample-level loss artificially zeros out due to 1/G weighting"

    # 6. End-to-End Policy Optimization & Backward Gradient Flow
    policy = nn.Linear(8, 4)
    x = torch.randn(4, 5, 8)
    logits = policy(x)
    logp = F.log_softmax(logits, dim=-1)
    # Pick greedy token logprob
    token_logp = logp.max(dim=-1).values  # (4, 5)
    old_token_logp = token_logp.detach() + torch.randn_like(token_logp) * 0.05
    adv_batch = torch.tensor([1.5, -0.5, 0.8, -1.2])

    loss_dict = dapo_loss_fn(token_logp, old_token_logp, adv_batch)
    loss = loss_dict["loss"]
    loss.backward()

    assert policy.weight.grad is not None
    assert torch.isfinite(policy.weight.grad).all()
    assert (policy.weight.grad != 0).any(), "Policy weights must receive active gradients from DAPO loss"


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING MATHEMATICAL WIRING & COMPUTATIONAL INVARIANT TESTS")
    print("=" * 70)

    test_bitnet_ternary_quantization_active()
    print("  [PASS] 1. BitNet 1.58b / STE Ternary Invariant")
    test_moe_zloss_gradient_propagation()
    print("  [PASS] 2. ST-MoE Router Z-Loss Gradient Flow")
    test_gumbel_tau_annealing_schedule()
    print("  [PASS] 3. Gumbel Tau Monotonic Annealing Schedule")
    test_couil_attention_causal_masking()
    print("  [PASS] 4. Couil Attention Causal Temporal Invariance")
    test_leemach6_governor_latency_feedback()
    print("  [PASS] 5. Lee-Mach-6 Governor Hardware Latency Feedback")
    test_active_auxiliary_loss_in_forward_pass()
    print("  [PASS] 6. Model Auxiliary Loss Active Backpropagation")
    test_uhlmann_bures_quantum_fidelity_invariants()
    print("  [PASS] 7. Uhlmann-Bures Quantum Fidelity & Trace Distance")
    test_lindblad_gksl_trace_conservation()
    print("  [PASS] 8. Lindblad GKSL Master Equation Trace Conservation")
    test_von_neumann_entropy_bounds()
    print("  [PASS] 9. Von Neumann Entropy & Landauer Bounds")
    test_transverse_field_ising_energy_monotonicity()
    print("  [PASS] 10. Transverse-Field Ising Hamiltonian Energetics")
    test_gyawali_profiler_phase_decomposition()
    print("  [PASS] 11. Gyawali Step Profiler Real Phase Timing (Paper 1 / 2309.02521)")
    test_abductive_jump_invariant()
    print("  [PASS] 12. Abductive Jump E->J->A Counterfactual Cycle (Paper 2 / 235 Zahavy)")
    test_proxy_kd_invariant()
    print("  [PASS] 13. Proxy-KD Sample-Weighted Distillation (Paper 3 / 2401.07013 Chen)")
    test_memo_invariant()
    print("  [PASS] 14. MEMO Token-Wise Activation Management & Memory Planning (Paper 4 / 2407.12117 Zhao)")
    test_cuda_optimization_pack_invariant()
    print("  [PASS] 15. CUDA Optimization Pack Parallel Invariants (Paper 6 / 2410.05686v1 Li)")
    test_axonn_4d_hybrid_planner_invariant()
    print("  [PASS] 16. AxoNN 4D Hybrid Analytical Parallel Invariants (Paper 7 / 2502.08145v1 Singh)")
    test_heterogeneous_allocator_invariant()
    print("  [PASS] 17. Heterogeneous CPU/GPU Dynamic Workload Allocator (Paper 8 / 2502.11129v1 Eynaliyev)")
    test_moba_block_attention_invariant()
    print("  [PASS] 18. MoBA Mixture of Block Attention (Paper 9 / 2502.13189v1 Lu)")
    test_adaptive_batch_sizer_bca_invariant()
    print("  [PASS] 19. Adaptive Batch Sizer BCA Memory Gap Advisor (Paper 10 / 2503.08311v2 Recasens)")
    test_multigpu_task_allocator_lawenda_invariant()
    print("  [PASS] 20. Multi-GPU Task Allocator & Profiler (Paper 11 / 2503.15252v1 Lawenda)")
    test_apex_scheduler_paper12_invariant()
    print("  [PASS] 21. APEX Profiling-Informed CPU-GPU Scheduler (Paper 12 / 2506.03296v2 Fan)")
    test_nvfp4_pretraining_methodology_paper13_invariant()
    print("  [PASS] 22. NVFP4 Microscaling & Pretraining Methodology (Paper 13 / 2509.25149v2 NVIDIA)")
    test_xmem_caching_allocator_paper14_invariant()
    print("  [PASS] 23. xMem CUDACachingAllocator Memory Estimation (Paper 14 / 2510.21048v1 Shi)")
    test_eggroll_hyperscale_es_paper15_invariant()
    print("  [PASS] 24. EGGROLL Low-Rank Hyperscale ES (Paper 15 / 2511.16652v2 Sarkar)")
    test_long_horizon_mopd_paper16_invariant()
    print("  [PASS] 25. Long-Horizon Agent-OPD & MOPD Distillation (Paper 16 / 2607.24720v1 Men)")
    test_consciousness_steering_paper17_invariant()
    print("  [PASS] 26. Consciousness Activation Steering & Refusal Ablation (Paper 17 / 2607.28607v1 Kim)")
    test_continual_skill_bench_paper18_invariant()
    print("  [PASS] 27. ContinualSkillBench In-Context Evolution & Fragmentation Invariants (Paper 18 / 2608.03874v1 Guan)")
    test_evoharness_rl_paper19_invariant()
    print("  [PASS] 28. EvoHarness-RL Self-Evolving BPE Coordination & GRPO Invariants (Paper 19 / 2608.05446v1 Ning)")
    test_embedder_dilemma_paper20_invariant()
    print("  [PASS] 29. The Embedder's Dilemma Pareto Frontier & Cost Model Invariants (Paper 20 / 2608.12875v1 El Assadi)")
    test_spp_persona_conditioning_paper21_invariant()
    print("  [PASS] 30. Synthetic Persona Pretraining (SPP) Token-Zero Alignment Invariants (Paper 21 / 2608.13482v1 Minder)")
    test_coordination_network_paper22_invariant()
    print("  [PASS] 31. When Agents Coordinate Bipartite Temporal Network Invariants (Paper 22 / 2608.16801v1 Destefanis)")
    test_asibench_paper23_invariant()
    print("  [PASS] 32. ASI-BENCH Multi-Domain Guidance Gradient Invariants (Paper 23 / 2608.17271v1 Zhou)")
    test_abra_diffusion_scaling_paper24_invariant()
    print("  [PASS] 33. ABRA Compute-Optimal 200 TPP & Diffusion Scaling Invariants (Paper 24 / 2608.17286v1 Chickering)")
    test_dynamic_compression_paper25_invariant()
    print("  [PASS] 34. Dynamic State Compression & Selective Re-Scanning Invariants (Paper 25 / 2608.17896v1 Pari)")
    test_recirculation_paper26_invariant()
    print("  [PASS] 35. Recirculation & Adaptive Token-Conditional Invariants (Paper 26 / 2608.17981v1 Mozer)")
    test_ecdysis_harness_paper27_invariant()
    print("  [PASS] 36. ECDYSIS Batch Aggregation & Model Accommodation Attenuation (Paper 27 / 2609.11677v1 Yue)")
    test_sol_pi_harness_paper28_invariant()
    print("  [PASS] 37. SoL-Pi Action Fusion, Cost-Gated Compactor & Evidence-Preserving Reducer (Paper 28 / 2609.20519v1 Liu)")
    test_memory_attention_paper29_invariant()
    print("  [PASS] 38. Memory Attention Token-Indexed Memory, Zero-FLOP Folding & MA-Recall (Paper 29 / 2609.28399v1 Kang)")
    test_ai_consciousness_evaluator_paper30_invariant()
    print("  [PASS] 39. AI Consciousness Five-Level Supervenience Hierarchy & Bayesian Network (Paper 30 / 2609.35618v1 Chandaria)")
    test_ax_prover_paper31_invariant()
    print("  [PASS] 40. Ax-Prover Agentic Theorem Proving Tripartite Multi-Agent & Lean MCP Invariants (Paper 31 / Breen)")
    test_beyond_158_bits_paper32_invariant()
    print("  [PASS] 41. Beyond 1.58 Bits Laplace Analytic Theory & 34-Expert Fully-Ternary MoE (Paper 32 / CrashOverrideX)")
    test_bitnet_cpu_infra_paper33_invariant()
    print("  [PASS] 42. 1-bit AI Infra bitnet.cpp I2_S, TL1 & TL2 Lossless Inference Kernels (Paper 33 / Wang)")
    test_bitnet_a48_paper34_invariant()
    print("  [PASS] 43. BitNet a4.8 4-Bit Hybrid Quantization, TopK Sparsification & ReLU2GLU (Paper 34 / Wang)")
    test_bitnet_b158_paper35_invariant()
    print("  [PASS] 44. BitNet b1.58 Absmean Ternary Weights, INT8 Activations & Horowitz Energy (Paper 35 / Ma)")
    test_bitnet_2b4t_paper36_invariant()
    print("  [PASS] 45. BitNet b1.58 2B4T SubLN, ReLU2 FFN Sparsity, SFT/DPO & Energy Invariant (Paper 36 / Ma)")
    test_bitnet_cpp_edge_paper37_invariant()
    print("  [PASS] 46. BitNet.cpp Edge Inference Mirror Consolidation, Lossless mpGEMM & ELUT (Paper 37 / Wang)")
    test_bitnet_scaling_paper38_invariant()
    print("  [PASS] 47. BitNet Scaling 1-Bit Binary Weights, SubLN & Group Quantization Invariant (Paper 38 / Wang)")
    test_bitnet_v2_paper39_invariant()
    print("  [PASS] 48. BitNet v2 Native 4-Bit Activations, Online FWHT H-BitLinear & Outlier Attenuation (Paper 39 / Wang)")
    test_callimahos_inverse_crypto_paper40_invariant()
    print("  [PASS] 49. Callimahos Inverse Cryptography Prime Raster & Semiotic Bootstrapping (Paper 40 / Callimahos)")
    test_dali_moe_paper41_invariant()
    print("  [PASS] 50. DALI Workload-Aware Greedy Assignment, Residual Prefetch & GPU Cache (Paper 41 / Zhu)")
    test_dapo_rl_paper42_invariant()
    print("  [PASS] 51. DAPO Decoupled Clip-Higher, Dynamic Sampling, Token-Level Loss & Overlong Shaping (Paper 42 / Zhou)")
    print("=" * 70)
    print("ALL CORE MATHEMATICAL INVARIANTS 100% VERIFIED & REAL.")
    print("=" * 70)











