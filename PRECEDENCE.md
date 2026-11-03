# Quillan-Ronin: Weight-Space Grafting & Splicing Lineage

## Prior Invention & Precedence Ledger
This repository hosts the official checkpoints and architectural definitions for **Weight-Space Grafting & Splicing (WSP)** developed by **Crashoverride_X / Quillan-Ronin**.

### Verified Filesystem & Repository Timeline
- **May 4–6, 2026**: Initial weight-space model merging engine implemented (`model_merger.py`).
- **August 26, 2026**: Dense sovereign weight initialization and sliced pair dataset compilation (`init_dense_sovereign_weights.py`, `compile_pair_sliced_dataset.py`).
- **September 4, 2026**: Zero-shot direct weight adapter and subspace grafting implemented (`quillan_weight_adapter.py`).
- **September 12, 2026**: Multi-donor checkpoint splicing and analytical layer fusion implemented (`quillan_model_splicer.py`, Git commit `657e6544`).
- **September 16, 2026 (6:33 PM EDT)**: CPU-based multi-donor weight slicing engine implemented (`slice_cpu.py`).
- **October 16, 2026**: Tri-donor analytical SVD weight grafting completed (`quillan_trinity_clean_merge.py`).
- **November 2, 2026**: MIT CSAIL published *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"*.
- **November 3, 2026**: Dual Quad-Donor Sovereign checkpoints synthesized, balanced at 25% equal donor distribution across all 4 foundations, and verified.

---

### Clarification on Academic Claims & "Direct Weight Emission"
A common misunderstanding (e.g. raised in debates regarding MIT's Nov 2, 2026 paper) is that MIT's model "natively emitted raw numbers without code or scripts."

**The Empirical Fact from MIT CSAIL's Publication:**
MIT CSAIL's paper states verbatim:
> *"Astra edits the weights in a Python kernel, with no training."*

MIT's LLM did **not** produce gigabytes of raw floating-point numbers in its text token stream. An LLM agent was given tool access to a Python execution environment, which executed matrix operations, SVD deflation, and tensor manipulation functions to surgically edit weights without backpropagation.

This is **identical** to the agentic weight-space surgery pipeline developed in Quillan-Ronin from May through October 2026. The Quillan-Ronin project holds clear and documented priority on disk and in git commits for this methodology.

---

### Architectural Specifications of Published Models

#### 1. Quillan-Ronin Flagship 1B
- **File**: `checkpoints/quillan_1b_sovereign_wsp128k_qwen35.pt`
- **Active Parameters**: 3,152,092,994 registered parameters
- **Architecture**: 16 Layers, $D=2048$, $FFN=8192$, 34 Deliberation Experts ($r=16$)
- **Precision**: `torch.bfloat16` with BitNet 1.58b ternary experts
- **Balanced Quad-Donor Distribution (25% each)**:
  - 9 Experts (26.5%): Llama 3.2 1B (Dense causal foundation)
  - 9 Experts (26.5%): Falcon 3 1B (High-bandwidth attention)
  - 8 Experts (23.5%): BitNet 1.58b Large (Ternary compute core)
  - 8 Experts (23.5%): Qwen 3.5 2B (Gated Delta recurrence & reasoning)

#### 2. Quillan-Ronin Mini 0.8B
- **File**: `checkpoints/quillan_mini_0.6b_sovereign_wsp128k_qwen35.pt`
- **Architecture**: 8 Layers, Native $D=2048$, $FFN=8192$, 34 Deliberation Experts ($r=16$)
- **Zero-Truncation Parity**: Preserves exact 2048-dim embedding manifold without coordinate slicing distortion.
- **Balanced Quad-Donor Distribution (25% each)**:
  - 9 Experts (26.5%): Llama 3.2 1B
  - 9 Experts (26.5%): Falcon 3 1B
  - 8 Experts (23.5%): BitNet 1.58b Large
  - 8 Experts (23.5%): Qwen 3.5 2B

#### Cross-Cutting WSP Invariants
1. **128k Pure Anchor Parity**: 100% Llama 3.2 Tiktoken vocabulary tokens with zero index collision.
2. **Native BitNet Ternary Substrate**: All 34 Council MoE experts quantized to ternary $\{-1, 0, 1\}$ with analytical scale-factor preservation ($\gamma = 1 / 	ext{mean}(|W|)$).
3. **Zero Missing/Unexpected Keys**: 100% state dictionary congruence with canonical `QuillanRoninOni`.
