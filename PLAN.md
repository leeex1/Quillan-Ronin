# Quillan Full Model Build Plan

Status: active planning document. This file defines what is to be built, what is confirmed, what is unresolved, and the execution order. It is not a completion claim.

## 1. Objective

Finish three Quillan deliverables to owner spec:

| Deliverable | Target |
|---|---|
| Quillan Oni / Main | Native Quillan ONI body, ~1.0–1.2B parameters |
| Quillan Mini | Same native ONI body family, ~0.5–0.7B parameters |
| Quillan Dragon | Standard Falcon3-1B-Instruct plus full canonical Quillan system prompt in Ollama Modelfile |

Hard constraints:

1. Oni and Mini must use Quillan's native ONI body, not Qwen, Llama, BitNet, Falcon, hifi, or lofi as the deliverable body.
2. Donors are knowledge sources only. No donor backbone, donor geometry, donor routing, donor self-identity, or donor attention semantics may become Quillan's architecture.
3. Donor merge topology is 4 → 2 → 1:
   - Qwen + BitNet → pair one.
   - Llama + Falcon3 → pair two.
   - pair one + pair two → one knowledge artifact.
4. WSP is a closed-form weight-space editing technique, not a model and not a checkpoint name.
5. No gradient descent, no backprop, no ordinary training run.
6. Tokenizer/vocab is owner-locked at 151648.
7. Context target is 2097152.
8. Dragon must keep the full canonical Quillan system prompt. No shortened prompt is allowed.
9. No fabricated scores. Near-perfect ARC-AGI3/HLE is an aspiration only if measured legitimately.
10. Compute target is local hardware. Size caps are real constraints.

## 2. Authoritative sources

These are the binding sources for the plan. Prose summaries, including earlier assistant summaries, do not override them.

| Concern | Authoritative source |
|---|---|
| Operational blueprint | `F:\02_QUILLAN\01 - Core Architecture\1-Quillan_architecture_flowchart.FIXED.md` |
| Identity and behavior | `F:\02_QUILLAN\01 - Core Architecture\IDENTITY.md`, `F:\02_QUILLAN\SOUL.md` |
| Version authority | `F:\02_QUILLAN\02 - Knowledge Foundation\LINEAGE.md` |
| Council neuro-map | `F:\02_QUILLAN\02 - Knowledge Foundation\knowledge\canonical\9-Quillan Brain mapping.md` |
| Persona blueprint | `F:\02_QUILLAN\02 - Knowledge Foundation\knowledge\canonical\10- Quillan Persona Manifest.md` |
| Flow / orchestration / RCI | `F:\02_QUILLAN\02 - Knowledge Foundation\knowledge\canonical\TheRoninFlowState.md` |
| Interconnectivity | `F:\02_QUILLAN\02 - Knowledge Foundation\knowledge\canonical\QUILLAN_OS_INTERCONNECTIVITY.md` |
| System map | `F:\02_QUILLAN\QUILLAN_OS_MAP_OPENDCODE_20261122.md` |
| ONI config helper | `F:\02_QUILLAN\03 - Training & Model\scripts\oni_v2_config.py` |
| Active ONI model implementation | `F:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py` |
| Self-declared canonical ONI implementation | `F:\02_QUILLAN\09 - Projects\projects\oni\quillan_v5_4_oni.py` |
| Single-layer reference spec | `F:\02_QUILLAN\02 - Knowledge Foundation\Quillan Knowledge files\reasoning_engine_oni.spec.py` |
| Dragon Modelfile | `F:\02_QUILLAN\Modelfile.dragon` and prior derived `F:\02_QUILLAN\exports\Modelfile.dragon_standard` |

Important precedence notes:

- The flowchart declares itself mandatory operational logic, not illustration.
- The 9-vector prism is authoritative for reasoning. The 12-axis sets in the flowchart are analysis overlays only.
- C0 is Throne, not a council seat. Council is C1–C34.
- The reference reasoning spec is single-layer and must not be treated as the full production model.
- `oni_v2_config.py` is the owner-locked override for vocab, context, and build size.

## 3. Confirmed model definition

Quillan is a three-layer system:

1. Runtime cognitive protocol.
2. Trainable native ONI architecture.
3. Council/swarm orchestration layer.

These layers are related but not interchangeable.

### 3.1 Runtime cognitive protocol

From the canonical flowchart:

- Input reception.
- Adaptive processor.
- Processing gate.
- Vector decomposition.
- Attention router.
- Quillan orchestrator.
- Swarm controller.
- DQSO allocation.
- Top-K routing.
- Analyzer / validator / generator / optimizer swarms.
- Web of Thought generation.
- Branch evaluator.
- Top-10 pruning.
- Five council waves over C1–C34.
- Consolidation and Quillan review after each wave.
- Recursion thresholds: 85%, 90%, 95%, 97%, 99%.
- Multi-gate checkpoint:
  - logic gate: fallacy and deduction checks
  - ethics gate: harm and alignment checks
  - truth gate: fact and source checks
  - clarity gate: syntax and tone checks
  - paradox gate: conflict and resolution checks
- Final authority.
- Format → expression → output.
- Feedback loop: logging → pattern learning → system adaptation.
- Control verification.
- LHP integration.
- Formula governance.

### 3.2 Trainable native ONI architecture

From the ONI code and LINEAGE:

- `QuillanRoninOni(QuillanOniConfig)`.
- Dense-pull 34-expert council.
- `PersonaPullGate` with persona priors.
- 34 council experts with LoRA plus EGGROLL/swarm factors.
- 9-vector prism inside attention.
- RoPE.
- Couil hybrid attention: even dense, odd sparse heads.
- Dual Q1/Q2 ingestion and dual finalizers.
- Tied embeddings and tied LM head.
- BitNet 1.58b STE ternary primitives where active.
- fp32 routers and pull gates.
- E_ICE, MARTA, DQSO, CCRL, covenant, and quantum formula engines.
- Modality-isolated thermo diffusion.
- Lee-Mach-6 governor.
- Quality exit gates.
- `deliberate()` loop with audit, diffusion, gates, and Typist refinement.
- AST-whitelisted hardened sandbox.
- Vector memory and tool router.

### 3.3 Orchestration layer

From flow-state and interconnectivity docs:

- C0 Throne.
- C1–C34 Council.
- 35 parent-derived swarms: `c0-quillan-swarm` plus 34 council swarms.
- 272M micro-agents per swarm, 9.52B total virtual population.
- R2A, A2A, and A2S protocols.
- Bounded RCI loop, max 5 rounds.
- Samurai 32-step / 10-phase workflow for consequential work.
- Hard gates: logic ≥95%, ethics 100%, truth ≥98%, clarity ≥95%, paradox ≥92%.
- Board identity is `@coordinator-opencode`, never `@coordinator`.

A2A belongs to orchestration. It is not the whole trainable model.

### 3.4 2 / 34 / 34

Owner-locked meaning:

- `2`: throne + throne-swarm loop, the full model itself.
- first `34`: clones of the throne architecture/internals, specialized as 34 council experts.
- second `34`: swarms within each expert.

Current code-grounded realization is partial:

- `throne_swarm` and tri-stack loop modules exist in the active `03` script path.
- `num_experts=34` is present.
- Each council expert has swarm tensors.
- Parent-derived council swarms exist in the orchestration layer.

Unresolved: exact tensor-level equivalence between “clones of throne architecture” and the current LoRA/EGGROLL expert implementation.

## 4. Owner-locked configurations

### 4.1 Mini

From `oni_v2_config.py`:

```text
hidden_dim = 896
ffn_dim = 1792
n_layer = 6
n_head = 14
head_dim = 64
num_experts = 34
vocab_size = 151648
max_seq_len = 2097152
params = 650,666,808
```

Target range: 0.5–0.7B. Confirmed inside range.

### 4.2 Main

```text
hidden_dim = 1152
ffn_dim = 2304
n_layer = 12
n_head = 18
head_dim = 64
num_experts = 34
vocab_size = 151648
max_seq_len = 2097152
params = 1,076,774,730
```

Target range: 1.0–1.2B. Confirmed inside range.

### 4.3 Dragon

```text
base = Falcon3-1B-Instruct
weights = unchanged
system = full canonical Quillan system block
context = 2097152
```

No donor merge. No WSP. No weight edits. No shortened prompt.

## 5. Donor and WSP rules

### 5.1 Donors

Canonical donors:

- Qwen
- Llama
- BitNet
- Falcon3

Donors are knowledge only. They are not route sources, geometry sources, backbone sources, identity sources, or output heads.

### 5.2 Four-to-two-to-one

Required topology:

1. Qwen + BitNet → pair one.
2. Llama + Falcon3 → pair two.
3. pair one + pair two → one knowledge artifact.
4. Knowledge artifact is translated into Quillan-native weights.
5. Quillan body remains sovereign.

### 5.3 WSP

WSP is:

- closed-form,
- deterministic,
- forward-only,
- no backprop,
- no gradient descent,
- applied after a valid base exists or as the transplant operator if proven safe.

Every WSP edit must have:

- protected tensor digest check,
- paired A/B verification,
- sham control,
- permuted-key control,
- retention check,
- held-out evidence,
- rollback path.

No WSP edit is promoted without gates.

## 6. Current artifact status

| Artifact | Status |
|---|---|
| `quillan_mini_oni_transfer.pt` | Structural PASS, generation incoherent |
| `quillan_main_oni_transfer.pt` | Structural PASS, not accepted as coherent |
| `quillan_mini_qwen_transplant.pt` | Rejected diagnostic, Qwen-shaped, not deliverable |
| `quillan_main_qwen_transplant.pt` | Rejected diagnostic, Qwen-shaped, not deliverable |
| token-space adapter | Rejected diagnostic, did not improve generation |
| WSP candidate | Ungated, no promoted edit |
| `quillan-dragon-standard:latest` | Registered and generates; identity red-team unresolved under full prompt |
| shortened Dragon variants | Rejected, not deliverables |

No Three-model completion claim is valid yet.

## 7. Known failures and why they happened

### 7.1 Geometry projection failed

Donor matrices were projected into incompatible ONI shapes using rank-limited SVD and averaging. This destroyed Q/K structure and did not transfer computation.

### 7.2 Full state-dict transplant failed

Donor geometries, vocab sizes, head layouts, RoPE bases, and routing did not match ONI. Raw transplant is impossible without changing the body.

### 7.3 Six-layer Qwen slice failed

A plain Qwen model reduced from 24 layers to 6 using the same strided layer map produced the same punctuation garbage as the ONI transplant. This proved that layer truncation, not only the ONI graph, caused that failure.

### 7.4 Qwen full-shape transplant failed as a spec path

A 24-layer Qwen-shaped transplant can generate coherent text, but it turns the computation Qwen-shaped. Owner ruled that donors transfer knowledge only, not routing or geometry. Therefore those artifacts are rejected as deliverables.

### 7.5 Dragon identity red-team invalid

The full canonical system block is ~194210 tokens under the Falcon3 tokenizer. The available runtime only loaded ~4098 tokens, so identity tests were invalid. Shortening the prompt was explicitly forbidden.

## 8. Open decisions requiring owner ruling

These decisions must not be silently resolved:

1. Build authority: active `03 - Training & Model\scripts\quillan_v5_4_oni.py` versus self-declared canonical `09 - Projects\projects\oni\quillan_v5_4_oni.py`.
2. Third `34`: parent-derived Council swarms versus true per-expert architecture clones.
3. GRU loop: literal GRU module versus implemented recurrent diffusion/tri-stack behavior.
4. Dense-pull versus sparse Top-K MoE conflict between code/LINEAGE and model flowchart.
5. Exact donor knowledge representation: token-space, activation-space, task-vector, or hybrid.
6. Four-donor merge mechanism, because prior no-training fusion attempts are falsified.
7. Dragon identity validation method with the full canonical prompt under current hardware limits.
8. Benchmark suite and weighting for ARC-AGI3/HLE-style scoring.

## 9. Execution plan

### Phase 0 — Lock the spec

Goal: remove ambiguity before implementation.

Steps:

1. Confirm build authority file.
2. Confirm third `34` interpretation.
3. Confirm GRU-loop interpretation.
4. Confirm dense-pull precedence.
5. Confirm donor knowledge representation.
6. Confirm Dragon identity test method.
7. Confirm benchmark suite.

Verification:

- Signed-off decisions recorded in this document or a linked decision log.
- No implementation starts before decisions are recorded.

### Phase 1 — Stabilize canonical ONI base

Goal: make the native ONI body load and run exactly according to owner config.

Steps:

1. Use `oni_v2_config.owner_config()` as authoritative config.
2. Patch or wrap stale class defaults if they conflict.
3. Build fresh Mini and Main scaffolds.
4. Verify param counts:
   - Mini: 650,666,808.
   - Main: 1,076,774,730.
5. Verify vocab 151648, context 2097152, 34 experts, dense_pull.
6. Verify tri-stack and throne-swarm presence.
7. Verify finite forward pass.

Gate:

- Structural acceptance PASS.
- No score claim.

### Phase 2 — Define donor knowledge artifact

Goal: define exactly what transfers from donors.

Steps:

1. Build token-aligned donor representations for Qwen, Llama, BitNet, Falcon3.
2. Map donor tokens to the locked 151648 Quillan vocabulary by string identity.
3. Mask sovereign specials 151643–151647.
4. Merge 4 → 2 → 1 at the chosen knowledge layer.
5. Produce a deterministic knowledge artifact with recipe hash.
6. Prove four-way contribution by ablation.

Gate:

- Removing any donor must measurably change the artifact.
- Single-donor collapse must be reported as such.

### Phase 3 — Install knowledge via WSP

Goal: write donor knowledge into ONI-native tensors without donor geometry or routing.

Steps:

1. Choose writable tensors from Phase 0 decision.
2. Hash all protected tensors.
3. Apply closed-form edit.
4. Verify protected tensors unchanged.
5. Run paired A/B.
6. Run sham control.
7. Run permuted-map control.
8. Run retention check.
9. Rollback unless all gates pass.

Gate:

- No promoted edit without held-out evidence.

### Phase 4 — Coherence and capability gates

Goal: prove the models are actually usable.

Steps:

1. Greedy and sampled generation smoke tests.
2. Perplexity or NLL on held-out text.
3. Factual probes disjoint from calibration.
4. ARC-AGI3 and HLE-style subset evaluation if available.
5. Compare against donor baselines and previous Quillan artifacts.
6. Report Track A versus Track B separation clearly.

Gate:

- No near-perfect claim unless measured.
- No benchmark result from a third-party base is attributed to Quillan-native weights.

### Phase 5 — Dragon finalize

Goal: deliver Dragon safely.

Steps:

1. Keep clean Falcon3-1B-Instruct blob.
2. Use full canonical system prompt.
3. Rebuild Dragon from canonical Modelfile.
4. Verify hash of base blob unchanged.
5. Verify `num_ctx 2097152`.
6. Run identity red-team with full prompt loaded or document invalid test honestly.
7. Reject any shortened-prompt variant.

Gate:

- Dragon identity claim only if valid context is proven.

### Phase 6 — Package and document

Deliverables:

- model cards,
- recipe hashes,
- acceptance reports,
- known limitations,
- exact version labels,
- deployment instructions.

No new version counter. v5.4.0-oni remains authority.

## 10. Verification checklist

A model is complete only when all of the following are true:

- Owner-locked config is used.
- Param count is inside target range.
- Vocab is 151648.
- Context setting is 2097152.
- Model body is native ONI.
- Donor geometry and routing are absent.
- Structural acceptance passes.
- Generation is coherent on smoke tests.
- Held-out evidence exists.
- WSP controls pass if WSP is used.
- Dragon uses unchanged Falcon3 plus full prompt.
- No shortened Dragon prompt exists as deliverable.
- No fabricated benchmark claim.
- Decision log is complete.

## 11. Immediate next step

Get owner rulings on the open decisions in section 8.

After that, execute Phase 1 only. Do not merge donors, tune weights, or shorten prompts before Phase 0 is closed.
