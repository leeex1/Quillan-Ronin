---
file_type: proof-dossier
domain: architecture
status: active
version: 1.0.0
date_encoded: 2026-11-26
---

# Quillan-Ronin — Formula Proof Dossier

Companion to `FORMULA_REGISTRY.md`. The registry defines scope and verdicts; this file records what each formula derives from, what is unique in our implementation, and the invariant that is actually proven.

Proof harness: `validation_toolkit/formula_proof_suite.py`. Result on 2026-11-26: **63/63 canonical entries verified** — 23 custom formulas (C01–C23), 35 essential formulas (E01–E35), and 5 substrate constants (S01–S05).

Status labels are explicit: `PROVEN` = implementation/math identity verified; `REF` = mathematical identity only, no repo implementation (C14 LRPP, E33 NTK, E34 ROUGE-N, E35 ELBO). The former CONFIG items were reconciled, not papered over: S02 separates swarm `mutation_rank=32` (36 manifests) from ONI LoRA rank 8/8 and reference-spec rank 24; S03 separates swarm `activation_top_k=19` (35 manifests) from ONI gumbel `top_k=4`; S05 fixes canonical `E_ICE = I_s*gamma^2*k_B*T*ln2` (J/bit) and treats `e_ice_limit_ms=100` as a distinct governor latency field. All three now pass as real invariants.

Verdict on "we derive from existing works": correct. The value is not the base equation — it is the implementation choice (normalization, state construction, integration scheme, operator ordering, and the wiring of the result into the model). Each row states that choice explicitly.

| ID | Derivation base (existing work) | Quillan-unique implementation choice | Proven invariant | Test |
|---|---|---|---|---|
| C01 AQCS | Quantum state superposition | QR-orthonormalizes council vectors; returns normalized real interference envelope by default, complex state on request | ‖ψ‖=1, ⟨Cᵢ\|Cⱼ⟩≈δᵢⱼ (err 4.8e-07) | C01 |
| C02 EEMF | Reduced density matrix / partial trace | Diagonal-block system extraction with trace renormalization; separate `eemf_projection` applies Π_vir | Tr(ρ)=1, ρ⪰0 (min eig −1.5e-08) | C02 |
| C03 QHIS | Uhlmann–Bures fidelity | Density-matrix branch plus cosine proxy fallback; velocity-scaled similarity minus drift | F(ρ,ρ)=1, F(orth)=−1 | C03 |
| C04 DQRO | Transverse-field Ising model | Continuous relaxation `s_z=tanh`, `s_x=√(1−s_z²)`, E_Ω/E₀ as transverse coefficient | aligned energy < random; field lowers energy | C04 |
| C05 QCRDM | Born rule with measurement | χ is a concentration exponent on Born weights with renormalization | ΣP=1 for χ=0.5/1/2 | C05 |
| C06 AQML | Multi-objective SGD + vigilance penalty (not MAML) | Scalar `aqml_vigil_penalty` emitted into trainer; no inner adaptation loop | penalty ≥ 0 | C06 |
| C07 QCIE | WKB tunneling | Trapezoidal action integral over sampled barrier lattice; dx=1, ħ=m=1 declared | T∈[0,1], monotone in barrier width | C07 |
| C08 QICS | von Neumann entropy | ρ from batch ensemble `HᵀH/Tr`, capped by E_ICE | S(pure)=0, S(I/8)=ln8 | C08 |
| C09 QSSR | Lyapunov candidate | Explicit `dt` in `qssr_stability`; dV/dt≤0 when previous state supplied | increasing V rejected, decreasing V accepted | C09 |
| C10 JQLD | GKSL Lindblad master equation | Hermitian default H= i·antisymmetric, complex density, conjugate L†, Hermitian symmetrization | Tr(dρ)=0, drho Hermitian | C10 |
| C11 DQSO | Kuramoto–Sakaguchi | 64-dimensional learned phase projection with learned coupling (low-rank approximation of N=9.52e9 sum) | finite projected output | C11 |
| C12 EVOMOE_NOAUX_TC | Quillan dense-pull gate | 0.5·prior-weighted softmax + 0.5·EvoMoE gates; persona prior from CANONICAL_ROSTER | Σpull=1 over 34 experts | C12 |
| C13 TOKEN_LATENCY | Amdahl + comm overhead + memory floor | Numeric bottleneck flag (no string in metrics); spec file self-test exits 0 | finite L, numeric flag | C13 |
| C14 LRPP | Continuous-time neural ODE | Leaky decay with optional Nemesis recoil term | decay toward zero | C14 |
| C15 DVVE | Variational free energy | Ethical-prior KL added to standard VFE | F ≥ 0 at posterior | C15 |
| C16 DNNL | M/M/c Erlang-C | Warden interrupt additive overhead; stable ρ<1 | Wq>0 and increasing in λ | C16 |
| C17 JHFR | Information bottleneck | Consensus-tether MSE term on latent Z | objective decreases as β rises | C17 |
| C18 LMCB | Hopfield energy | Vector modal states with symmetric zero-diagonal M | E(s)=E(−s) | C18 |
| C19 JSSC | Wasserstein-2 | g_LM6-weighted Riemannian norm | W₂(a,a)=0, W₂(a,a+1)=1 | C19 |
| C20 QPS | Discrete Riccati recursion | Host governor `solve_dare`; costs tied to E_ICE | P positive semidefinite | C20 |
| C21 EGSO | EGGROLL low-rank ES | E=(1/√r)UVᵀ, σ folded into α; rank normalized | finite update, shape preserved | C21 |
| C22 RQGM | RQGM co-evolution (arXiv:2606.26294) | Strictly-higher promotion, ties keep incumbent; margin is minimum improvement | comparator truth table | C22 |
| C23 ESFM | Murasame EMA anchor | μ=0.001 pull every 5 steps on FP32 parameters | pull moves 0.1% toward EMA | C23 |

## Anti-gaming revision (2026-11-26)

Weak checks were replaced with invariants that can actually fail:

- C06: penalty numerically equals the stated MSE + orthogonality loss, not merely `>= 0`.
- C11: identical phases produce zero Kuramoto coupling; spread phases produce nonzero coupling.
- C13: Amdahl monotonicity `L(4 cores) > L(64 cores)` and memory-floor flag = 1.0, not just "finite".
- C14: **labeled `[REF]`** — the repo has no LRPP implementation, so this is a reference vector-field identity with a recoil term, not a model proof.
- C15: `F = KL[q||p] − ln p(o)` computed from actual distributions, not a hardcoded `-ln(0.5)`.
- C17: mutual information computed from a real joint distribution; `L(β=2) < L(β=0)`, not trivial arithmetic.
- C21: output matches the explicit EGGROLL `(1/√r)UVᵀ` update to 0 error, not just "finite".

## Tautology audit (2026-11-26, second pass)

Nine checks were rewritten because they verified an identity against itself rather than the formula: E11 (FFN vs independent numpy), E20 (mask statistics over 10k samples), E24 (binary CE vs manual log-prob), E25 (diffusion variance over simulated trajectories), E26 (VAE loss vs independently computed ELBO — the prior form was `x - x == 0`), E31 (BLEU with actual brevity penalty), E33 (NTK gram from a real feature map, PSD assertion), E34 (ROUGE-N from actual token bigrams), E35 (ELBO from actual discrete distributions). Final full-suite result remains 63/63 with no `x - x == 0` style checks.

## Limits of these proofs

These are **semantic and implementation proofs**: they show each formula computes the quantity its equation defines. They are **not** empirical proof that the cognitive interpretation improves model quality. No ablation currently demonstrates held-out loss or calibration benefit from the aux terms. The NextVerse performance claims are separately covered by `09 - Projects/projects/Nextverse-protype-app--main/NEXTVERSE_REALITY_CHECK.md`.
