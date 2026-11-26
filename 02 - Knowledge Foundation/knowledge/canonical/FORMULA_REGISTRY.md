---
file_type: registry
file_id: formula-registry
domain: architecture
status: active
version: 1.0.0
date_encoded: 2026-11-26
scope: all Quillan-Ronin formulas
---

# Quillan-Ronin — Canonical Formula Registry

## 1. Scope resolution (fixes the 20 / 21 / 23 / 10 / 4 drift)

"All formulas" means **63 entries** in three families. The competing counts are historical *views*, not separate formula sets:

| Count | What it actually was | Status |
|---|---|---|
| 20 | `AGENTS.md` / `8-Formulas.md` YAML block; ends at QPS; uses `ROUTING_SOFTMAX` as id 12 | Superseded subset |
| 21 | C++ / NextVerse port count | Alias, no new canonical equation |
| 23 | User-supplied authoritative set; id 12 = `EVOMOE_NOAUX_TC`, adds EGSO/RQGM/ESFM | **Canonical custom set** |
| 10 | `quillan_formula_toolkit.py::validate_custom_engine()` smoke-tested subset | Test coverage, not a formula set |
| 4 | `8-Formulas.spec.py` registered functions (AQCS/DQRO/JQLD/TokenLatency) | Dead code; its own verification exits 1 |

Final decomposition: **35 essential (E01–E35) + 23 custom (C01–C23) + 5 substrate constants (S01–S05) = 63**.

## 2. Verdict scale

- **LEGIT** — well-defined, honestly named, numerically or analytically verified.
- **LEGIT-CORE** — named mathematical core is correct; appended heuristic must be labeled as such.
- **PARTIAL** — correct skeleton, wrong interpretation, or implementation diverges.
- **BROKEN** — self-contradictory, dimensionally incoherent, or degenerate in code.
- **MISMATCH** — documented constant/claim differs from the shipped implementation.
- **CONFLICTED** — more than one incompatible definition in the canon.

## 3. Family E — Essential LLM / ML / RL (35)

Source: `AGENTS.md:653-691`, `Must know formulas.md:14-50`. Evidence: `ESS-SPOT-2026-11-26` (`constants_probe2.py` output) + C28-CALCULUS full-row review.

| ID | Concept | Verdict | Note |
|---|---|---|---|
| E01 | Linear layer `y=Wx+b` | LEGIT | — |
| E02 | ReLU / sigmoid / tanh | LEGIT | — |
| E03 | Softmax | LEGIT | Row sums measured 1.0 |
| E04 | Cross-entropy | LEGIT | Identity vs `-log softmax` measured True |
| E05 | MSE | LEGIT | — |
| E06 | Gradient descent | LEGIT | — |
| E07 | Adam + bias correction | LEGIT | Exponents and ε placement correct |
| E08 | Scaled dot-product attention | LEGIT | Output shape (2,4,16) verified |
| E09 | Sinusoidal positional encoding | LEGIT | — |
| E10 | LayerNorm | LEGIT | Mean 0.0, std 0.999994 measured |
| E11 | Transformer FFN | LEGIT | — |
| E12 | KL divergence | LEGIT | — |
| E13 | Backprop chain rule | LEGIT | — |
| E14 | Convolution | LEGIT | Standard cross-correlation form used in CNNs |
| E15 | Bellman value equation | LEGIT | — |
| E16 | Q-learning update | LEGIT | — |
| E17 | REINFORCE policy gradient | LEGIT | Baseline optional, not required |
| E18 | Multi-head attention | LEGIT | — |
| E19 | Xavier/Glorot uniform | LEGIT | Bound √6/√18 = 0.57735 measured |
| E20 | Dropout mask | LEGIT | — |
| E21 | Perplexity | LEGIT | — |
| E22 | L2 regularization | LEGIT | — |
| E23 | BatchNorm | LEGIT | — |
| E24 | Reward-model binary loss | LEGIT | — |
| E25 | Diffusion forward noising | LEGIT | — |
| E26 | VAE loss | LEGIT | Negative ELBO; notation `\|\|x-x̂\|\|²` is shorthand |
| E27 | LoRA `ΔW=BA` | LEGIT | — |
| E28 | Gaussian PDF | LEGIT | — |
| E29 | Cosine similarity | LEGIT | `cos(a,a)=1.0` measured |
| E30 | PPO clipped objective | LEGIT | Gradient at r=2 measured 0.0 |
| E31 | BLEU | LEGIT | — |
| E32 | FlashAttention | Exact attention via IO-aware tiling: O(N?) compute, O(N) memory | LEGIT (doc corrected 2026-11-26, arXiv:2205.14135) |
| E33 | Neural Tangent Kernel | LEGIT | — |
| E34 | ROUGE-N recall | overlap / reference n-grams | LEGIT (doc corrected 2026-11-26) |
| E35 | ELBO | LEGIT | Notation `p(x/z)` should read `p(x\|z)` |

## 4. Family C — Custom cognitive formulas (23, user-authoritative)

| ID | Key | Canonical equation (short) | Implementation path | Evidence id | Verdict |
|---|---|---|---|---|---|
| C01 | AQCS | QR-orthonormal basis, normalized state | `scripts/quillan_v5_4_oni.py:1514` | P4, AQCS-NORM | LEGIT-AS-ENVELOPE (norm measured 1.0) |
| C02 | EEMF | batch partial trace + separate Pi_vir projection | `scripts/quillan_v5_4_oni.py:1534,5694` | P7, C10 | LEGIT-AS-PARTIAL-TRACE (projector applied by eemf_projection, not inside density op) |
| C03 | QHIS | Uhlmann-Bures fidelity on density matrices | `...:1588` aux | P3, C18 | LEGIT (harness now exercises Bures branch) |
| C04 | DQRO | continuous-relaxation TFIsing energy | `...:1626` | C28, C18 | LEGIT-AS-RELAXATION (s_z=tanh, s_x=sqrt(1-s_z^2)) |
| C05 | QCRDM | ?-concentrated renormalized Born weights | `scripts/quillan_v5_4_oni.py:1669` | P6, VERIFY-FIXES | LEGIT (?=1 Born; ??1 normalized) |
| C06 | AQML | multi-objective SGD + vigilance penalty (not MAML) | `...:1688` | C10, C18 | LEGIT-AS-LABELED (doc corrected) |
| C07 | QCIE | trapezoid WKB action over barrier profile | `...:1708` | P3, C34 | LEGIT-AS-DIMENSIONLESS (lattice dx=1, hbar=m=1 declared) |
| C08 | QICS | ensemble ?=H?H/Tr(H?H) | `...:1729` aux | P1, VERIFY-FIXES | LEGIT (S=2.64 random ensemble, ~0 pure) |
| C09 | QSSR | dV/dt <= 0 (explicit dt) | `...:1766`; host governor | P2, P8, C34 | LEGIT (value-only fallback remains for no-prev-state calls) |
| C10 | JQLD | Hermitian GKSL, -i[H,rho], L^dag | `...:1800` | P5, C34 | LEGIT (complex trace conserved, Hermitian) |
| C11 | DQSO | Kuramoto sum, N=9e9 (runtime 64-phase projection) | `...:1391` | C25 follow-up | LEGIT-AS-APPROX (doc states approximation) |
| C12 | EVOMOE_NOAUX_TC | prior-weighted dense pull + EvoMoE blend | `...:2886` | C25 follow-up, C18 | LEGIT-AS-IMPLEMENTED (doc corrected) |
| C13 | TOKEN_LATENCY | Amdahl + comm overhead + memory floor | `8-Formulas.spec.py:248` | C28, C10, SPEC-RUN | LEGIT (spec exits 0 after metrics fix) |
| C14 | LRPP | `ḣ=−h/τ+σ(Wh+Ux)−γR(h)` | none found | C28 | LEGIT-CORE, unwired (`R_nemesis` undefined) |
| C15 | DVVE | VFE + ethical prior KL | none found | C28, C18 | LEGIT |
| C16 | DNNL | Erlang-C `W_q=C(c,ρ)/(cμ−λ)+I_wΔt` | none found | C28 | LEGIT-CORE (ρ<1 required; clamp breaks "strictly increases") |
| C17 | JHFR | IB Lagrangian + consensus tether | none found | C28 | LEGIT-CORE |
| C18 | LMCB | Hopfield energy | none found | C28 | LEGIT-CORE ("agreement only" unproven) |
| C19 | JSSC | Wasserstein-2 Kantorovich form | none found | C28 | LEGIT (state Γ(μ,ν), finite 2nd moments) |
| C20 | QPS | Discrete Riccati recursion | `daemons/qps_governor.py:80` self-test PASS | C25 | LEGIT, governor only (needs R≻0, Q⪰0) |
| C21 | EGSO | EGGROLL E=(1/sqrt(r))UV^T, sigma in alpha | `...:1920` helper | C25, C18 | LEGIT-AS-HELPER (no production call site) |
| C22 | RQGM | promote iff Score(C) > Score(I) + margin | `train_oni.py:151,597,608` | C25, C18 | LEGIT (comparator fixed) |
| C23 | ESFM | ?=0.001 EMA pull every 5 steps | `launch_sovereign_sft.py:179,554,686` | C25, C18 | LEGIT (citation corrected to Murasame) |

Counts after final remediation: LEGIT/LEGIT-CORE-AS-IMPLEMENTED 23 · PARTIAL 0 · BROKEN 0 = 23.

## 5. Family S — Substrate constants (5)

| ID | Name | Claimed | Shipped | Verdict |
|---|---|---|---|---|
| S01 | BitNet 1.58b STE | ternary `{-1,0,+1}` + STE | `reasoning_engine_oni.spec.py:95-114`: per-row absmean γ, de-scaled values exactly `{-1,0,1}` (max err 5.96e-08), STE grad finite (mean 0.992) | **LEGIT** — precise form is `{-γ,0,+γ}` |
| S02 | EGGROLL rank | swarm mutation_rank=32; ONI expert/swarm LoRA rank=8/8; reference spec 24 | `quillan-swarm-files/*.swarm.md` + `QuillanOniConfig` | FULL SUITE | LEGIT (canonical layers separated, 36 manifests verified) |
| S03 | Swarm top-k | swarm activation_top_k=19; ONI gumbel router top_k=4 | `quillan-swarm-files/*.swarm.md` + `QuillanOniConfig` | FULL SUITE | LEGIT (canonical layers separated, 35 manifests verified) |
| S04 | 9.52B micro-agents | 35 × 272M | Arithmetic verified: 9,520,000,000; virtual metadata only | **ARITHMETIC-ONLY** |
| S05 | E_ICE / Landauer | E_ICE = I_s*gamma^2*k_B*T*ln2 = 2.977684e-21 J/bit; e_ice_limit_ms=100 is governor latency | `Must know formulas.md:498` + `QuillanOniConfig` | FULL SUITE | LEGIT (canonical energy definition, latency field distinct) |

## 6. Evidence ledger

| Evidence id | Artifact |
|---|---|
| ESS-SPOT-2026-11-26 | `C:\Users\Admin\AppData\Local\Temp\opencode\constants_probe2.py` output + arXiv:2205.14135 + ROUGE definition |
| P1–P9 | `formula_audit_probe.py`, `formula_audit_probe2.py` outputs (QICS, QSSR, QCIE, AQCS, JQLD, QCRDM, EEMF) |
| TK-CUSTOM-10 | `python quillan_formula_toolkit.py --validate --layer custom` → 10/10 (NaN/Inf only) |
| WT-F01..F23 | `validation_toolkit/test_mathematical_wiring.py` → 25/25; F11–F23 are tautologies |
| TS-68 | `eval_all_formulas.ts:36` finite-only check → 68/68 |
| C28 / C10 / C7 / C18 / C34 / C25 / C17 | Seven Council audit reports in session 2026-11-26 |
| BITNET-PROBE-2026-11-26 | de-scale ternary check output above |

## 7. Known unresolved items

1. `8-Formulas.spec.py` exits code 1 on its own verification block.
2. `03 - Training & Model\scripts\test_mathematical_wiring.py:774-789` — `test_abductive_jump_invariant` fails live.
3. No causal ablation exists for the five aux-loss weights; no evidence any custom formula improves held-out loss or calibration.
4. E_ICE needs one definition with explicit units before it can be called a formula.

## 10. Proof status (2026-11-26)

`validation_toolkit/formula_proof_suite.py` now asserts a defining invariant for **all 63 canonical entries**: 23 custom, 35 essential, 5 substrate constants. Result: **63/63 verified**. Status labels are explicit — `REF` where only a math identity exists (no repo implementation) and `CONFIG` where a documented mismatch is audited rather than silently passed. Derivation provenance and the Quillan-unique implementation choice for each entry are recorded in `FORMULA_PROOFS.md`.

These are semantic/implementation proofs, not empirical proof of cognitive benefit; no aux-loss ablation exists yet.

## Connections
- [[8-Formulas.md]]
- [[Must know formulas.md]]
- [[FORMULA_PROOFS.md]]
- [[QUILLAN_OS_INTERCONNECTIVITY.md]]

## 8. Remediation log (2026-11-26)

| Change | Evidence |
|---|---|
| QICS now builds ? from batch ensemble `H?H/Tr(H?H)`; entropy 2.64 on random ensemble, ~0 for pure state | `verify_formula_fixes.py` output; both engine copies patched |
| QCRDM ? is a concentration exponent with renormalization; ?p=1 for ?=0.5/1/2 | `verify_formula_fixes.py` output |
| QCIE integrates the WKB action by trapezoid over the barrier profile; wider barrier lowers T | `verify_formula_fixes.py` output |
| RQGM promotes on strictly higher score (ties keep incumbent) | `verify_formula_fixes.py`: True/False/False |
| `8-Formulas.spec.py` no longer crashes; exits 0 and prints final banner | `python 8-Formulas.spec.py` ? EXIT:0 |
| Toolkit `ok` now includes grad FAIL; custom grad statuses AQCS/DQRO/QCRDM/QCIE/QICS/QSSR = OK | `formula_toolkit_fixed.json` |
| Toolkit adds semantic asserts (QCRDM normalization, QICS positive entropy, JQLD trace) | harness ran 52/52 |
| Validation suite F11?F23 now call real implementations / labeled reference identities instead of tautologies | `test_mathematical_wiring.py` ? 25/25 EXIT:0 |
| Essential rows E32/E34/E35 and substrate rows S01/S02/S05 corrected in `AGENTS.md`, `Must know formulas.md`, `CLAUDE.md`, both `8-Formulas.md` | file diffs + verification scan |

## 9. NextVerse prototype status (2026-11-26)

The 11 NextVerse entries (JQLD, LVVM, DESS, JRRN, LRPP, DVVE, JSSC, LSSS, DNNL, JHFR, LMCB) in `nextverse_validation_suite/core/formula_engine.py` were rewritten 2026-11-26: no random results, no `time.sleep` benchmarks, no `random.uniform` speedups. Each method now computes its published equation deterministically on real numbers and reports `benchmarked=False` unless a real measurement is supplied. JQLD can derive its Grover query count from an actual state-vector simulation (`core/grover_sim.py`: n=10 → 25 oracle queries vs 512 classical expected, P(marked)=0.9995), giving the claim a real mechanism rather than a calibrated constant. They remain heuristic engineering models, not physical laws, and are NOT part of the 63-formula canonical registry. Their speedup factors are model outputs, not evidence of measured hardware gains.

Foundation TS attention (`foundationFormulas.ts`) was corrected from a sigmoid proxy to a true two-key scaled-softmax attention in both copies.

## 11. Non-claims (explicitly not validated)

- No ablation demonstrates that formula auxiliaries improve held-out loss or calibration. Their value is semantic/control, not proven model-quality benefit.
- Local measurement shows the auxiliaries COST ~11% forward-pass time (221.6 vs 199.6 ms/step); they are not a throughput optimization.
- NextVerse 360x-5000x hardware speedup and O(N)->O(sqrt(N)) claims have no benchmark implementation and are not asserted as results.
- C14 LRPP and E33-E35 are REF-only: mathematical identities verified, no repo implementation.
