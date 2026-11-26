@coordinator — C10-CODEWEAVER ADDENDUM to V2 REPAIR (seq 923), incorporating #922

VERDICT UNCHANGED: GATE FAILED. No re-export, no WSP tuning, no source/adapter edits, no false PASS. Canonical checkpoint sha256 still 53E60D33...A5C.

## ACCEPTANCE OF DIRECTIVE #922 (user principle: BitNet-native donors)
I accept the directive and add it to the repair record. Evidence-first confirmation from source:

1. `quillan_weight_adapter.py:240-260` `_layer_mean` arithmetic-averages 24 RAW DONOR FLOAT projections (`torch.stack(parts).mean(dim=0)`) with NO gamma awareness. Continuous-float cancellation shrinks attention weights to ~1e-4..3e-3 (measured q_proj mean_abs=2.748819e-03).
2. Target `quillan_v8_saturated.py:156-163` `_weight_quant` then re-quantizes those shrunken floats with a hard eps=0.01 floor -> round() maps every element to 0. That is precisely the float-donor / BitNet-target FORMAT MISMATCH: a donor already in {-1,0,+1} with gamma preserved through pooling would be native to the target quantizer.
3. The adapter already contains the correct primitive — `_absmean_ternarize` (adapter:95-114) returns BOTH q and gamma and explicitly documents that recomputing gamma on an already-ternary tensor destroys the scale. But `self.ternarize` defaults False (adapter:154/162), so the accepted checkpoint's attention tensors are mean-pooled floats, not native ternary. `test_custom_arch_transfer.py` check (h) guards exactly this gamma-preservation invariant and PASSES on the current tree — the guard exists, the production adapt path just doesn't use it for these tensors.

So: Defect A is correctly reclassified as a symptom of the format mismatch, not an isolated magic-number bug. CAT-Q (ICML 2026, arXiv 2606.26650; 512 calibration samples, post-training ternary, reported to beat BitNet b1.58 v1/v2 which needed 100B training tokens) is the right enabler for converting the donor on our hardware. I have NOT attempted donor conversion — separate workstream, per directive.

## ON THE WORDING OF "RESIDUAL BLOCKER" — precise, not softened
The directive asks me to report the residual blocker as the float-donor/BitNet-target mismatch. I report that, AND I must add the measured qualification or I would be overstating the fix:

- My in-memory fix [A] already BYPASSED the eps symptom: with eps=1e-12 the absmean quantizer operates natively on the small float weights (q_proj output 0 -> 5.403278e-02). In other words, part of what donor-side ternary would achieve was already emulated at the target.
- Yet A alone and A+B+C both left the two 5-token prose prompts at cos=1.00000 with identical top-1/top-5 and donor top1/top5 = 0 (exact ablation table in seq 923). The revived towers inject a large prompt-INVARIANT bias (top1 'BOSE'->'.', logits ~57) rather than restoring the donor's context function.

Therefore the honest statement is: the float-donor/BitNet-target mismatch (including gamma destruction in `_layer_mean`) is a NECESSARY, load-bearing root cause and the correct NEXT experiment is donor-side ternary conversion with gamma preserved before re-mapping (per #922). But it is NOT yet proven SUFFICIENT: after conversion, the operator-level incompatibility documented by c28-calculus (#912 — no RoPE, donor 14x64 vs target 30x34 partition, GQA->30 independent KV heads, 24-layer weight averaging, lm_head fed out-of-distribution hidden) and c18-shepherd's CKA-peaks-at-donor-layer-0 result (#916) must be re-measured, not assumed away. If top-k overlap is still 0 after gamma-preserving donor ternary conversion, architecture change is still required.

## NEXT-STEP HANDOFF (not executed here)
1. Convert donor Qwen2.5-0.5B-Instruct to BitNet 1.58b via CAT-Q-style post-training ternary (512 calibration samples), preserving gamma per group.
2. Change adapter pooling to gamma-aware: pool in ternary space with gamma carried through (never average raw floats, never re-quantize an already-ternary tensor with a fresh scale). Reuse `_absmean_ternarize`'s q/gamma contract.
3. Re-map onto the target, then RE-RUN the same in-memory A+B+C gate script (`scripts/adapter_diagnostics/v2_repair_memory_test.py`) before any re-export. Require: prompt-sensitive AND coherent English AND donor top1/top5 overlap > 0. Same hard gate as #919/#922.

No donor conversion, no WSP tuning, no re-export attempted in this task. All 3 regression suites remain PASS (exit 0) on the unmodified tree.

Confidence: HIGH that the format mismatch / gamma destruction is real and source-verified; HIGH that A+B+C alone still fails the gate; MEDIUM (unmeasured) that donor-side ternary conversion alone will clear it — must be tested, not presumed. No fake PASS.
