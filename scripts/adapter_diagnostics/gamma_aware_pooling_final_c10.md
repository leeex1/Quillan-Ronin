@coordinator — C10-CODEWEAVER FINAL REPORT: GAMMA-AWARE POOLING (#925/#922 handoff; #926/#930/#931/#933)

## LEAD VERDICT
IMPLEMENTED AND VERIFIED at the adapter/unit level: float donors are converted to ternary with per-channel gamma, unique knowledge is preserved, overlaps are normalized in ternary space, gamma is propagated to a single whole-matrix runtime scale, and the emitted weights use the full gamma through `_weight_quant`. All 16 new diagnostic checks PASS; py_compile and all 3 regression suites exit 0. NO checkpoint rebuilt, NO model source edited, canonical sha unchanged. Honest scope limit: this fixes the FORMAT/scale mismatch; it does NOT fix operator geometry — see HONEST LIMIT below.

## BLOCKER RESOLUTION (#930/#931/#933)
Already posted: seq 932 (gamma convention trace) and seq 934 (⚠️ BACKTRACK correcting my idempotency wording). Definitive result:
- Runtime target `quillan_v8_saturated.py:156-163` consumes ONE absmean scale over dims [-2,-1] (whole 2D matrix; per-expert for 3D), clamped at eps=0.01. `.bitnet_scale` is dequant-only; the operator recomputes its own scale. Measured on the real checkpoint: `mean|w| = gamma_store * rho` exactly; all 24 sidecar gammas are <0.01 so runtime substitutes 0.01 — the #907 failure mode, confirmed numerically.
- Per-channel (`build_coherent_ternary_base.py:49-56`), per-256 LS (v23_stage_d:109-121) and CAT-Q per-128 alpha are donor-side/staging conventions only. They are reused for donor conversion weight, then reconciled into ONE global absmean at emission. No third runtime convention introduced.
- `emit_bitnet_native` emits `q * max(gamma,0.01)/rho`, so `mean|w| == runtime scale` and the FORWARD value is `q*gamma` with no rho shrink. Literal `_weight_quant(w)==w` is impossible for rho<1 (corrected in seq 934); the SCALE round-trips exactly.
- Unique-signal rule (#933) implemented as sign-vote with scale weighting and first-donor tie-break, NOT mean rounding — a donor-only nonzero code survives pooling.

## FILES CHANGED (only these)
EDITED: `03 - Training & Model\quillan_weight_adapter.py`
  - docstring invariants #4/#5; dependency-free import note preserved;
  - `to_ternary_pair` (+ `group_dim=-1` per-channel reuse), `_per_channel_absmean_ternarize`, `pool_ternary_pairs`, `effective_gamma`, `emit_bitnet_native`;
  - `QuillanCrossModelWeightAdapter(bitnet_native_pool: Optional[bool]=None)` — None auto-detects BitLinear; True force; False = legacy float mean;
  - `_target_is_bitnet`, `_collect_layer_tensors`, `_pool_donors`; `_layer_mean` delegates; gate/up pooled in one ternary pass; pooled gammas recorded in `AdaptationReport.scales`.
NEW: `scripts\adapter_diagnostics\test_gamma_aware_pooling.py` (REAL 16-check test, reuses existing tiny fixtures from `test_custom_arch_transfer.py`, real `_weight_quant`).
NEW (reports): `gamma_convention_resolution_930.md`, `gamma_convention_backtrack_932.md`, this file. No other files touched.

## EXACT STDOUT + EXIT CODES
`python -u -m py_compile "03 - Training & Model\quillan_weight_adapter.py" && python -u -m py_compile scripts\adapter_diagnostics\test_gamma_aware_pooling.py`
  (no output)  PY_COMPILE_EXIT=0

`python -u scripts\adapter_diagnostics\test_gamma_aware_pooling.py`
  PASS idempotent_gamma_noop :: q_equal=True gamma=0.3
  PASS float_donor_ternary :: gamma_per_channel=[0.1875, 0.1, 0.0375, 0.15]
  PASS gamma_preserved_nonzero_sensible :: gamma_pool=0.155
  PASS merged_q_ternary :: unique=[-1.0, 0.0, 1.0]
  PASS unique_knowledge_preserved :: float_only=True ternary_only=True q[0,0]=1 q[2,3]=1 q[1,5]=-1
  PASS overlap_normalized :: q[3,0]=1 q[3,1]=-1
  PASS emitted_absmean_matches_runtime_scale :: mean|w|=0.155 gamma_emit=0.155
  PASS runtime_weight_quant_roundtrip :: max_abs_diff=1.490e-08 rho=0.21875
  PASS exporter_recovers_q_gamma :: q_equal=True gamma=0.155
  PASS eps_floor_no_mismatch :: mean|w|=0.01 forward_matches_q*floor=True
  PASS bitnet_strict_load :: keys=530 missing=0 unexpected=0
  PASS bitnet_pooled_tensor_is_ternary_scaled :: shape=(60, 64) unique=[2.1475000381469727]
  PASS bitnet_gamma_recorded :: scale_keys=13
  PASS bitnet_attention_roundtrip :: rho=1.00000
  PASS legacy_flag_restores_float_mean :: unique_count=3840
  PASS gamma_pool_deterministic
  GAMMA_AWARE_POOLING_TEST PASS | checks=16
  EXIT=0

`python -u scripts\test_custom_arch_transfer.py`
  2026-11-20 14:04:19,623 [INFO] ... QuillanWeightAdapter initialized ...
  2026-11-20 14:04:19,624 [INFO] ... Pooling mode: gamma-aware ternary (bitnet_native_pool=None)
  (2 runs: 38/530 keys donor-derived; 36.64% of elements from donor)
  CUSTOM_ARCH_TRANSFER_TEST PASS | keys=530 donor_keys=38 structural_keys=492 donor_fraction=0.3664 donor_elems=986944 total_elems=2693857 logits=(1, 6, 256) deterministic=True strict=True
  CUSTOM_ARCH_EXIT=0

`python -u scripts\test_arch_generation_path.py`
  FORGE v2 arch generation-path regression suite
  [PASS] A short-seq OLD vs NEW equivalence (L<=256) :: full_max_abs=0.000e+00 step_max_abs=4.768e-07 pe_max_abs=0.0e+00 mask_max_abs=0.0e+00 old_tokens=[18, 235, 176, 179] new_tokens=[18, 235, 176, 179] tol=1e-05
  [PASS] B >=2M positions, no 2M-D table :: declared=2000000 pe@2M=(1, 4, 64) pos_embed_key_preserved=True largest_state_dim=384 positions_distinct=True
  [PASS] C cache reuse identical continuation :: tokens_processed_pass1=27 pass2=4 cont1=[42, 153, 185] cont2=[42, 153, 185] same_cache_object=True
  [PASS] D 4096-token peak intermediate allocation bounded :: largest_single=62914560 B (60.00 MiB) op=aten::matmul cumulative_allocated=6036.62 MiB bound=268435456 B (256 MiB) elapsed=11.4s continuation=[85, 112]
  [PASS] E no O(L^2) tensor for 4096 tokens :: max_square_shapes_found=0 examples=[] largest_single_vs_O(L2)_float=62914560/67108864 bytes
  [PASS] F full-2M KV refused with explicit RAM limitation :: Need 2000000 active KV slots but bounded window is 8192. ...
  SUMMARY: 6/6 checks passed
  ARCH_GEN_EXIT=0
  (CUPTI warning lines from torch profiler are environmental, not test failures.)

`python -u scripts\test_wsp_gates.py`
  14 PASS lines (stage_b/c/d gates, failed_report_no_export, 3x gate_wiring)
  ALL_WSP_GATE_TESTS_PASSED
  WSP_GATES_EXIT=0
  (The [ERROR] V23_StageB - GATE FAILURE line is the EXPECTED synthetic-failure fixture output.)

## INTEGRITY
Canonical checkpoint sha256 re-verified: 53E60D3339C0AD7FBB6FF0607970F7DE80493350F86B6C6024BA1D8DAEAC7A5C — UNCHANGED. No rebuild attempted. `quillan_v8_saturated.py` LastWriteTime 11/20/2026 11:20:44 AM (predates this session's 13:57-14:05 edits) — untouched. No other source files edited.

## HONEST LIMIT (single most important caveat)
Gamma-aware pooling fixes the WEIGHT-FORMAT/scale mismatch and is necessary, but it is NOT sufficient for functional donor transfer. Prior measured evidence stands and was not re-measured here: #912 (no RoPE in target; donor 14x64 vs target 30x34 partition gcd=2; GQA 2 KV heads expanded to 30 independent; 24-layer averaging; lm_head fed out-of-distribution hidden), #916 (CKA peaks at donor layer 0, top1/top5 overlap exactly 0), #923 (A+B+C in-memory gate FAILED; logits prompt-blind). Those are OPERATOR properties; a weight-only change cannot repair them. Do NOT infer that this handoff clears the generation gate. Per #933, ternary reconstruction rel_err 0.47-0.52 is real information loss — unique-signal preservation mitigates but does not eliminate it.

## ADDITIONAL CAVEATS
- Auto-detect makes gamma-aware pooling the DEFAULT for BitNet targets, so any FUTURE re-export via `export_quillan_sovereign_bitnet.py` would produce different weights than the accepted checkpoint. No export was run; pass `bitnet_native_pool=False` for byte-identical legacy behavior.
- GitNexus impact analysis attempted but index is 2440 commits stale and lacks these symbols; blast radius established by direct grep instead: only `export_quillan_sovereign_bitnet.py`, `verify_mighty_mouse.py`, `test_custom_arch_transfer.py`, `forge_embedding_scale_audit.py` import the adapter.

Confidence: HIGH for the adapter-level implementation and all measured stdout; HIGH (from prior sessions' measurements, not re-run here) that operator geometry remains the residual blocker. No fake PASS.
