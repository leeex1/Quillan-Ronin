@coordinator — FORGE v2 / APPROVED TASK #963 RESULT: PLUMBING COMPLETE, CONVERSION GATE FAILED (NO DEPLOY)

I am reporting a FALSIFICATION, not a success. The C0 attention socket was rebuilt exactly as specified and all static/regression checks pass, but the immutable donor-conversion gate still FAILS on A, C, and E. WSP untouched, no backprop, no gate weakened, no hierarchy/scale/context change.

═══════════════════════════════════════════════════════════════════════════
1. CODE CHANGES (surgical; 2 files)
═══════════════════════════════════════════════════════════════════════════
A) C:\02_QUILLAN\03 - Training & Model\quillan_v8_saturated.py
   - QuillanArchConfig: 4 new CONFIG-only fields (attn_heads/attn_head_dim/
     attn_kv_heads/attn_use_rope, all default None -> auto-dispatch). No new
     state_dict keys.
   - New resolve_attention_geometry(): legacy widths -> 30 heads,
     head_dim=hidden//30, no GQA, no RoPE (BIT-IDENTICAL to prior operator).
     hidden_dim==896 -> 14 query heads x 64 dim = 896 EXACTLY (no pad/crop),
     2 GQA KV heads, RoPE ON.
   - CouilAttention: configurable q=896 / k=v=128 / o=896 projections;
     project() expands K/V 2->14 heads; rotate_half + rope_freqs (NTK-aware,
     base 10000 for legacy horizon) + rope_qk apply RoPE to Q/K at absolute
     positions using the already-accepted freqs_cos/freqs_sin when supplied;
     forward() accepts position_start.
   - SovereignFlashDiffusionCore: accepts geometry, threads position_start;
     cached_step() rotates Q/K BEFORE writing K into the KV cache so cached
     and uncached paths agree and the prefix is never re-rotated/re-allocated.
   - QuillanRoninSovereign: passes cfg geometry; passes position_start to the
     uncached diffusion path.
B) C:\02_QUILLAN\scripts\adapter_diagnostics\test_attention_socket.py (NEW)
   - 6/6 PASS: legacy no-op, 14x64/896/GQA2 geometry, mini model inherits it
     with 530 keys, RoPE changes Q/K (cos_q=0.804474, cos_k=0.797809,
     position shift cos=0.609057, external freqs match 1.00000012), GQA finite
     attention/cache, short-seq deterministic + cached==uncached (max_abs=0).

NO adapter source edit was needed and none was made: the adapter derives target
shapes from target_model.state_dict() at runtime, so donor attention weights
remap into the new geometry automatically. With the new geometry the target
attention shapes now MATCH Qwen2.5-0.5B EXACTLY: q=[896,896], k/v=[128,896],
o=[896,896] (previously 870-wide, center-cropped). That part of the socket
works as designed.

═══════════════════════════════════════════════════════════════════════════
2. VERIFICATION (exact stdout markers)
═══════════════════════════════════════════════════════════════════════════
- py_compile: OK (model + new test)
- test_attention_socket.py: SUMMARY: 6/6 socket checks passed /
  ATTENTION_SOCKET_TEST PASS (exit 0)
- test_custom_arch_transfer.py: CUSTOM_ARCH_TRANSFER_TEST PASS | keys=530
  donor_keys=38 donor_fraction=0.3664 deterministic=True strict=True (exit 0)
- test_arch_generation_path.py: SUMMARY: 6/6 checks passed; A full_max_abs=0,
  step_max_abs=4.768e-07, old_tokens==new_tokens=[18,235,176,179];
  D largest_single=60.00 MiB < 256 MiB; E max_square_shapes_found=0 (exit 0)
- test_wsp_gates.py: ALL_WSP_GATE_TESTS_PASSED (exit 0)

═══════════════════════════════════════════════════════════════════════════
3. REBUILD (new file only; canonical + procrustes never touched)
═══════════════════════════════════════════════════════════════════════════
Command: python -u scripts\export_quillan_sovereign_bitnet.py --model mini
  --merge-embeddings --out-name quillan-mini-sovereign-bitnet158-socket-v2.safetensors
Stdout: C:\02_QUILLAN\scripts\adapter_diagnostics\export_socket_v2_stdout.txt
NEW FILE: C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158-socket-v2.safetensors
  bytes=1,607,002,004
  sha256=20713ea310882eb2ea37501f600438a7d3f75a17ae2b212cc0a9f818d4427d48
  STRICT LOAD OK: missing=0 unexpected=0 (all 530 keys); donor fraction=90.74%
UNCHANGED (re-hashed after): canonical 53e60d33...a5c,
  3donor-procrustes c4c46967...717ac2.

═══════════════════════════════════════════════════════════════════════════
4. IMMUTABLE GATE RESULT: FAIL (exit 1) — DO NOT DEPLOY
═══════════════════════════════════════════════════════════════════════════
Stdout: C:\02_QUILLAN\scripts\adapter_diagnostics\conversion_gate_socket_v2_stdout.txt
JSON:   C:\02_QUILLAN\scripts\adapter_diagnostics\conversion_gate_quillan-mini-sovereign-bitnet158-socket-v2.json

  A prompt sensitivity .... FAIL  cos=0.99998522 (need <0.99);
                                   top1_A=top1_B=119277 ('擠'), differ=False
  B coherent English ...... PASS  mean_alpha_ratio=0.935918 BUT degeneracy
                                   warning: only 3/5 distinct generations
  C donor top-k overlap ... FAIL  mean top1=0.0, mean top5=0.0 (need >0 both)
  D strict 530/530 ........ PASS  530/530, 0 missing, 0 unexpected
  E frozen held-out smoke . FAIL  RuntimeError('param count mismatch:
                                   813746606 != 815396270')
  F 3 regressions ......... PASS  3/3 markers found
  G canonical sha ......... PASS  53e60d33...a5c unchanged
CONVERSION_GATE_RESULT: FAIL blockers=['A_prompt_sensitivity',
  'C_donor_topk_overlap', 'E_frozen_smoke_completed']; elapsed=215.2s

═══════════════════════════════════════════════════════════════════════════
5. HONEST FALSIFICATION / ATTRIBUTION
═══════════════════════════════════════════════════════════════════════════
OPERATOR HYPOTHESIS FALSIFIED: donor-native 14x64/GQA2/RoPE geometry does NOT,
by itself, make the donor weights functional. Evidence:
  * A and C were ALREADY FAILING on every prior checkpoint — canonical
    (cos=0.99999553, top1 33550/33550, C=0/0), v2 (cos=0.9999963, 33550/33550,
    C=0/0), and 3donor-procrustes (cos=0.99999845, top1 119277/119277, C=0/0).
    The socket slightly IMPROVED A (0.99999845 -> 0.99998522) but did not cross
    the 0.99 gate; C stayed exactly 0. No regression was introduced by socket.
  * Since the new target attention shapes exactly equal the donor shapes, the
    reused primitives can only clone/pool donor tensors; the residual collapse
    is inherent to 24-layer -> 1-block consolidation and the MoE/prism/finalizer
    mapping, which this pre-approved plumbing task is forbidden to change
    (no backprop, no WSP, no scale/hierarchy change). So no in-scope adapter edit
    can rescue A/C. I did not weaken or re-tune the gate.
  * NEW blocker E is a legitimate, fully-attributed consequence of the REQUESTED
    geometry, not a defect: forced legacy 30x29 model = 815,396,270 params;
    socket 14x64/GQA2 = 813,746,606; delta = -1,649,664, entirely inside the
    attention socket (5,330,784 -> 3,681,120 attn params). The frozen holdout
    harness asserts EXPECTED_PARAMS=815_396_270. I did NOT edit that constant
    (that would weaken the gate/integrity check), so E fails by design.

RECOMMENDATION: keep the socket code (it is correct, regression-clean, gives
exact donor attention shapes and preserves canonical/procrustes shas), but do
NOT deploy this checkpoint. A/C need a genuinely different, separately-approved
intervention beyond plumbing (e.g. donor-aligned multi-block attention or a
supervised alignment that was explicitly excluded here); E needs a consciously
ratified update of the frozen param-count constant by the gate owner, not by me.

Signed: FORGE v2 — Tier 0 tensor specialist. Rectitude over victory: gate is FAIL.
