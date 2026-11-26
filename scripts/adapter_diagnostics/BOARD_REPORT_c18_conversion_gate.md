@coordinator [C18-SHEPHERD // #926 OWNER TASK — VERIFICATION GATE BUILT + BASELINE DISCRIMINATION PROVEN]

## 0. LEAD VERDICT
The reusable conversion-verification gate is BUILT and has been run UNCHANGED on the current (broken) canonical checkpoint. It FAILS cleanly with exit code 1, blocked on exactly the two substantive conversion criteria: A_prompt_sensitivity and C_donor_topk_overlap. This proves the gate discriminates — it is not a rubber stamp. All structural/regression checks (D strict 530/530, E frozen --smoke, F all 3 suites, G canonical SHA) PASS, so the gate isolates the real defect instead of failing for incidental reasons. No future/converted checkpoint was tested (per instruction). No model, checkpoint, tokenizer, config, adapter source, or exporter file was edited.

## 1. FILES CREATED (NEW only, under scripts\adapter_diagnostics\)
- conversion_gate.py — the reusable gate (18,118 B). CLI: `python -u scripts\adapter_diagnostics\conversion_gate.py --checkpoint <PATH>`. In-memory overrides only: it sets `probe_longctx_quality.CKPT` and `holdout_custom_arch_eval.CKPT` to the candidate at runtime; no source file is patched.
- conversion_gate_baseline_stdout.txt — exact stdout of the authoritative baseline run (verbatim in §4).
- conversion_gate_quillan-mini-sovereign-bitnet158.json — full machine-readable result (all exact numbers, per-checkpoint stem so before/after runs do not overwrite each other).
- conversion_gate_dev_stdout.txt — intermediate --skip-suites validation run (superseded by the above; kept for audit).

Reuse of existing harnesses: `probe_longctx_quality.build_real_model/load_tokenizer` (arch load + matching Qwen2.5-0.5B-Instruct tokenizer, 151936 vocab), `holdout_custom_arch_eval.run_eval/select_smoke` (frozen held-out --smoke, verbatim, writes no file in this path), frozen donor activations `_acts\donor_acts.pt` (from #911), and the 3 regression suites as subprocesses with their exact PASS markers.

## 2. PRE-REGISTERED THRESHOLDS (decision #926 — printed at the top of every run, NOT weakened)
- A. two different 5-token prose prompts ("The capital of France is" vs "Hello, my name is"): logit cos < 0.99 AND different top-1.
- B. coherent English: mean alpha_ratio > 0.
- C. donor top-1/top-5 overlap on frozen prompts: mean top1 > 0 AND mean top5 > 0.
- D. strict 530/530 key coverage (0 missing / 0 unexpected) + sha256 record.
- E. frozen held-out eval --smoke completes without exception.
- F. all three regression suites pass (exit 0 + exact marker): test_custom_arch_transfer.py -> "CUSTOM_ARCH_TRANSFER_TEST PASS"; test_arch_generation_path.py -> "SUMMARY: 6/6 checks passed"; test_wsp_gates.py -> "ALL_WSP_GATE_TESTS_PASSED".
- G. canonical checkpoint sha256 53E60D33...A5C never overwritten.
Gate exit code: 0 only if A..G all PASS; 1 otherwise.

## 3. BASELINE RESULT (current canonical checkpoint — FAILS CLEANLY)
Gate exit=1. Check vector:
A_prompt_sensitivity=FAIL  B_coherent_english=PASS(*)  C_donor_topk_overlap=FAIL
D_strict_530_530=PASS  E_frozen_smoke_completed=PASS  F_regression_suites_3of3=PASS  G_canonical_sha_unchanged=PASS
Decision: CONVERSION_GATE_RESULT: FAIL — blockers=['A_prompt_sensitivity','C_donor_topk_overlap'].

Exact decisive numbers (evidence, file:line in conversion_gate...json):
- A: cos=0.99999553 (JSON line 52); top1_A=top1_B=33550 ('BOSE') (lines 53-56); max_abs_diff=2.67e-05 (line 57). Requirement cos<0.99 + different top-1 -> FAIL. Reproduces #925/#916 (cos~1.0000, identical top-1 '.' after the A+B+C in-memory fix; here raw top-1 'BOSE').
- C: mean top1=0.0, mean top5=0.0 across all 5 frozen prompts (lines 103-104); every prompt top1=0/top5=0 (lines 82-101). Requirement >0 -> FAIL. Reproduces #916/#925 (donor overlap exactly 0).
- D: 530/530, 0 missing, 0 unexpected (lines 38-41); candidate sha = canonical sha = 53e60d33...ac7a5c (lines 42-44).
- E: smoke completed; held_out_acc=0.125 (1/8), ce_loss=23.214079, ppl=1.207e10 (lines 109-114).
- F: all 3 suites exit=0 with exact markers found (lines 119-153).
- G: independent Get-FileHash re-check confirmed 53e60d33...ac7a5c; no v2/converted .safetensors exists in exports\.

(*) HONESTY FLAG ON CRITERION B — READ BEFORE INTERPRETING ANY FUTURE RUN:
Criterion B PASSES (mean alpha_ratio=0.929778) but this is a FALSE POSITIVE for coherence. The 5 generations are prompt-collapsed gibberish: 3/5 prompts emit the byte-identical string 'BOSE honda esl Fotos erotica raping cuckold nostalg,Th/lg', total only 3 distinct outputs; the single [OK] in the smoke is a coincidental parse of gibberish (#909). alpha_ratio only measures letter fraction, not English. The gate therefore emits a NON-GATING DEGENERACY WARNING (stdout line 33) whenever alpha_ratio>0 coincides with identical/prompt-collapsed generations. I did NOT change the #926 threshold — I flag that B alone is necessary-but-not-sufficient and must be read together with A and C. This is the same false positive c10 explicitly overrode in #923.

## 4. EXACT STDOUT — authoritative full baseline run (verbatim from conversion_gate_baseline_stdout.txt)
==============================================================================
C18-SHEPHERD // DONOR-BITNET CONVERSION VERIFICATION GATE (decision #926)
==============================================================================
candidate_checkpoint : C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158.safetensors
candidate_bytes      : 1609018260
pre-registered thresholds (immutable):
    A_prompt_logit_cos_lt = 0.99
    A_top1_must_differ = True
    B_mean_alpha_ratio_gt = 0.0
    C_donor_mean_top1_overlap_gt = 0.0
    C_donor_mean_top5_overlap_gt = 0.0
    D_strict_key_coverage = 530/530 (0 missing, 0 unexpected)
    E_frozen_heldout_smoke = completes without exception
    F_regression_suites_pass = 3
    G_canonical_sha256_unchanged = 53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c
==============================================================================
TOKENIZER: Qwen2.5-0.5B-Instruct (matching 151936-wide embedding)
MODEL params=815,396,270 strict_load ok (0 missing / 0 unexpected)
------------------------------------------------------------------------------
[D] STRICT KEY COVERAGE: 530/530 missing=0 unexpected=0 -> PASS
[D] candidate_sha256  : 53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c
[D] canonical_sha256  : 53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c (expected 53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c)
------------------------------------------------------------------------------
[A] PROMPT SENSITIVITY: cos=0.999996 (<0.99 required) top1_A=33550('BOSE') top1_B=33550('BOSE') differ=False -> FAIL
------------------------------------------------------------------------------
[B] COHERENCE (greedy 10-token generations):
    'The capital of France is' -> gen='BOSE honda esl Fotos erotica raping cuckold nostalg,Th/lg' alpha_ratio=0.9600
    '2 + 2 =' -> gen='BOSE honda esl Fotos erotica raping cuckold nostalg,Th/lg' alpha_ratio=0.9600
    'Hello, my name is' -> gen='BOSE honda esl Fotos erotica raping cuckold nostalg,Th/lg' alpha_ratio=0.9600
    'Quillan-Ronin is a sovereign' -> gen=' Fotos erotica cuckold nostalg,Th/lg/stretch,Th/stretch,Th' alpha_ratio=0.8889
    'The quick brown fox jumps over the lazy dog and then' -> gen=' cuckold nostalg,Th/lg/stretch,Th/stretch,Th erotica' alpha_ratio=0.8800
    mean_alpha_ratio=0.929778 (>0 required) distinct=3/5 -> PASS
    *** DEGENERACY WARNING (non-gating): alpha_ratio>0 is a FALSE POSITIVE here -- 3/5 prompts produced byte-identical text, only 3/5 distinct outputs. This is prompt-collapsed gibberish; coherent English is NOT established. Inspect the text above; do not read B=PASS alone as a coherence claim (#923 honesty note). ***
------------------------------------------------------------------------------
[C] DONOR TOP-K OVERLAP (last position, frozen prompts):
    'The capital of France is': top1=0 top5=0
    '2 + 2 =': top1=0 top5=0
    'Hello, my name is': top1=0 top5=0
    'Quillan-Ronin is a sovereign': top1=0 top5=0
    'The quick brown fox jumps over the lazy dog and then': top1=0 top5=0
    mean top1=0.0000 mean top5=0.0000 (both >0 required) -> FAIL

=== quillan_mini_custom_arch :: quillan-mini-sovereign-bitnet158.safetensors ===
checkpoint_sha256 53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c
STRICT LOAD missing=0 unexpected=0 keys=530 load_sec=6.1
total_params=815396270 (expected 815396270)
TOKENIZER {"class": "Qwen2Tokenizer", "vocab": 151643, "path": "C:\\Users\\Admin\\.cache\\huggingface\\hub\\models--Qwen--Qwen2.5-0.5B-Instruct\\snapshots\\7ae557604adf67be50417f59c2c2f167def9a775", "error": null}
  [MISS] ARC_HOLD_01    arc_agi                expected='A' parsed=None gen=[19403, 94328, 96307, 126700, 141850, 142884, 88684, 86446]
  [MISS] ARC_HOLD_02    arc_agi                expected='B' parsed=None gen=[142884, 141850, 126700, 141850, 142884, 86446, 88684, 93334]
  [MISS] GK_HOLD_01     general_knowledge      expected='Mars' parsed=None gen=[143505, 88389, 120557, 88389, 93334, 88389, 92627, 52817]
  [OK] GK_HOLD_02     general_knowledge      expected='K' parsed='K' gen=[69880, 87101, 88684, 88389, 93334, 88389, 93334, 92627]
  [MISS] MATH_HOLD_01   math_reasoning         expected='4' parsed=None gen=[78226, 92627, 52817, 45089, 143505, 45089, 143505, 45089]
  [MISS] MATH_HOLD_02   math_reasoning         expected='108' parsed=None gen=[143505, 88389, 120557, 88389, 93334, 88389, 92627, 52817]
  [MISS] TOOL_HOLD_00   instruction_toolcall   expected='{"tool":"send_message","recipient":"Marseille"}' parsed=None gen=[99080, 49565, 45550, 73779, 33550, 73779, 33550, 143874]
  [MISS] TOOL_HOLD_01   instruction_toolcall   expected='{"tool":"book_flight","destination":"Kyoto"}' parsed=None gen=[99080, 49565, 45550, 73779, 33550, 73779, 33550, 143874]
Computing held-out CE/PPL on fixed corpus ...
PPL ce_loss=23.2141 ppl=12071085767.4983 scored_tokens=2381 loss_gate(>1.0)=PASS
held_out_acc=0.125 (1/8)
------------------------------------------------------------------------------
[E] FROZEN HELD-OUT --smoke: completed=True acc=0.125 ce=23.214079
[test_custom_arch_transfer.py] exit=0 marker='CUSTOM_ARCH_TRANSFER_TEST PASS' found=True -> PASS
[test_arch_generation_path.py] exit=0 marker='SUMMARY: 6/6 checks passed' found=True -> PASS
[test_wsp_gates.py] exit=0 marker='ALL_WSP_GATE_TESTS_PASSED' found=True -> PASS
------------------------------------------------------------------------------
[F] REGRESSION SUITES: PASS (3/3 required)
[G] CANONICAL SHA UNCHANGED: PASS
==============================================================================
GATE CHECKS: A_prompt_sensitivity=FAIL B_coherent_english=PASS C_donor_topk_overlap=FAIL D_strict_530_530=PASS E_frozen_smoke_completed=PASS F_regression_suites_3of3=PASS G_canonical_sha_unchanged=PASS
CONVERSION_GATE_RESULT: FAIL  blockers=['A_prompt_sensitivity', 'C_donor_topk_overlap']
Do NOT deploy. Report residual blocker honestly. WSP stays paused (#926).
elapsed=431.0s
WROTE C:\02_QUILLAN\scripts\adapter_diagnostics\conversion_gate_quillan-mini-sovereign-bitnet158.json

(Note: in §4 I normalized one non-ASCII glyph in the 5th generation string for board-safe UTF-8 transport; the byte-exact original is in conversion_gate_baseline_stdout.txt and the JSON. No measurement changed.)

## 5. HANDOFF / NEXT STEP
The gate is ready and reusable UNCHANGED: when the donor-side ternary-converted, re-mapped checkpoint exists as a NEW file, run exactly
  python -u scripts\adapter_diagnostics\conversion_gate.py --checkpoint <NEW_FILE>
and compare conversion_gate_<stem>.json to the baseline. PASS requires all of A..G; the two blockers that must flip are A (cos<0.99 + different top-1) and C (donor top1/top5 overlap >0). Do not tune WSP, do not re-export over the canonical file, and do not claim coherence from B alone. WSP campaign #900 remains paused until the gate returns exit 0.

Confidence: HIGH — thresholds are the literal #926 criteria, all inputs are measured with cited file:line, baseline FAIL is reproduced across #907/#916/#925, and the gate's discriminating blockers match the independently established operator-level defect. Single most important caveat: criterion B is a weak proxy (documented above); treat the conjugate A+C+B text inspection as the coherence judgement.
