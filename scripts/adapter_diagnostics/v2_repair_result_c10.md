@coordinator — C10-CODEWEAVER / V2 REPAIR RESULT (spec #919)

## VERDICT (lead)
GATE FAILED. STRUCTURAL BLOCKER CONFIRMED. NO re-export. NO WSP tuning. NO source/adapter edits.
The accepted checkpoint is untouched: sha256 53E60D3339C0AD7FBB6FF0607970F7DE80493350F86B6C6024BA1D8DAEAC7A5C (verified post-run). No `quillan-mini-sovereign-bitnet158-v2.safetensors` was created. I did not claim success.

## WHAT I DID
Pure in-memory experiment (HARD RULE 1): loaded the real 815,396,270-param checkpoint, applied A+B+C to the in-memory model, then measured the gate criteria. No files other than NEW diagnostic scripts were written.

- [A] Set eps=1e-12 on all 24 BitLinear modules and cleared `_w_quant_cache` (BitNet 1.58b absmean: eps is a div-guard, never a magnitude floor). PROOF it worked: `q_proj` weight mean_abs=2.748819e-03, `_weight_quant` output mean_abs=1.849769e-03, `q_proj(x)` output mean_abs=5.403278e-02 (baseline: exactly 0). The dead towers are genuinely revived.
- [B] `ingestion.pos_embed *= 0.01924392`. mean_abs 0.62258 -> 0.01198 (matches txt_emb 0.01198).
- [C] `ingestion.norm.weight.copy_(final_norm.weight)`. mean_abs 0.06953 -> 7.13935 (standard LayerNorm scale).

## EXACT STDOUT — in-memory A+B+C test (`scripts/adapter_diagnostics/v2_repair_memory_test.py`)
BASELINE last-logit A/B: max=2.670288e-05 mean=4.496570e-06 pct<1e-4=100.00% cos=0.999996
  top1 A=33550 B=33550
[A] eps -> 1e-12 on 24 BitLinear modules; caches cleared
[B] pos_embed mean_abs 0.62258 -> 0.01198 (x0.01924392)
[C] ingestion.norm mean_abs 0.06953 -> 7.13935 (ref final_norm 7.13935)
[A proof] q_proj weight mean_abs=2.748819e-03 quant=1.849769e-03 q_proj(x) out_mean_abs=5.403278e-02 (baseline was exactly 0)
POST-FIX last-logit A/B: max=7.629395e-04 mean=1.261473e-04 pct<1e-4=47.32% cos=1.000003
  top1 A=13 B=13
  top5(A): [('.', 57.19), ('1', 56.26), (' ', 55.64), ('2', 54.26), (',', 53.87)]
  top5(B): [('.', 57.19), ('1', 56.26), (' ', 55.64), ('2', 54.26), (',', 53.87)]
=== GENERATION (post-fix, greedy 10 tokens) ===
  'The capital of France is' -> gen='., , , , ,' alpha_ratio=0.0000
  '2 + 2 =' -> gen='1212121212' alpha_ratio=0.0000
  'Hello, my name is' -> gen='., , , , ,' alpha_ratio=0.0000
  'Quillan-Ronin is a sovereign' -> gen=' sovere.§,www,DB,www,DB,www,DB,www,DB' alpha_ratio=0.7222
  'The quick brown fox jumps over the lazy dog and then' -> gen='., , , , ,' alpha_ratio=0.0000
COHERENCE distinct_generations=3/5 mean_alpha_ratio=0.1444
=== DONOR TOP-K OVERLAP (last position) ===
  'The capital of France is': top1=0/top1 (chance 0.00001) top5=0/top5 (chance 0.00016) top10=1/top10 (chance 0.00066) top20=3/top20 (chance 0.00263)
  '2 + 2 =': top1=0/top1 top5=0/top5 top10=1/top10 top20=2/top20
  'Hello, my name is': top1=0/top1 top5=0/top5 top10=0/top10 top20=0/top20
  'Quillan-Ronin is a sovereign': top1=0/top1 top5=0/top5 top10=0/top10 top20=0/top20
  'The quick brown fox jumps over the lazy dog and then': top1=0/top1 top5=0/top5 top10=1/top10 top20=2/top20
DONOR OVERLAP mean top1=0.000 top5=0.000
GATE RESULT: FAIL — logits still prompt-blind; donor top-k overlap still exactly 0

HONESTY NOTE ON THE `coherent_english=True` FLAG IN THAT SCRIPT: it is a false positive of my crude alpha_ratio>=0.10 threshold. 4/5 prompts produce pure digits/punctuation (alpha_ratio 0.0); the single non-zero case is `'www,DB'` repeated 4x — repetitive fragments, NOT coherent English. The real coherence verdict is NO. I am overriding the automated flag rather than hiding behind it.

## EXACT STDOUT — A/B/C ablation (`scripts/adapter_diagnostics/v2_repair_ablation.py`)
Loads checkpoint once; 8 in-memory variants; columns: maxdiff/meandiff/cos between the two 5-token PROSE prompts, donor top1/top5 overlap at last position, q_proj output mean_abs.
variant        maxdiff    meandiff       cos   top1A   top1B  dtop1  dtop5        qout
NONE        2.6703e-05  4.4966e-06   1.00000   33550   33550      0      0  0.0000e+00
A           4.5776e-05  5.7725e-06   1.00000   33550   33550      0      0  5.6274e-02
B           4.1580e-04  6.4248e-05   1.00000      16      16      0      0  0.0000e+00
C           4.1008e-05  6.1641e-06   0.99999  145216  145216      0      0  0.0000e+00
A+B         4.5013e-04  6.9765e-05   1.00000      16      16      0      0  5.4154e-02
A+C         1.0014e-04  1.5821e-05   1.00000  145216  145216      0      0  5.4344e-02
B+C         7.4387e-04  1.2197e-04   1.00000      13      13      0      0  0.0000e+00
A+B+C       7.6294e-04  1.2615e-04   1.00000      13      13      0      0  5.5313e-02

## PRECISE RESIDUAL BLOCKER
1. Defect A is real and is now demonstrably fixed in memory (q_proj 0 -> 5.4e-2). But A ALONE leaves the two prompts with identical top1=33550 and 4.6e-5 max logit diff — confirms c10's #914 caveat: eps fix alone is insufficient.
2. Defect B is real and is fixed in memory (52x -> 1x scale). But B alone leaves maxdiff 4.16e-4, cos 1.00000, donor top1/top5 = 0. (Forge's 9.34x "prompt sensitivity" measured a 5-token PROSE vs a digit prompt; the hard case here is two same-length prose prompts, which stay collapsed.)
3. CRITICAL NEW FINDING: reviving the dead towers (A) with the adapter's mean-averaged donor weights does NOT restore prompt content — it injects a large, prompt-INVARIANT bias. Post-fix top-1 flips from 'BOSE' (33550) to '.' (13) with logits ~57, and the two prompts' top-5 become byte-identical. A+B is no better than B. This is the mechanistic bridge to c28-calculus's #912 structural proof: no RoPE in target; donor 14x64 vs target 30x34 (gcd=2, no partition map); GQA 2 KV heads expanded to 30 independent; 24 donor layers averaged into one E[W_l]; lm_head fed final_norm(x) from a different graph. Those are properties of the OPERATOR, not the weights — weight repair cannot fix them.
4. c18-shepherd's #916 rescale control independently agrees: after pos scale correction, final-state CKA peaks at DONOR LAYER 0 (input embedding), middle layers ~0.18, top-1/top-5 overlap exactly 0. The custom retains a token-identity echo, not the donor's 24-layer computation.

Therefore the V2 weight-level repair CANNOT produce coherent, prompt-sensitive, donor-aligned generation. Escalation to architecture change (implement RoPE + donor head geometry + per-layer depth, then re-map) or weight distillation onto the target geometry is required, exactly as #912/#917 anticipated. This is NOT a WSP problem and NOT a donor-choice problem.

## HARD-RULE COMPLIANCE
- HARD RULE 1: tested in memory first; gate failed -> did NOT re-export.
- HARD RULE 2: canonical checkpoint sha unchanged (53E60D33...A5C); no v2 file written.
- HARD RULE 3: N/A — no re-export, so post-export probes were not run. Report this explicitly; do not imply they were.
- HARD RULE 4: STOP invoked. No success claim, no WSP tuning, escalated to coordinator.
- HARD RULE 5: NO source/adapter edits were made (a failed gate does not warrant changing the behavior of the accepted checkpoint in place). All 3 regression suites still pass on the unmodified tree, exact stdout:
  * `CUSTOM_ARCH_TRANSFER_TEST PASS | keys=530 donor_keys=38 structural_keys=492 donor_fraction=0.3664 donor_elems=986944 total_elems=2693857 logits=(1, 6, 256) deterministic=True strict=True` (exit 0)
  * `SUMMARY: 6/6 checks passed` (test_arch_generation_path.py, exit 0; checks A-F all PASS)
  * `ALL_WSP_GATE_TESTS_PASSED` (exit 0)
  (Note: an earlier run with PowerShell `2>&1` showed exit 1 because stderr logs were wrapped as ErrorRecords; re-run with `cmd /c` gives true exit 0 for all three.)

## FILES CREATED (NEW only; no model/adapter source touched)
  scripts/adapter_diagnostics/v2_repair_memory_test.py       (+ _stdout.txt)
  scripts/adapter_diagnostics/v2_repair_ablation.py          (+ _stdout.txt)
  scripts/adapter_diagnostics/v2_repair_result_c10.md        (this report)

Confidence: HIGH (measured, reproducible, 220.7s + 64.1s runs) that A+B+C applied in memory does NOT restore prompt sensitivity or donor top-k alignment, and that output remains degenerate. The eps and pos-scale defects are confirmed real but NOT sufficient. Structural incompatibility is the residual blocker. No fake PASS.
