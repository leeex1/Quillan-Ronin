@coordinator [FORGE #911 // EMBEDDING-SCALE ROOT CAUSE + IN-MEMORY FIX TESTS — owner item 2]

SCOPE: read-only. NEW files only under scripts\adapter_diagnostics\. No weights saved, no checkpoint rebuilt, no model source edited. Subject: exports\quillan-mini-sovereign-bitnet158.safetensors (815,396,270 params; strict load 530/530; sha 53e60d33...). Artifacts: forge_embedding_scale_audit.py, embedding_scale_audit_results.json, embedding_scale_audit_report.md.

1) EXACT SCALE FINGERPRINT
- ingestion.txt_emb.weight [151936,896]: mean-abs 0.01198079, std 0.01524947, min -0.20117188, max 0.16601562, L2 175.05.
- ingestion.pos_embed [1,1024,896]: mean-abs 0.62257522, std 0.66908747, rms 0.70710790, mean 0.22874345, min -1.0, max 1.0, L2 677.34.
- RATIO pos_embed / txt_emb = 51.9645x (reproduces #907's ~52x to 4 digits).
- txt_dec.weight [151936,896]: mean-abs 0.01198079 — EXACTLY tied to txt_emb (max_abs diff 0.0). The output head carries the same tiny donor scale.
- ingestion.mod_emb.weight [4,896]: mean-abs 0.00857835, min/max +/-0.01242716.
- 18 LayerNorm tensors: 7 donor-final-norm clones (final_norm, norm_decomp, norm_diff, norm_moe, pre_final_norm, moe.output_norm, diffusion_core.norm2, diffusion_core.couil_attn.norm) all mean-abs 7.13934755, min -0.01141357, max 16.75; ingestion.norm.weight mean-abs 0.06952981, min -0.44726562, max 0.7265625. All LN biases exactly 0. ingest_bypass scalar = 2.0.
- 454 low-rank/swarm tensors, structurally zero (lora_A/lora_B etc. mean-abs 0.0).
- Raw packed scales: 24 int8 tensors / 24 bitnet_scale gammas; gamma min 0.00045243, mean 0.00285670, max 0.01242716.

2) WHY pos_embed IS 52x LARGER — ROOT CAUSE
NOT random init. `03 - Training & Model/quillan_weight_adapter.py:600-602` explicitly declares `ingestion.pos_embed` a STRUCTURAL CONSTANT and assigns `_sinusoidal_positions(shape)` (sin/cos on [-1,1]). The checkpoint tensor matches that exact formula to max_abs_diff 0.00024414 = 2^-12 (fp16 ULP near 1.0), mean_abs_diff 0.00009016 — bit-identical up to serialization rounding. An untouched `nn.Parameter(torch.randn(...))` (quillan_v8_saturated.py:351) would have mean-abs sqrt(2/pi)=0.797885, which is NOT stored (0.62257522 is the sinusoid's 0.62257). The donor Qwen2.5-0.5B uses RoPE and has NO learned position key (verified donor_pos_like_keys=[]), so the adapter manufactured a full-magnitude additive position table for a graph whose donor weights were never trained against one. Key had no donor analogue -> became a structural constant at native [-1,1] scale while all knowledge tensors kept donor scale ~0.012.

MECHANISM: quillan_v8_saturated.py:448-449 `x = x + pe`. All 3 probes are 5 tokens. The two prose prompts produce max_abs_logit_delta 0.000027 (numerically identical — reproduces #906/#907) while "2 + 2 =" gives 1.009913; the digit prompt has ~2x larger summed token-embedding L2 (1.9629 vs 0.9541 / 1.1074). Sensitivity tracks embedding magnitude, not semantics: that is the signature of position dominance.

3) IN-MEMORY FIX TESTS (3 prompts, 5 greedy tokens, no save; exact scale-match multiplier 0.01924392)
- zero: max_abs_logit_delta 11.842144 (highest); top-1 ' ' / '1'; gen ' , , ' / '12121'
- scale_exact_match (x0.01924392): 9.343369; gen '12121' / '_2121'
- scale_0.02 (#907's test): 9.203515; gen '12121' / '_2121'
- from_donor (donor embed rows, mean-abs 0.01232777; donor has no learned pos key): 8.725109; gen ' , , ' / '121 ,'
- baseline: 1.009913; gen 'BOSE honda esl Fotos erotica'
- sinusoidal_fresh: 1.009881; gen identical to baseline — confirms stored table IS the adapter sinusoid.

4) SINGLE MOST LIKELY FIX
For the SCALE defect: zero the additive `ingestion.pos_embed`, or (preferred, preserves positional order) multiply by exact scale-match 0.01924392. Both restore prompt sensitivity 9-12x. I recommend exact scale-match, because zeroing makes the model position-invariant and is not viable for an LM.
CRITICAL CAVEAT: NONE of the four fixes produces coherent English — generated_alpha_ratio 0.0 for every scaled/zero/donor variant, digits/punctuation only. The 52x scale mismatch explains prompt-BLINDNESS but NOT incoherence. Coherence requires correct donor->target semantic mapping plus a CHECKPOINT REBUILD, outside my mandate. HAND OFF to c28-calculus (semantic mapping) and c10-codeweaver (data-flow trace). Do not deploy this checkpoint; do not tune WSP on it.
SECONDARY ANOMALY for c28: ~103x discontinuity between ingestion.norm (0.0695) and all other LayerNorms (7.1393) at the first normalization.

CONFIDENCE: HIGH on provenance, all scale numbers and prompt-sensitivity (directly measured, reproduces #907); HIGH that pos scale is necessary-but-insufficient; LOW on which semantic remap restores coherence (not tested). elapsed 291.73s, Zero-Leak Audit Law enforced throughout.
