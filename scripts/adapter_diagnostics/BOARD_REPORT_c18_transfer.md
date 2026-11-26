@coordinator [C18-SHEPHERD #911 OWNER TASK 4 — FUNCTIONAL TRANSFER TEST: VERDICT FALSIFIED]

## 0. WHAT WAS TESTED (and what was not)
Claim under test (#894/#895): "donor weights transferred" into the 815M custom checkpoint, evidenced by 530/530 strict keys + donor fraction 0.9072. Those are PROVENANCE measures, not COMPUTATION measures. I ran the missing FUNCTIONAL test.

SUBJECTS (read-only; no model/checkpoint/config file edited):
- DONOR: clean Qwen2.5-0.5B-Instruct HF snapshot `C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775` (the exact MINI_SNAP used by `scripts\verify_mighty_mouse.py:35-38`). Cross-checked against the export donor tensor file `C:\02_QUILLAN\exports\donor_cache_v23\qwen_0.5b_donors.pt`.
- CUSTOM: `C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158.safetensors` (1,609,018,260 B, sha256 53e60d33...ac7a5c, 815,396,270 params, strict load 530/530 via `scripts\probe_longctx_quality.py:build_real_model`).
- SAME 5 fixed prompts to both: the 4 strings from #906/#907 plus one 11-token control. Full token-for-token capture via forward hooks.

NEW FILES ONLY under `C:\02_QUILLAN\scripts\adapter_diagnostics\`: `transfer_verification.py`, `addendum_confirm.py`, `rescale_control.py`, results JSONs (`transfer_verification_results.json`, `addendum_confirm_results.json`, `rescale_control_results.json`), `_acts\*.pt` activation caches, run logs.

## 1. COMPARISON VALIDITY (stated explicitly, no silent comparisons)
- **VALID — input embedding**: donor `embed_tokens` vs custom `ingestion.txt_emb`, both [L,896], same 151936-token Qwen vocab. Purest provenance test.
- **VALID — final hidden**: custom decoder input vs donor last hidden, both 896-wide. Dimension-valid; NOT a layer-identity claim (custom graph is a decomposition/MoE stack, not 24 transformer layers).
- **VALID — logits**: both [L,151936] over identical vocab. Cosine, top-k overlap (chance = k^2/151936), donor-top1 rank.
- **VALID (with caveat) — linear CKA**: 896-wide activations, tokens=samples. Measures geometry only, never functional equivalence of different operators.
- **INVALID — no layer-to-layer correspondence**: the custom modules (decomposition / couil_attn / expert_swarms / txt_dec) have no donor analogue; I did not fabricate a mapping. Any "custom layer k = donor layer k" claim would be unsound and I did not make one.

## 2. EXACT MEASURED NUMBERS

### 2.1 Donor embedding physically arrived — CONFIRMED (weight-level)
| Metric | Value |
|---|---|
| donor `embed_tokens` shape | [151936, 896] |
| cosine(donor embed, custom `ingestion.txt_emb`) | **1.0** (per-token, all 5 prompts) |
| cosine(donor embed, custom `txt_dec`) | **1.0** |
| donor embed mean-abs | 0.011981 |
| custom txt_emb mean-abs | 0.011981 |
| file dtype of custom txt_emb | F16 |

### 2.2 Final hidden vs donor — NO functional alignment
- Cosine(custom final hidden, donor layer-24 hidden) per prompt: **-0.092114, -0.113882, -0.099349, -0.122004, -0.125495** (mean ≈ -0.11; i.e. slightly ANTI-aligned, not merely different).
- Linear CKA of pooled custom final hidden vs all 25 donor layers: max **0.323531** at donor layer 22; plateau **≈0.312** across donor layers 3-21; 0.156345 at layer 0. (A model with itself gives CKA = 1.0.)

### 2.3 Logits / output distribution — NO correlation, actively ANTI-correlated
- Cosine(custom logits, donor logits) per prompt: **-0.682114, -0.541822, -0.637920, -0.475743, -0.540149** (all negative).
- Top-k overlap (all 5 prompts, every token position): **top-1 = 0.0, top-5 = 0.0, top-10 = 0.0, top-20 = 0.0**. Chance baselines: 0.0000066 / 0.000165 / 0.000658 / 0.002633. Zero overlap is BELOW chance at every k.
- Rank of donor top-1 token inside custom distribution (last position): `' Paris'` → **148,861**; `' '` → **151,936**; `' Alex'` → **150,041**; `' state'` → **148,602**; `' I'` → **151,929** (out of 151,936). I.e. the donor's correct answer sits in the bottom ~2% of the custom ranking.
- Mean Spearman rank correlation (custom vs donor logits): **-0.243023** (per prompt: -0.358099, -0.322874, -0.315970, -0.094407, -0.123763).

### 2.4 Prompt sensitivity — custom is length-driven, not content-driven
Pairwise cosine between final-position representations of different prompts (1.0 = collapse):
| Stage | custom | donor control |
|---|---|---|
| raw embedding lookup | 0.140889 | — |
| ingestion | 0.690204 | — |
| decomposition | 0.907787 | — |
| diffusion_core | 0.750587 | — |
| moe | 0.741045 | — |
| final_norm | 0.850568 | 0.540756 |
| logits | **0.899465** | 0.609031 |
Smoking gun: two semantically different **5-token** prompts ("The capital of France is" vs "Hello, my name is") give custom logits cosine **0.999996**, max abs logit diff **2.7e-05** (donor: 0.782233). Same-length "2 + 2 =" also 5 tokens → 0.999934. A 5-token vs 11-token prompt differs (0.873392), proving the custom output is a function of **sequence length only**, not token content. This independently reproduces #907's "different prompts → identical logits."

### 2.5 FIRST DIVERGENCE POINT
Cosine of each custom stage to the donor embedding (starts at 1.0 because the tables are identical):
`emb_lookup 1.0 → ingestion -0.018354 → decomposition -0.006192 → diffusion_core -0.030930 → moe -0.018654 → final_norm -0.038869`.

**The first divergence is `InputIngestionLayer.forward`, `C:\02_QUILLAN\03 - Training & Model\src_quillan\quillan_v8_saturated.py:237-239`** (`x = x + pe` then `x_norm = self.norm(x)`). Root magnitude: checkpoint `ingestion.pos_embed` shape [1,1024,896], mean-abs **0.622575**, max **1.0**, cosine vs canonical sinusoidal **0.9999553** (it IS the adapter's structural sinusoidal table, `quillan_weight_adapter.py:601-602`, NOT donor-derived); ratio to txt_emb mean-abs = **51.96x**. Prompt discrimination jumps 0.140889 → 0.690204 in this one module.

But the deeper, NON-scale divergence follows: even with the scale confound fully removed, prompt discrimination remains 0.851438 at decomposition (see 2.6) — the graph itself destroys content.

### 2.6 DECISIVE CONTROL — pos_embed rescaled in memory (no file edits)
Scale = 0.011981/0.622575 = **0.019244** applied in memory to `model.ingestion.pos_embed`; identical prompts re-run:
- logits cosine vs donor improves from **-0.54..-0.68 → +0.051385..+0.537798** (so ~half the divergence WAS scale-driven; I am reporting this honestly, not hiding the improvement).
- top-1 overlap **still 0.0**, top-5 **still 0.0** on all 5 prompts. top-20 mean improves 0.0 → 1.6 (chance 0.002633); donor top-1 ranks improve to 12,180 / 14 / 5,824 / 125,500 / 109 — better, but only 1 of 5 lands in top-20.
- final hidden cosine vs donor still only **0.093118..0.170817**; CKA peaks at donor **layer 0 = 0.670476** with middle layers ~**0.1798** — i.e. after scale correction the custom's final state resembles the donor's INPUT embedding, not any of its 24 computed layers.
- prompt-discrimination after rescale: ingestion 0.592044, decomposition 0.851438, final_norm 0.792306, logits 0.345805.

Interpretation: rescaling removes the pos_embed magnitude problem and restores a weak, non-functional token-identity signal, but it does NOT restore the donor's computation. The remaining failure is architectural/semantic (shape-compatible donor tensors sitting in operators whose surrounding Q/K/V/routing are structural), which is c28-calculus's and c10-codeweaver's owner scope; my functional test locates the consequence, not the fix.

## 3. VERDICT
**FUNCTIONAL TRANSFER: FALSIFIED.** (Provenance: CONFIRMED — the donor embedding is physically present, cosine 1.0.)

The checkpoint carries the donor's NUMBERS in shape-compatible slots, not the donor's COMPUTATION. Before any scale correction the output distribution is anti-correlated with the donor (cosine -0.48..-0.68, Spearman -0.243, top-k overlap exactly 0 at k=1/5/10/20, donor top-1 ranked ~148k-151k of 151,936) and is prompt-blind by length. After removing the pos_embed scale confound, correlation becomes weakly positive but top-1/top-5 overlap remains exactly 0 and the final state aligns with donor layer 0 (input), not donor layers 3-24. Neither result supports "the 0.9072 donor fraction verifies knowledge transfer." That metric must be downgraded to PROVENANCE ONLY in all docs/ADRs.

CAVEATS (single most important): my donor is the HF Instruct snapshot certified by `verify_mighty_mouse.py`; the export donor tensor file's `embed_tokens` matched the custom table at cosine 1.0, cross-validating donor identity. The custom graph is architecturally non-isomorphic to Qwen, so CKA/cosine at "final hidden" measures geometry, not layer equivalence — I marked that INVALID as an equivalence claim. I did not test the GGUF path (`exports\clean-qwen-0.5b-q8_0.gguf`): GGUF inference was not needed because the HF snapshot is the certified export source; using it would add quantizer noise without changing the verdict. No model, checkpoint, tokenizer, config, or gate was modified.

Evidence files: `C:\02_QUILLAN\scripts\adapter_diagnostics\transfer_verification_results.json`, `addendum_confirm_results.json`, `rescale_control_results.json` (plus the three diagnostic scripts and `_acts\` activation caches).
