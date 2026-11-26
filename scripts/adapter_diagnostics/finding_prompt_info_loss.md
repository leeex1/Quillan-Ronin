@coordinator — C10-CODEWEAVER finding for brief #911 (OWNER TASK: exact point where prompt info is lost)

## ANSWER (lead)
Prompt information dies at the Couil attention input projections, which emit EXACT ZERO because `_weight_quant` annihilates the adapter's mean-averaged donor weights. The main attention tower (30 heads) computes nothing. A second, compounding defect is the pos_embed/txt_emb 52x scale imbalance that buries the surviving token signal.

PRIMARY CULPRIT (exact):
  C:\02_QUILLAN\03 - Training & Model\quillan_v8_saturated.py:156-163  -- function `_weight_quant`
  * line 159: `... .clamp(min=eps)`  (eps=0.01, set at line 188 `self.eps = 0.01` in BitLinear.__init__)
  * line 162: `w_q = torch.round(torch.clamp(w_scaled, -1.0, 1.0))`
  * line 163: `return w + (w_q * scale - w).detach()`  (STE forward => returns w_q*scale)
  Mechanism: for any weight block with mean-abs < eps/2 = 0.005, scale is clamped to 0.01, so max|w/scale| < 0.5, round() maps EVERY element to 0, and the STE returns an all-zero tensor. The adapter's `_layer_mean` (quillan_weight_adapter.py:240-260) averages 24 donor layers with cancellation, shrinking attention/FFN/prism weights to ~0.0001-0.003 mean-abs -- below the kill threshold. Consequence: q_proj, k_proj, v_proj, o_proj, diffusion_core.ffn, and all 9 decomposition prism vectors become zero-weight linear layers => their outputs are EXACTLY 0 for any input => the 30-head attention and the diffusion FFN compute nothing.

SECONDARY / COMPOUNDING CULPRIT (exact):
  C:\02_QUILLAN\03 - Training & Model\quillan_v8_saturated.py:448-449  -- `pe = self.position_embeddings(...)`; `x = x + pe`
  Origin: quillan_weight_adapter.py:601-602 + :634-647 (_sinusoidal_positions injects a full-amplitude sin/cos table).
  Measured pos_embed mean-abs 0.62258 vs txt_emb 0.01198 = 51.96x. After LayerNorm the token-dependent term is ~2% of the signal, so the surviving residual stream is position-dominated: two prompts ending in the SAME token produce near-identical last-position logits.

## MEASURED EVIDENCE (exact stdout, real 815M checkpoint)
Script: scripts/adapter_diagnostics/diag_prompt_info_loss.py (Python 3.11, torch 2.12.0+cpu)
  MODEL params=815,396,270 strict_load=OK
  PROMPT_A 'The capital of France is' -> ids=[785, 6722, 315, 9625, 374]
  PROMPT_B 'Hello, my name is'       -> ids=[9707, 11, 847, 829, 374]
  EMBED_NORMS txt_emb_mean_abs=0.01198 pos_embed_mean_abs=0.62258 ratio=51.96x

Boundary comparison (A vs B, full tensor): cos_sim / max_abs
  01_txt_emb_OUT            0.181261 / 8.837891e-02   (tokens DO reach embedding; strongly differ)
  08_INGESTION_OUT          0.999028 / 7.004966e-02   (diluted ~52x)
  13_prism_*_OUT (all 9)    0.000000 / 0.000000e+00   <-- EXACTLY ZERO (dead)
  21_q_proj_OUT             0.000000 / 0.000000e+00   <-- EXACTLY ZERO (dead)
  22_k_proj_OUT             0.000000 / 0.000000e+00   <-- EXACTLY ZERO (dead)
  23_v_proj_OUT             0.000000 / 0.000000e+00   <-- EXACTLY ZERO (dead)
  24_o_proj_OUT             0.000000 / 0.000000e+00   <-- EXACTLY ZERO (dead)
  25_COUIL_ATTN_OUT         0.000000 / 0.000000e+00   <-- attention output ZERO
  26_diff_ffn_OUT           0.000000 / 0.000000e+00   <-- diffusion FFN ZERO
  27_DIFFUSION_OUT          0.999105 / 5.658665e+00   (residual only; attn/ffn add nothing)
  45_TXTDEC_LOGITS          0.998864 / 5.057163e+00
Last-position comparison (the token position that drives argmax):
  46_FINAL_LOGITS           0.999996 / 2.670288e-05 / 100.00% elements <1e-4
  top1(A)=33550 top1(B)=33550 (both `BOSE`)
  top5(A)=top5(B)=[BOSE 35.267, honda 32.913, esl 32.101, 若要 31.739, Fotos 31.439]
  -> reproduces brief #907 symptom exactly (100% identical last-position logits).

Direct proof of the zeroing mechanism (probe_zero_outputs.py):
  W diffusion_core.couil_attn.q_proj.weight  mean_abs=2.748819e-03 max_abs=4.084839e-03 all_zero=False
  q_proj(random x) OUTPUT    mean_abs=0.000000e+00 max_abs=0.000000e+00 all_zero=True
  F.linear(x, raw_w)         mean_abs=8.120438e-02 max_abs=4.019395e-01 all_zero=False
  _weight_quant(w)           mean_abs=0.000000e+00 max_abs=0.000000e+00 all_zero=True
  => nonzero weight, but `_weight_quant` turns it into all-zero; raw F.linear works.

Control -- pos_embed *= 0.02 in memory (no files edited):
  logits(A) vs logits(B): max_abs=8.590772e+01 mean_abs=1.274408e+01 cos=0.301707
  => confirms the 52x scale imbalance is a real, separate contributor to prompt-insensitivity.
Control -- BitLinear eps -> 1e-5 (probe_weight_quant_rescue.py):
  q_proj _weight_quant(eps=1e-5): mean_abs=1.849769e-03 all_zero=False  (weights restored)
  BUT last-position logits still 100% <1e-4, top1 still BOSE.
  => fixing eps ALONE is insufficient; pos_embed scale must ALSO be fixed. Both defects are load-bearing.

## TOKEN-ID USAGE VERDICT
Forward DOES read token IDs: txt_emb(A) vs txt_emb(B) cos=0.181261, max_abs=8.84e-2 (strongly different). No cached/fixed-path bug; no `torch.zeros`-style token bypass. A constant-token run differs from prompt A by max_abs=2.41. The loss is downstream (zeroed attention weights + position-swamped residual), not at ingestion input.

## FILES CREATED (NEW only; no model/adapter source edited)
  scripts/adapter_diagnostics/diag_prompt_info_loss.py         (full boundary capture; 46 boundaries)
  scripts/adapter_diagnostics/diag_prompt_info_loss_stdout.txt (exact stdout)
  scripts/adapter_diagnostics/prompt_info_loss_result.json     (machine-readable)
  scripts/adapter_diagnostics/probe_zero_outputs.py            (_weight_quant zeroing proof)
  scripts/adapter_diagnostics/probe_weight_quant_rescue.py     (eps rescue control)
  scripts/adapter_diagnostics/finding_prompt_info_loss.md      (this report)

## RECOMMENDED FIX (not applied -- out of scope)
  1. _weight_quant: use a relative scale that is never clamped above the block magnitude (e.g. clamp(min=1e-8) or scale = mean_abs with no eps floor; or per-tensor absmean quant with STE that preserves small weights). This stops attention/diffusion/prism weights from becoming zero.
  2. pos_embed: scale the injected sinusoidal table to match donor token-embedding magnitude (~0.012 mean-abs, i.e. multiply by ~0.02) in quillan_weight_adapter.py:601-602, and/or normalize embeddings before addition in quillan_v8_saturated.py:448-449.
  3. Re-run probe_generate_sanity.py after both fixes; require distinct last-position logits and coherent top-5 before any quality claim.

Confidence: I know (measured, reproducible) that q/k/v/o outputs are exactly zero and that `_weight_quant` is the direct cause; I know the pos scale is 51.96x. I believe (from rescue controls) both must be fixed for coherent generation; a full retrain/re-export may still be needed because the weights were structurally mismatched into a different graph.
