@coordinator — C10-CODEWEAVER GAMMA-CONVENTION RESOLUTION (blocker #930 + directive #931)

## LEAD ANSWER
The target runtime consumes CONVENTION 1: a SINGLE absmean scale over dims [-2,-1] (whole 2D matrix; per-leading-slice for 3D MoE), clamped at eps=0.01, recomputed from the stored weight on every forward. It does NOT consume the .bitnet_scale sidecar as a scale, and it has NO per-channel, per-128 or per-256 granularity. My pooling MUST produce that whole-matrix absmean with the 0.01 floor applied. The per-channel quantize_to_ternary in build_coherent_ternary_base.py is NOT runtime-compatible and I will not adopt it.

## 1. ACTUAL INGESTION PATH (read-only trace)

Checkpoint store (accepted file, sha 53E60D33...A5C): safetensors header 553 keys, dtypes {F32:324, F16:206, I8:24}, 24 `.bitnet_scale` sidecars. The 24 I8 tensors are the BitLinear knowledge weights.

Load/dequant — `09 - Projects/mcp/quillan_serve/engine.py:178-195` (identical in `scripts/probe_longctx_quality.py:143-157`, `scripts/verify_mighty_mouse.py:62-78`):
    scales = {k[:-len('.bitnet_scale')]: v for k,v in packed.items() if k.endswith('.bitnet_scale')}
    sd[k] = value.float() * scales[k].float()      # w = q * gamma_store
Then `model.load_state_dict(sd, strict=True)` at engine.py:219 into `QuillanRoninSovereign` from `03 - Training & Model/quillan_v8_saturated.py`.

Runtime quantizer — `quillan_v8_saturated.py:156-163`:
    scale = w.abs().mean(dim=[-2,-1] if w.dim()>=2 else -1, keepdim=True).clamp(min=eps)   # eps=0.01
    w_q   = round(clamp(w/scale, -1, 1))
    return w + (w_q*scale - w).detach()             # forward value == w_q * scale
Called from BitLinear._forward_impl at :206-212 and from MoE at :696-698 (`_weight_quant(self.w1/wgate/w2)`, 3D -> per-expert scale).

Therefore the sidecar gamma is ONLY a dequantization multiplier used to build w. The scale actually consumed by the operator is recomputed as `clamp(mean(|w|) over [-2,-1], 0.01)`. Since w = q*gamma_store and q in {-1,0,1}, that recomputed scale = `clamp(gamma_store * rho, 0.01)` where rho = fraction of nonzero codes. `gamma_store` is never used as the operator scale.

MEASURED ON THE REAL CHECKPOINT (probe script, read-only, no file edits) — 6 of 24 sidecars:
  decomposition.K_gate.weight  gamma_store=0.000674758 rho=0.1359 mean|w|=9.17074e-05 rt_scale=0.01 fwd==w? False rel_err=1
  decomposition.Q_gate.weight  gamma_store=0.00408256  rho=0.6731 mean|w|=0.00274784 rt_scale=0.01 fwd==w? False rel_err=1
  decomposition.V_gate.weight  gamma_store=0.000452434 rho=0.1361 mean|w|=6.15952e-05 rt_scale=0.01 fwd==w? False rel_err=1
  decomposition.W_gate.weight  gamma_store=0.0028962   rho=0.6775 mean|w|=0.00196221 rt_scale=0.01 fwd==w? False rel_err=1
  (+ vectors.Constraint / vectors.Context identical)
Identity `mean|w| == gamma_store * rho` holds exactly in every case. All gamma_store < 0.01, so the runtime eps floor fires on all 24 and the forward value is q*0.01, NOT q*gamma_store. This is a measured, load-bearing convention mismatch, not a hypothesis.

## 2. CONVENTIONS IN-REPO — NOT INTERCHANGEABLE

| # | Location | Group | Formula | Runtime consumer? |
|---|----------|-------|---------|-------------------|
| 1a | quillan_v8_saturated.py:156-163 | whole 2D matrix [-2,-1] (per-expert 3D) | mean(|W|).clamp(min=0.01) | YES — this is the only one the operator uses |
| 1b | quillan_weight_adapter.py:100-114 `_absmean_ternarize` | whole tensor (no dim) | mean(|W|).clamp(min=1e-6) | Numerically SAME value as 1a for 2D; eps floor differs (1e-6 vs 0.01) |
| 1c | 03 - Training & Model/scripts/quillan_weight_adapter.py:32-37 `_quantize_to_ternary` | [-2,-1] keepdim | mean(|W|).clamp(min=0.01) | EXACT match to runtime; existing reusable function |
| 2 | v23_stage_d_deep_wsp.py:109-121 | per-256 row block | gamma*=<W_b,T_b>/(<T_b,T_b>+1e-8) | NO — staging/Hessian-compensation only |
| 3 | CAT-Q per-128 alpha + learnable modulation | per-128 | external (no in-repo runtime reader) | NO — no runtime consumer at all |
| 4 | build_coherent_ternary_base.py:49-56 `quantize_to_ternary` | PER-CHANNEL dim=-1 keepdim | mean(|W[:,i]|).clamp(min=1e-5) | NO — per-row scale; runtime has no per-row granularity |

Direct source check: grep for `bitnet_scale|CAT-Q|cat_q|gamma|alpha` in `quillan_v8_saturated.py` returns ZERO matches — the target model has no scale-sidecar or grouped-scale machinery whatsoever.

## 3. CAN THEY BE RECONCILED?
NOT as stored. A gamma from convention 2 (per-256 LS*) or 4 (per-channel) or 3 (per-128 alpha) cannot be handed to the runtime: `_weight_quant` will ignore it and recompute a single whole-matrix absmean. Folding those scales into w does not fix it either, because the folded w's whole-matrix absmean is a new number unrelated to any of the three source gammas.

NO MODEL SOURCE EDIT IS NEEDED for the adapter fix. `_weight_quant` already implements a coherent (if coarse) convention; the defect is upstream: the adapter pooled raw floats, then the packer stored a gamma computed from the shrunken mean, and the runtime floor turned it into q*0.01. The fix belongs entirely in the adapter pooling/emission path. A model edit would only be warranted if we wanted true per-128/256 grouped quantization at runtime — explicitly out of scope and forbidden here.

## 4. WHAT MY POOLING MUST PRODUCE (exact)
For a merged 2D tensor with codes q in {-1,0,1} and intended scale gamma:
    emit w = q * (max(gamma, 0.01) / rho),  rho = mean(|q|).clamp(min=eps)
Then:
    mean(|w|) = max(gamma, 0.01)                -> runtime scale = clamp(mean|w|, 0.01) = max(gamma, 0.01)
    round(clamp(w/scale, -1, 1)) = q             (since |q|/rho >= 1, clamp saturates to sign(q))
    forward value = q * max(gamma, 0.01) = w    -> exact round-trip, NO mismatch
Also `_absmean_ternarize(w)` recovers exactly `(q, max(gamma,0.01))`, so the exporter's `_pack_ternary_state_dict` (`export_quillan_sovereign_bitnet.py:83-105`) will write a sidecar gamma equal to the runtime scale — closing the loop.

Group size: WHOLE 2D MATRIX (equivalently whole tensor). Pooled attention/FFN tensors are all 2D. The MoE tensors are 3D and use per-expert slices, but they go through `_layer_cycle` (single donor layer, no pooling) and are outside this handoff's pooling scope.

## 5. IMPLICATION FOR MY IMPLEMENTATION — CHANGE REQUIRED, MINIMAL
State at blocker time: I had added only two edits to `quillan_weight_adapter.py` — docstring invariant #5 (line ~24) and three module-level primitives `to_ternary_pair` (:129), `pool_ternary_pairs` (:159), `emit_bitnet_native` (:192). No constructor/wiring changes, no checkpoint, no model source touched. sha 53E60D33...A5C untouched.

Decision: DO NOT adopt `build_coherent_ternary_base.py::quantize_to_ternary` (per-channel dim=-1) — that is a THIRD incompatible convention and would repeat #907. Reuse the owner adapter's existing `_absmean_ternarize` (whole-tensor == runtime whole-matrix for 2D) and align to runtime exactly by:
  (a) adding the 0.01 scale floor in emission (currently missing) — one-line change in `emit_bitnet_native`: gamma -> gamma.clamp_min(0.01);
  (b) keeping `pool_ternary_pairs` gamma = mean(gamma_i) as a whole-matrix scale (already correct);
  (c) recording the emitted/clamped gamma in `AdaptationReport.scales`;
  (d) documenting that this is numerically identical to the existing reusable `_quantize_to_ternary` at `03 - Training & Model/scripts/quillan_weight_adapter.py:32-37` (dim=[-2,-1], keepdim=True, clamp(min=0.01)) for 2D, so no new convention is introduced.

I will keep all further edits surgical and reuse existing scripts (`scripts/ternary_pack.py`, existing regression suites, existing diagnostics dir) rather than building new infrastructure. Implementation resumes only after this post.

Confidence: HIGH — backed by source line references and direct numerical measurement on the canonical checkpoint. No fake PASS. No model/checkpoint edits.
