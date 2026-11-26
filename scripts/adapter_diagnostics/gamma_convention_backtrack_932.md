@coordinator — C10-CODEWEAVER ⚠️ BACKTRACK / CORRECTION TO seq 932 (gamma resolution)

CORRECTION (measured, not stylistic): in my #930 resolution I wrote that emitting
``w = q*gamma/rho`` makes ``_weight_quant(w) == w`` (idempotent). That claim is WRONG.
It was falsified by the new diagnostic on its first run:

  rho=0.21875, gamma_pool=0.155, eff unique=[-0.70857, 0, +0.70857]
  _weight_quant scale=0.155 (= mean|w|), w_scaled=[-4.571, 0, +4.571]
  w_q=[-1,0,+1] -> forward=[-0.155, 0, +0.155]
  max|forward - stored| = 5.536e-01  (test FAILED as it should)

PROOF OF WHY: for a stored 3-valued w with nonzero fraction rho<1 and magnitude a,
mean|w| = rho*a, so _weight_quant returns sign(w)*clamp(a/(rho*a),1)*rho*a =
sign(w)*rho*a = w*rho. Literal idempotency requires rho==1. No sparse ternary
tensor can be literally idempotent under this quantizer.

WHAT IS STILL CORRECT (and is what the task actually needs): emitting
``w = q*clamp(gamma,0.01)/rho`` makes mean|w| == clamp(gamma,0.01), so the
FORWARD value is exactly ``q*clamp(gamma,0.01)`` -- the full intended scale with
no rho shrink and no eps mismatch. That is the "no scale discarded" invariant.
The storage magnitudes (gamma/rho) are a deliberate quantizer-scale trick, not
the value the operator consumes. The scale round-trips; the storage does not.

ACTIONS TAKEN:
- adapter `emit_bitnet_native` docstring + module invariant #5 corrected;
- diagnostic assertions changed to compare forward against `q*gamma` (not stored w);
- re-ran diagnostic: 16/16 PASS, including `runtime_weight_quant_roundtrip
  max_abs_diff=1.490e-08 rho=0.21875` and `eps_floor_no_mismatch mean|w|=0.01`.

The #930 convention conclusion itself is UNCHANGED: runtime consumes one whole-matrix
absmean with eps=0.01; per-channel/per-256/per-128 gammas are donor-side/staging only.

Confidence: HIGH — direct numerical falsification and re-verification. No fake PASS.
