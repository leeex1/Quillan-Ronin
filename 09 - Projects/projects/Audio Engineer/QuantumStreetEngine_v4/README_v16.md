# Quantum Street Studio v16 — Verified Master

File: `Quantum_Street_Studio_v16.html`.

## Improvements over v15

1. **Level-matched reference comparison.** `eqFromSpectrum()` now accepts the
   LUFS difference between source and reference and subtracts it from the
   reference band energies before computing EQ. Without this, a reference
   mastered 3 dB louder looked "brighter" and "wider" purely because of level —
   exactly the bias every mastering source warns about.
2. **The exported file is measured, not assumed.** `verifyRender()` analyzes
   the rendered buffer after export (LUFS, true peak, clipped-sample runs, PSR,
   crest, LRA, section spread, pump delta, correlation) and re-renders the
   12-point PASS/FAIL checklist from the **actual file**. Status bar reports
   e.g. `Exported file verified: −15.9 LUFS · −6.0 dBTP · 7/12 checks pass`.
3. **Dither last.** 16-bit WAV export now applies TPDF dither at the final
   quantizer, per the mastering rule that nothing follows dither.
4. **Whole-song verification in FULL MIX.** After the deep tune and daemon
   passes, the final rendered master is verified against the same checklist.
5. Export filename corrected to `Quantum_Master_v16.wav`.

## Verified

- Fresh load: 0 console errors.
- SMART MIX + export: report generated, file verified at −15.9 LUFS / −6.0
  dBTP, 7/12 checks on synthetic material (honest failures shown).
- Valid 768 KB WAV downloaded with dither.
- FULL MIX & MASTER: 3.45 s, 5 findings, no errors.

## Unchanged

Spectrum-driven EQ (28-band LTAS with tilt fit), genre targets only for
delivery metrics, active-listening daemon with 12-section gain map, 38-control
deep tune, 12-point checklist, BPM-aware pump detection, momentary level-matched
A/B, and 3 dB / 6 dB corrective guardrails.

## Honest limit

The app measures and adapts; it cannot hear intent. A level-matched reference
is the strongest input you can give it — load one whenever you have a target
sound in mind.
