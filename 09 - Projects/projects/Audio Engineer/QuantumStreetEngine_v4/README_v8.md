# Quantum Street Studio v8 — calibrated master

File: `Quantum_Street_Studio_v8.html`. Fixes the "too much space" problem in v6.

## What was actually wrong (not a taste issue)

1. **Convolver gain was ~13× too high.** The generated impulse response was
   normalized by RMS (`0.12/√(e/N)`), but a convolution's gain scales with the
   *total* IR energy (`√e`). At 9 % wet the reverb return was effectively
   ~100 % — a wash. Fixed to `0.8/√e`, so the wet control now means what it says.
2. **No Abbey Road filtering.** Research (Music Guy Mixing, Sage Audio) says
   master-bus reverb should be high-passed ~550 Hz and low-passed ~6.5 kHz, and
   kept around 10 % wet — "feel it, don't hear it." v6 had only a 3.5 kHz HP and
   no LP, so it added glare and mud. v8 has both filters; defaults 10 % / 0.20 s /
   550 Hz HP / 6.5 kHz LP.
3. **Width was too much and in the wrong place.** Published guidance: single
   digits above 100 %, and place the width move *before* dynamics, because
   widening after limiting undoes peak control. v8 uses 104 % default and moves
   the M/S matrix to before the glue compressor.
4. **v7 fixed the runaway assistant**: it now analyzes the *rendered output*
   instead of the immutable source, so findings retire after a fix; gain moves
   are absolute and capped (±8 dB pre-limit drive), and width findings no longer
   fight the M/S stage.

## Measured

- Convolver alone: 0.5 sine → 0.092 RMS return (correct scale).
- Full chain A/B: wet 0 vs 100 changes the output by 0.033 RMS; wet 10 % is a
  subtle ~0.003 — audible as space, not as a wash.
- Live graph converges: reverb HP 550 Hz, LP 6500 Hz, wet 0.10, width gains
  1.02 / −0.02 (104 %).
- Fresh load: 0 console errors, 38 sliders, 9 panels. Assistant converges
  (preDrive 0 → 2 after FIX ALL, stays bounded), export valid 705,644-byte WAV.

## Practical note

Mastering reverb is genuinely rare — one 20-year engineer quoted fewer than 10
uses in thousands of masters. If your source mix already sounds cohesive, the
correct setting is often 0–5 %. Use the A/B bypass and trust your ears over the
number.
