# Quantum Street Studio v6 — Space & Width

File: `Quantum_Street_Studio_v6.html`. Adds the spatial finishing stage that v5
was missing. v5.1 remains available for A/B comparison.

## The gap you heard

v5's chain was rumble → sub → tone → exciter → assistant EQ → glue → clip →
limiter. There was **no reverb, no stereo-width stage and no whole-mix
saturation** — so even a clean master reads dry next to commercial records.

## What's new (panel 8: SPACE & WIDTH)

| Control | Range | Default | What it does |
|---|---|---|---|
| Glue Reverb | 0–20 % | 9 % | Short generated master-room tail, high-passed before the convolver so it can't muddy low end |
| Room Size | 0.08–0.8 s | 0.28 s | Impulse length — kept short for mastering |
| Reverb HP | 1–12 kHz | 3500 Hz | High-pass on the tail; early reflections are added explicitly |
| Stereo Width | 0–180 % | 118 % | True M/S matrix (L′=M+S, R′=M−S), not a mono-making trick |
| Tape Sheen | 0–100 % | 20 % | Gentle `tanh` whole-mix harmonic saturation |

Signal order: glue comp → (dry + reverb return) → M/S width → tape → clipper →
limiter → output. Reverb is **before** the limiter so it cannot undo the
ceiling. The IR is generated deterministically per size/tone setting and cached.

The assistant also gained a **"Mix reads dry / narrow"** finding: if reverb is
under 3 % or width under 100 %, FIX restores conservative defaults (6 % / 0.25 s
/ 3500 Hz / 115 % / 15 %).

## Verified

- Loads with 0 console errors, 9 panels, 37 sliders (30 controls + 7 assistant EQ).
- Live graph converges correctly: reverb HP 3500 Hz, wet gain 0.081, M/S gains
  1.09 / −0.09, 12,348-sample stereo IR generated.
- Offline A/B on the same signal: reverb off → RMS 0.2339, peak 0.513,
  correlation 0.811; reverb on → RMS 0.2621, peak 0.574, correlation 0.744.
  Wider and louder, still no clipping.
- Export after applying the assistant: valid 705,644-byte WAV, −14.1 dB RMS,
  peak 0.595, no clipping.

## How to hear it

Load your track, then use **DSP ACTIVE / BYPASSED** to compare. Try **Glue
Reverb** between 6–12 % and **Stereo Width** 110–130 %. If the low end gets
soft, raise **Reverb HP** to 5 kHz or drop **Room Size** to 0.18 s — that keeps
the space up top where it sells "industry" width without blurring the kick.
