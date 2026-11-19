# Quantum Street Studio v5.1 — professional-grade Smart Assistant

File: `Quantum_Street_Studio_v5.html` (self-contained, open in a browser).

## What changed from v4, and why (research-backed)

I checked current published guidance: Spotify artist docs (2026), EBU R128 /
Tech 3343, and the AES/ITU-based streaming summaries. The key facts:

- **Spotify**: target −14 LUFS integrated, keep **below −1 dBTP**; if the master
  is *louder* than −14, keep below −2 dBTP. Positive playback gain is limited
  by headroom (≈1 dB reserved for lossy encode).
- **EBU R128**: −23 LUFS broadcast, **−1 dBTP** max, and loudness must be
  measured with **K-weighting + 400 ms blocks + absolute/relative gating**
  (ITU BS.1770).
- **True peak ≠ sample peak**: a limiter at −1 dB sample peak does not guarantee
  −1 dBTP. Lossy encoding can create inter-sample overs.
- **Dynamics matter**: loudness range and crest factor describe whether a master
  is crushed. Encrypting everything to a target number destroys impact.

### Implemented in v5.1

1. **Real LUFS measurement** — K-weighting high-pass @ 38 Hz + +4 dB high shelf
   @ 1681.97 Hz, 400 ms blocks, absolute gate −70 LU and relative gate −10 LU
   (computed from *linear energy*, not averaged LUFS). The readout now says
   "ITU BS.1770", not the old crude RMS+7 estimate.
2. **True peak** — 4× oversampled Catmull–Rom interpolation estimate, displayed
   in dBTP. The old sample-peak number was misleading.
3. **Headroom-safe gain moves** — before applying any loudness change the
   assistant predicts `truePeak + gainDelta`. If that exceeds the target
   ceiling it applies only the safe amount and tells you why. Gain is now
   applied by a dedicated **Pre-Limit Drive** control (into the glue/limiter),
   not by the post-limiter output trim.
4. **Dynamics findings** — crest factor and loudness range (LRA) are measured;
   over-crushed material triggers a finding that relaxes the compressor rather
   than adding more limiting.
5. **Gentle reference matching** — assistant EQ is clamped to ±6 dB and
   neighbouring bands are smoothed; targets are capped so matching a reference
   cannot wreck the mix's character.
6. **FIX / IGNORE with confidence + memory** — each finding is a card with a
   confidence score; IGNORE lowers future confidence for that finding type via
   `localStorage`, so the assistant learns your taste.
7. **Target + style selectors** — Streaming −14, Club −9, Broadcast −16,
   YouTube −13, Lo‑fi −18; plus genre target curves.
8. **Self-check passes** — renders the chain offline, re-analyzes the *output*,
   corrects, and repeats 3 times, showing LUFS/dBTP convergence per pass.

## Verified

- Fresh load: 0 console errors, 25 control sliders + 7 assistant-EQ bands, 8 panels.
- `analyse()` on the 268.7 s test track: 2.5 s, returns plausible values.
- Assistant on a synthetic mix: 6 findings with confidence; FIX changed 2 knobs;
  self-check converged −16.3 → −15.5 LUFS while true peak stayed at −4.8 dBTP
  (well under −1).
- Reference test: a deliberately bass-heavy reference moved assistant EQ from
  `[0,…]` to `[+12,+12,+4.5,−4.5,−4,0,+1]` (pre-clamp); with the ±6 clamp now
  active it stays gentle.
- Export: valid 705,644-byte WAV, no clipping (peak 0.639).

Bugs found and fixed during this work: double division in the K-weighting shelf
coefficients (21 dB error), wrong relative-gate domain, a duplicate `exportWav`
that bypassed the assistant EQ, and a missing `wavBlob` after cleanup.

## Your track

Measured with the new meter: **−15.7 LUFS, −1.1 dBTP, crest 13.9 dB, LRA 7 LU,
width 0.91**. That is already within a dB of the −14 streaming target, at the
−1 dBTP ceiling, with healthy dynamics. Your instinct that you follow best
practices is correct — the assistant's job here is to preserve those qualities,
not to push the level harder.
