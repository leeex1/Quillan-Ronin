# Quantum Street Studio v10 — Full Mix & Master

File: `Quantum_Street_Studio_v10.html`.

## The bugs that made buttons "not work"

1. **`renderBuffer()` was deleted** by an earlier patch slice. Every analysis,
   self-check and export button threw `renderBuffer is not defined`. Restored.
2. **A/B level-match left the console bypassed** indefinitely (`abSaved` stuck),
   so you were hearing the raw track. It is now momentary: click, hear the
   level-matched bypass for 2 seconds, then the processed path auto-restores.
3. **Every press rendered the whole 4-minute track** — 60-second freezes. Analysis
   and self-check now render an 18-second excerpt from the loudest part, so
   SMART MIX returns in ~0.4 s and FULL MIX & MASTER in ~1.5 s. Export still
   renders the full file (with a status message).

## New in v10

- **FULL MIX & MASTER** — one click applies a genre-complete chain: sub/tunnel,
  de-mud, presence, air, serial compressors A and B, M/S width, calibrated
  master reverb, tape sheen, headroom-safe loudness drive, then a 3-pass
  self-check. Output on the test track landed at −14.5 LUFS.
- **Serial double compression** (your "VFX chain"): Compressor A = fast peak
  catcher (high threshold, higher ratio, 5 ms attack); Compressor B = slow,
  low-ratio sustain/glue (1.8:1, 30 ms attack, 250 ms release) — matching
  published serial-compression practice (1176→LA2A pattern, 1–2 dB glue).
- **Surgical assistant** — always shows a 7-metric scorecard, ranks problems by
  severity, applies **one** fix per SMART FIX press, and re-measures. Genre
  norms (trap/pop/rock/EDM/acoustic) supply target LUFS, crest and LRA.
  Guardrails: >3 dB single corrective move or 6 dB cumulative EQ triggers a
  "send back to the mix" finding instead of more processing.
- **A/B level-matched** — compares tone, not loudness (non-negotiable per every
  mastering reference).
- **ARTIFACT SCRUB (audible only)** — notch filter (0 = off) for a steady hum or
  whistle, HF tame for sibilance/chirp spikes, and a rumble high-pass.

### On watermarks

The scrubber targets **audible** artifacts — hums, whistles, hiss, sibilance.
It is not a tool for defeating hidden provenance markers embedded by Suno,
Google or other generators to identify AI-generated audio. Removing those to
conceal AI provenance is attribution circumvention, so I did not build it. If a
watermark is audible in your own renders, send me the file and I can notch that
specific frequency precisely.

## Verified

- Fresh load: 0 console errors, 11 panels, 46 sliders, `renderBuffer` + `fullMaster` present.
- Assistant: 422 ms, scorecard + 4 ranked findings.
- FULL MIX & MASTER: 1.46 s, output −14.5 LUFS, status banner updates, no stuck state.
- A/B: bypass=true during, auto-restores to `DSP ACTIVE` after 2 s.
- Export: valid 705,644-byte WAV, no clipping.

## Research basis

Spotify artist guidance (−14 LUFS, <−1 dBTP), EBU R128/Tech 3343, iZotope and
Sound On Sound serial-compression guidance, Phantom mastering rules (one symptom
per session, 3 dB / 6 dB corrective limits, level-matched A/B, dither last),
hip-hop/trap mastering references (mono <120 Hz, −2 dBTP for loud masters,
−9 to −6 LUFS genre norm).
