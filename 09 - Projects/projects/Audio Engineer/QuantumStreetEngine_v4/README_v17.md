# Quantum Street Studio v17 — Magic Producer

File: `Quantum_Street_Studio_v17.html`.

## What changed

1. **No template drives a slider.** The genre `FULL` map no longer sets any
   control. `deepTune()` derives every target from the measured audio: 28-band
   LTAS spectrum, sub excess, PSR, crest, LRA, section spread, pump at detected
   BPM, clipped-sample runs and stereo correlation. Genre only supplies
   delivery tolerances (LUFS / PSR / LRA).
2. **Second spectrum analyzer.** A new LTAS canvas draws **source (cyan) vs
   processed (amber)** long-term spectra with frequency ticks — the EQ moves
   are now visually auditable against the actual sound.
3. **Coupled controls.** Moving one slider rebalances its related partners:
   - Tunnel drive ↑ → sub trim ↓ and tape sheen ↑ (level compensation)
   - Sub tightness ↑ → width ↓ (protects mono low end)
   - Presence ↑ → de-ess and de-harsh ↑
   - Air ↑ → exciter ↓ (avoids double brightness)
   - Comp ratio ↑ → threshold ↑ and attack ↑ (keeps GR sane)
   - Reverb wet ↑ → reverb HP ↑ and LP ↓ (keeps the tail out of mud)
   - Width ↑ → sub tightness ↑, and vice versa
   - Pre-drive ↑ → input trim ↓ and clip knee ↓ (total gain preserved)

   Coupling only runs on **your** slider moves; the assistant's programmatic
   changes do not re-trigger it.

## Verified

- FULL MIX on synthetic mix: 25 sliders changed, audio-derived EQ
  `[0, −1, −2, −1.5, 0, 0, +1]`, −14.3 LUFS, 9/12 pass, 0 console errors.
- LTAS canvas captured source and processed spectra.
- Coupling test: tunnel drive 35→90 moved sub trim −0.5→−1.6 and tape 15→18.3;
  presence 3.5→5 moved de-ess 0→0.52.
- All v16 features intact: level-matched reference, post-export verification,
  TPDF dither, 12-section daemon, 38-control deep tune, guardrails.

## Honest limit

"Magic producer" is a direction, not a finished destination. An LTAS cannot
hear that a vocal whisper is intentional. The second analyzer, the amber
highlights and level-matched A/B exist so every automated move can bechecked
and reversed by ear.
