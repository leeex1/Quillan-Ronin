# Quantum Street Studio v12 — Deep Tune

File: `Quantum_Street_Studio_v12.html`.

## What changed from v11

`DEEP TUNE · ALL PARAMETERS` replaces the old self-check. Each pass measures
the processed excerpt and evaluates **all 38 metric-linked controls** against
specific evidence, not a genre preset:

| Evidence | Controls evaluated |
|---|---|
| Integrated LUFS / true peak / clipped-sample runs | preDrive, inputTrim, truePeakCeiling, softClipKnee, masterOutGain |
| PSR, crest, LRA, section spread, transient sharpness, BPM-locked pump | comp A/B threshold, ratio, attack, release |
| Sub / mud / presence / air band energy | subTrim, subTunnelDrive, subCrossover, lowCutFreq, hpRumble, mudFreq/Dip/Q, boxDip, presenceFreq/Gain/Q, airShelfFreq, airBoost, shorExciter, highSmooth |
| Stereo correlation / mono fold | stereoWidth, subTightness |
| Space, saturation, artifacts | spaceWet/Size/Tone/LP, tapeDrive, deSess, deToneDepth |
| Reference / genre spectrum | all 7 assistant-EQ bands |

Guardrails unchanged: >3 dB single corrective move or 6 dB cumulative EQ
triggers "send back to the mix" rather than more processing. A/B stays
level-matched and momentary.

## New diagnostics (from research)

- **PSR** (peak-to-short-term loudness ratio) — the direct number for "squashed".
- **Clipped-sample runs** — consecutive full-scale samples, not just peak level.
- **Pump delta** — rhythmic modulation depth of the output minus the source, at
  the detected BPM; release is retimed to recover before the next kick
  (`60000/BPM` ms).
- **Section spread** — loudest minus quietest of 8 blocks; <3 LU means the
  chorus does not lift above the verse.
- **Transient bluntness** — mean onset jump; low values mean fast attack is
  shaving the snap off kicks and snares.

FULL MIX & MASTER now calls deep tune, so a single click measures the song and
adapts compression timing to its groove rather than applying fixed numbers.

## Verified

- Fresh load: 0 console errors, 46 sliders, 38 tracked controls.
- Deep tune on synthetic mix: 4 passes in 1.2 s, 12–14 parameter checks per
  pass, LUFS converging −15.9 → −15.5, true peak always under ceiling.
- FULL MIX & MASTER on the 4-minute user track: 4.8 s, 3 passes, output −14.9
  LUFS, findings included PSR, pumping, mud and blunted transients.
- Export: valid WAV, no clipping.

## Honest limit

This is measurement-driven adaptation, not listening. It cannot know that a
quiet breakdown is meant to feel intimate, or that a vocal resonance is
intentional character. The scorecard and A/B exist so you make that call.
