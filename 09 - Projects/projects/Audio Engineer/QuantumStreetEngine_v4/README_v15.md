# Quantum Street Studio v15 — Spectrum-Driven EQ

File: `Quantum_Street_Studio_v15.html`.

## The problem you heard

v14's EQ used genre curves — a preset template that boosted trap sub and cut
low-mids whether the song needed it or not. That's why the bass got heavy and
the EQ felt blind.

## What v15 does instead

`ltas()` computes a **long-term average spectrum** of the actual song across
28 log-spaced bands (30 Hz–15 kHz), fits the track's own spectral tilt by least
squares, and reports deviations from that tilt. `eqFromSpectrum()` then sets
the 7 assistant EQ bands to correct the **measured** anomalies:

- a resonance/dip is cut or lifted only when it exceeds ~1.5 dB
- neighbouring bands are averaged so one spike doesn't create a weird curve
- sub level is judged by comparing <60 Hz energy to 60–120 Hz, not by genre

Genre now only sets broad delivery targets (LUFS, PSR, LRA, compression style)
— never the EQ curve. Blanket bass boosts were removed: `subTrim` defaults to
0–1 dB, and deep tune only changes it when measured sub excess exceeds ±3 dB.

## Verified

- Synthetic mix: detected mud at 280–560 Hz (+2.3…+2.6 dB) and a dull 12–15 kHz
  region (−2.7…−3.4 dB); EQ became `[+0.5, −0.5, −1.5, −1, 0, 0, +1.5]` —
  cuts where the track had excess, lift where it lacked air.
- User's trap track: detected deep sub deficit (−5.2 dB at 30–40 Hz) and mud at
  280 Hz; `subTrim` stayed at 0, EQ was driven by the measurements; 11/12 pass
  with pump delta the only review item; −13.7 LUFS; 0 console errors.
- 22–24 sliders changed per full run; highlights persist.

## Honest limit

A long-term average spectrum cannot distinguish a deliberate sub-heavy mix
from an accidental one. That judgment still needs ears at level-matched A/B.
The assistant now reports the exact anomalies it found so you can audit them.
