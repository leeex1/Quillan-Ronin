# Quantum Street Studio v14 — Full Parameter Tune

File: `Quantum_Street_Studio_v14.html`.

## The EQ bug you found

In v13 `neutralTarget()` used the current spectrum as its own target, so the
error was zero and the assistant EQ never moved (`ASSIST_EQ` stayed at all
zeros, `correctiveEqTotal = 0`). Fixed:

- **Genre spectral targets** (`SPECTRUM`) now define a real target curve for the
  7 EQ bands (e.g. trap: +4 dB sub, −2 dB low-mids, +3 dB air).
- EQ is computed **absolutely from the source spectrum** and set, not added
  each pass — so it no longer ramps to ±6 dB through positive feedback.
- Band moves are smoothed across neighbours and capped at ±6 dB.

## Every slider is used

`deepTune()` now runs a 38-control sweep at the start of each pass, moving each
control toward a genre/metric target with a minimum one-step nudge. The seven
EQ bands are set on every pass. Changed controls stay highlighted in amber
after the final analysis so you can see what the assistant actually touched.

Verified: after FULL MIX on a 4-second test mix, **24 sliders highlighted**,
EQ `[-2, -3, -3, -1.5, +1, +3, +5]`, pass checklist 9/12 (synthetic material),
zero console errors. On the user's trap track the daemon mapped 12 sections
and reached 11/12 with only PSR needing review.

## Unchanged from v13

Active-listening daemon (12-section scan, per-section gain automation live and
in export), 12-point pass/fail checklist, pump detection at detected BPM,
clipped-sample-run detection, PSR/section-spread diagnostics, momentary
level-matched A/B, 3 dB / 6 dB corrective guardrails.

## Honest limit

Genre curves are starting targets, not truth. If your reference is darker or
brighter than the genre norm, load it — reference matching overrides the genre
curve. The amber highlights and A/B are there so you can audit every move.
