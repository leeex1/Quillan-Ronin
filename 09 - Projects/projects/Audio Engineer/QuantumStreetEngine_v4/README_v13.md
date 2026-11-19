# Quantum Street Studio v13 — Active Listening Daemon

File: `Quantum_Street_Studio_v13.html`.

## DAEMON · ACTIVE LISTEN

Press **DAEMON · ACTIVE LISTEN** (or FULL MIX & MASTER, which runs it):

1. `scanSong()` divides the whole track into 12 sections and measures LUFS,
   true peak, correlation and clipped samples in each.
2. `daemonPlan()` builds a per-section gain map — quiet sections get up to +2 dB,
   loud sections up to −2 dB. The arrangement's intent is preserved; it never
   flattens the song to one level.
3. The plan is applied on **live playback** (pre-limit gain follows the current
   section) and **baked into export** via AudioParam automation at each section
   boundary.
4. `deepTune()` runs 3 more passes over all 38 metric-linked controls.
5. A **PASS / FAIL** checklist is rendered from the full-song measurements.

## Pass / fail criteria (12 checks)

Integrated LUFS vs target · true peak vs ceiling · clipped-sample runs · PSR ·
crest factor · loudness range · section spread · pump delta · stereo
correlation · cumulative corrective EQ · reverb amount · width amount.

Each row shows the measured value, the target and PASS/REVIEW. On the user's
4-minute trap track the finalized master reported **12/12** at −14.7 LUFS.
On synthetic material it returned 7/12 with named failures — the point is to
show what still needs human review, not to claim a perfect result.

## Accuracy improvements

- Deep tune runs **after** the section scan, so its decisions see the whole
  arrangement, not just an excerpt.
- Pump detection compares output vs source modulation depth at the **detected
  BPM** and retimes compressor release to recover before the next kick.
- Clipping is detected by consecutive full-scale sample runs, not peak level.
- PSR and section spread catch "squashed" and "chorus doesn't lift" problems
  that a simple crest measurement misses.
- A/B is level-matched and momentary (2 s), so comparisons judge tone, not
  loudness.

## Verified

- Fresh load: 0 console errors, 46 sliders, daemon/checklist present.
- DAEMON on a 4-minute track: 12 sections mapped, 12/12 criteria pass,
  −14.7 LUFS, 9.4 s, no errors.
- Deep tune: 38 controls evaluated per pass.
- Export with section automation: valid WAV (small test file); a 4-minute WAV
  exceeds this test harness's 5 MB download limit, not the app's capability.

## Honest limit

The daemon measures and adapts; it does not hear intent. A quiet breakdown is
not necessarily an error, and a resonance can be character. The checklist is a
technical gate, not a substitute for level-matched listening.
