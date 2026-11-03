# JDXX Chains — build status

## 2026-11-02 — Phase 1, first cut (M1 in progress)

Scaffold is in place: work copy of RoninEQ → `jdxx-chains`, renamed, modules
implemented clean-room, five chains wired as data, smoke suite written,
Windows MSVC build attempted.

### Written (compile attempted; results below)

- `Source/Chains/Modules.{h,cpp}` — gate, HPF, tape sim, 7-band graphic EQ,
  de-esser, FF/FB compressor, Freeverb-style reverb, stereo delay w/ HP/LP,
  tremolo, BBD chorus ("Dimension B"), vibrato, 4-stage phaser, M/S widener,
  exciter, limiter, gain — JDXX spec 2026-11-02 values.
- `Source/Chains/ChainSpecs.{h,cpp}` — the five chains as data.
- `Source/Chains/ChainEngine.{h,cpp}` — build/process/reset; `getLatencySamples() == 0`.
- `Source/Tests/chain_smoke.cpp` — the M1 proof suite (see README).
- `Source/PluginProcessor.{h,cpp}`, `PluginEditor.{h,cpp}` — host shell; M1
  exposes: mode (5), output gain, bypass. Per-module host automation = M2.

### Verified

(To be filled from this machine's build + `chain_smoke` output — never claim
audits that did not run. See the bottom of this file for the actual log.)

### Assumptions ledger (must be signed off by ear / JDXX)

- A1 gate floor fixed −80 dB; hysteresis 6 dB.
- A2 tape sim drive/bias/wow interpretations (see ChainSpecs.cpp comments).
- A3 graphic EQ bells Q = 1.0.
- A4 de-esser Q = 2.0, whole-band reduction.
- A5 FBK "Threshold 20" → −20 dBFS (BandLab unit ambiguity, ASSUMPTION).
- A6 reverb comb scaling by `size`; spread 23 samples.
- A7 delay time in seconds (tempo-independent).
- A8 Dimension B voicings: "02"/"03" guessed voicings (ASSUMPTION).
- A13 limiter: zero-lookahead, fixed −1 dBFS threshold, no LUFS auto-gain.
- A15 tremolo spread interpreted as L/R phase degrees.

### Not done

- Per-module host automation (M2) and bypass per stage.
- pluginval strictness 5+ (binary not present on this machine).
- Listening pass by JDXX on real material (the actual acceptance test).
- Installer packaging (`Install-JDXXChains-Admin.bat`) — pattern exists in RoninEQ.
- M2 metering trio for the "ears" (Phase 2).

### The fight is with the build, not the plan

Expected JUCE-8-API-sensitive spots: nothing exotic — plain `juce_audio_processors`.
If the build fails, start at: `JUCE_VST3_CAN_REPLACE_VST2`, `FontOptions`,
`NormalisableRange` signatures, and the `APVTS` parameter layout ctor.

### Build log (this machine, 2026-11-02)

```
(pending — will be appended after the first configure+build attempt)
```
