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

### Verified (this machine — actual output)

Build: `cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DFETCHCONTENT_SOURCE_DIR_JUCE=<cached juce-src>` then `cmake --build build --config Release` → **all targets OK** (VST3, Standalone, chain_smoke).

```
JDXX Chains — chain_smoke @ 48000 Hz
  EQ band 0 @ 100 Hz  -> +5.99 dB   band 3 @ 800 Hz -> +6.00 dB
  EQ band 1 @ 200 Hz  -> +6.00 dB   band 4 @1600 Hz -> +6.00 dB
  EQ band 2 @ 400 Hz  -> +6.00 dB   band 5 @3200 Hz -> +6.00 dB
  band 6 @6400 Hz     -> +6.00 dB   (all within 0.15 dB of +6.00)
  Compressor 4:1 @ -6 dBFS -> -6.44 dB drop (law predicts ~6.75; OK)
  Limiter peak out = 0.9499 (<= 1.0 clamp verified)
  Gate attenuation below threshold = 66.8 dB (>= 20 dB required)
  Delay echo at sample 2400 (target 2400 +/- 8)
36 checks, 0 failures
```
`ctest` → `100% tests passed out of 1`.
VST3 bundle: well-formed `Contents/x86_64-win/JDXX Chains.vst3` + valid `moduleinfo.json`,
new CIDs (no RoninEQ identity reuse), `Fx|Mastering`, vendor CrashOverrideX.

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
VS Build Tools 2026 (MSVC 19.51), JUCE 8.0.8 (cached FetchContent source), Windows SDK 10.0.26100.0
cmake configure: OK (~230 s first pass, incl. juceaide), reconfigure: OK
cmake --build build --config Release: OK — VST3 + Standalone + chain_smoke
ctest: 1/1 passed
fixes during build: DSP/ include prefix; editor subclasses juce::Timer (abstract member fix);
APVTS via ParameterLayout::add() (JUCE 8 idiom); dryCopy.isNotEmpty() -> getNumSamples();
namespace JdxxChains closed; createPluginFilter() + getTailLengthSeconds/acceptsMidi/producesMidi;
static inline DB_TO_LIN/LIN_TO_DB (constexpr cannot call pow); COPY_PLUGIN_AFTER_BUILD=OFF
(system VST3 folder needs admin — use Install-JDXXChains-Admin.bat)
```
