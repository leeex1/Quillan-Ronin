# RoninEQ — build status

## 2026-09-28

**DSP core: compiled + smoke-tested on Linux 2026-09-28. JUCE plugin shell/editor: written against JUCE 8 API, NOT compiled here — first full build happens on the user's Windows machine with Visual Studio; fix any JUCE API mismatches then.**

### Verified (ran here, exit 0, warning-clean)

```
g++ -std=c++17 -Wall -Wextra -O2 \
  Source/DSP/*.cpp Source/Tests/dsp_smoke.cpp \
  -o /tmp/roneq_smoke && /tmp/roneq_smoke
```

`ALL CHECKS PASSED` — 22 checks:

- Bell +6 dB at center reads +6.00 dB; far-field ≈ 0 dB
- Low shelf +6 dB / high shelf −6 dB land within 0.1 dB deep in the shelf
- HP/LP passbands ≈ 0 dB, stopbands < −24 dB
- Notch: −103 dB null at center, ≈ 0 dB two octaves up
- Quiet in-band burst (−33 dB RMS): passes untouched (−32.8 dB out)
- Loud in-band burst (−9 dB RMS): cut to −18.7 dB (−9.7 dB drop, ≈ the
  −10.5 dB the 4:1 law predicts; remainder is detector ripple)
- Dynamic meter reads −9.93 dB during the cut
- Static +6 dB boost at 5 kHz lands at exactly +6.0 dB with dynamics idle
- Left-only loud burst still cut (stereo-linked detector works)
- After 1 s of true silence the meter relaxes to −0.00 dB (release works)
- Mid-stream parameter change: no NaN/inf

Two test bugs were found and fixed during verification (both in the test,
not the DSP): RMS expectations now measure the actual input buffer instead
of assuming peak==RMS, and the release "silence" now feeds fresh zeros
instead of accidentally creating a 1-sample feedback loop through the
3-band chain.

### Written, not compiled here (JUCE side)

`CMakeLists.txt`, `Source/PluginProcessor.*`, `Source/PluginEditor.*`,
`cmake/PluginInfo.h.in` are written against the JUCE 8 API from the
JUCE 8.0.x headers/docs but have **never been compiled** — there is no JUCE
here. The first real build is the user's `cmake --build build --config
Release` on Windows. Expected risk points to check first if it fails:

1. `juce::dsp::FFT::performFrequencyOnlyForwardTransform(float*)` —
   single-argument call; confirmed present in JUCE 8's own
   `juce_Oversampling.h` doc comment, but verify the 2×size scratch
   requirement on the target JUCE version.
2. `juce::dsp::Oversampling<float>` constructor/processing call shapes.
3. `AudioProcessorValueTreeState` attachment constructors.
4. `juce_generate_juce_header` + `#include <JuceHeader.h>` under
   FetchContent (standard, but version-sensitive).

### Suggested first-run checklist (Windows)

1. `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`
2. `cmake --build build --config Release`
3. `ctest --test-dir build -C Release --output-on-failure` (DSP smoke via MSVC)
4. Load the VST3 in a host, then run
   [pluginval](https://github.com/Tracktion/pluginval) strictness 5+
5. Dogfood on a JDXX track: drag nodes, Alt-drag Q, enable dynamics on
   band 1 and watch the GR ring.

## 2026-09-29 (Windows MSVC build)

Built + linked on Windows (VS Build Tools 2026, JUCE 8.0.8): VST3,
Standalone, `ctest dsp_smoke` passes under MSVC. Fixed 6 JUCE-8 API
mismatches (`RangedAudioParameter`, `FontOptions`, `Biquad::setUnity`,
`ValueTree::readFromStream`, `JUCE_VST3_CAN_REPLACE_VST2=0`) plus a
band-switch bug where new attachments echoed values into the previous
band (teardown-first fix in `rebuildBandAttachments`).

Display upgrades: live dynamic summed curve (bright) over a dim static
ghost, so compression pumps the curve FabFilter-style; analyzer range
widened to +24/−60 dB so quiet signals stay visible.
