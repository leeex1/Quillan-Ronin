# JDXX Chains — AI DAW Copilot, Phase 1 ("the hands")

The VST3 plugin that carries **JDXX's own mixing and mastering chains** — extracted
from the live BandLab presets (2026-11-02) and reimplemented as clean-room DSP.
Free for everyone. VST3 first (AU/AAX later from the same codebase).

> **Jdxx, not generic.** These are JDXX's chains, from JDXX's ear — not a preset
> pack, not a skin over stock EQ.

* **Derived from RoninEQ** (AGPL-3.0) by CrashOverrideX — see `LICENSE`.
  JDXX Chains is therefore **AGPL-3.0** and stays free. The RoninEQ name is not reused.
* **JUCE 8.0.8** via `FetchContent` (offline: pass `-DFETCHCONTENT_SOURCE_DIR_JUCE=<path-to-juce-src>`).
* **License:** AGPL-3.0. Free for all users. Source offer ships with builds.

## What it is (and is not)

* A conversational AI copilot for music production is the product (Phase 2+). **This
  repo is Phase 1: the chains plugin that the brain will eventually drive.**
* It is **not** BandLab. BandLab's built-in internals are proprietary; every module
  here is an original implementation matched to the documented JDXX parameters.
  The parameter values are JDXX's own choices (see `docs/JDXX_CHAIN_SPEC.md`).
* **Zero latency** reported to the host; no lookahead anywhere in the chains.
* **No auto-loudness.** Nothing here pushes toward a LUFS target — JDXX's settings,
  JDXX's balance. The plugin never "helps" with gain staging.

## The five chains (JDXX spec 2026-11-02)

| Role | Chain | Stages |
|---|---|---|
| Vocals main 1 (and main 2 for adlibs) | `recording JDXX` | 9 |
| Vocals adlib | `adlib` | 8 |
| Vocals intro | `intro` | 8 |
| Vocals pan | `Pan` | 7 |
| Master bus | `Mastering JDXX` | 8 |

Chain order is load-bearing. Never reorder. Values live in
[`docs/JDXX_CHAIN_SPEC.md`](docs/JDXX_CHAIN_SPEC.md) and in code as data
(`Source/Chains/ChainSpecs.cpp`) — the same table, two places that must agree.
Assumptions (ambiguous BandLab scales) are ledgered in the spec doc and in
`ChainSpecs.cpp` comments.

## Build (Windows + VS Build Tools)

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64   # 17 2022 also works
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Output: `build/JdxxChains_artefacts/VST3/JDXX Chains.vst3` (copied to the system
VST3 folder automatically) plus a Standalone app and `chain_smoke`.

## Verify (what you can check yourself)

```powershell
.\build\chain_smoke.exe
```
Expect `N checks, 0 failures`. The suite covers: all five chains load with
exact stage counts; silence→silence; no NaN/Inf under sustained input;
EQ bell magnitude ±0.15 dB at band centers; FF compressor ratio ±2 dB of the
law; limiter peak clamp; gate attenuation; delay echo timing ±8 samples;
zero reported latency.

Host-load check (VST3 in any host) is the last gate — no JUCE-level check can
prove the host's view. When available: `pluginval` strictness 5+
(see docs/JDXX_CHAIN_SPEC.md, "Acceptance").

## Source layout

```
Source/
  PluginProcessor.{h,cpp}    Host shell: mode select, out gain, bypass (M1)
  PluginEditor.{h,cpp}       Minimal UI: mode, gain, bypass, spec read-out
  DSP/                       Biquad, EnvelopeFollower (reused, renamed namespace)
  Chains/
    Modules.{h,cpp}          Clean-room modules (gate..limiter)
    ChainSpecs.{h,cpp}       The five chains as data (spec 2026-11-02)
    ChainEngine.{h,cpp}      Build/reset/process, zero latency
  Tests/chain_smoke.cpp      Standalone verification (no JUCE)
```

## Status

See [`STATUS.md`](STATUS.md) — what is verified, what is assumed, what is not
yet host-checked. Trust that file, not marketing.

## License

AGPL-3.0. This program is free software: you can redistribute and/or modify it
under the terms of the GNU Affero General Public License v3.0 (or later) as
published by the Free Software Foundation. See `LICENSE`. JUCE is used under its
AGPLv3 open-source terms (which this license satisfies).
