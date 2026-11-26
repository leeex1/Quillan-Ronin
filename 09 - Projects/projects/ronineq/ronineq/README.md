# Ronin-EQ

## 🧭 Canonical project context

| | |
| :--- | :--- |
| **This repo** | Component repository in the Quillan-Ronin ecosystem. |
| **Canonical package** | Three deployed models: Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B). The ONI Python sources are implementation/training only. |
| **Donor design** | Flagship and Mini: Qwen 0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B **as a group** (cross-match fix in progress). Dragon: Falcon3 with an Ollama system message running Quillan's system prompt. |
| **Reading the scores** | The 95–100% table in the root README is **Track A** — the Quillan harness on a third-party frontier base, not a Quillan checkpoint. Hard-exam cells are pending re-test; ARC-AGI-3 public ~99s, private pending. |
| **Historical framing** | Edo period, rōnin and bushidō are real Japanese history and ethics — not roleplay, game mechanics, or fantasy lore. |
| **Canonical docs** | [Root README](https://github.com/leeex1/Quillan-Ronin#readme) · [Fleet model cards](https://github.com/leeex1/Quillan-Ronin/blob/main/docs/MODEL_CARDS.md) |

<!-- QUILLAN-DOCS-PROVENANCE:BEGIN -->
<details>
<summary><b>📌 Quillan documentation provenance</b> — expand for canonical sources</summary>

> **Documentation provenance.** This repo is part of the Quillan-Ronin ecosystem (CrashOverrideX / `leeex1`). Canonical sources live in the root repository: `README.md` (**READ FIRST** — Track A harness-on-frontier vs. Track B native WSP models vs. Track B′ custom Mini) and `docs/MODEL_CARDS.md` (fleet identity, verified facts, open gaps). The large 95–100% benchmark table in the root README is a **harness-on-frontier** result, not a score of any Quillan-owned checkpoint. Creator artifacts (books, music, game demo) have primary-source links in the root README's **Creator and verifiable artifacts** section.  Four-donor design: Flagship and Mini receive all four donors (Qwen ~0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B) **as a group**; Dragon is Falcon3 with an Ollama system message running Quillan's system prompt. Cross-match fix in progress.  The canonical package is the three deployed models — Quillan-Ronin Flagship (~1B), Quillan-Ronin-Mini (~0.8B) and Quillan-Dragon (~1B); the ONI Python sources are the implementation/training package, not the shipped product. Canonical official links (site, social, repos, Ollama, HF, Grokipedia, NotebookLM, DeepWiki, Genius, YouTube/Suno) are in the root README's **Official links** directory. Historical framing is real Japanese history (Edo period, rōnin, bushidō) used as an ethical reference frame — not roleplay, game mechanics, or fantasy lore.

</details>

<!-- QUILLAN-DOCS-PROVENANCE:END -->

A clean-room **dynamic Quantum equalizer** in the FabFilter Pro-Q / TDR Nova class —
12 dynamic parametric bands, Dynamic Q and band adjustments ,Pro-Q-style draggable curve, pre/post spectrum
analyzer, 2× oversampling, built in dual compression, Built in multiband compressor, novel audio entaglment adjustments. 100% original code, no forks, no copied DSP.

* **DSP core**: hand-rolled C++17 (RBJ cookbook biquads, peak envelope
follower, per-band downward dynamic gain). Zero dependencies — it builds
and smoke-tests with just a C++ compiler.
* **Plugin shell**: JUCE 8 (VST3 + Standalone), CMake `FetchContent`.
* **License**: AGPL-3.0 (see `LICENSE`). JUCE is used under its AGPLv3
open-source terms, which this license satisfies.

## Features (v1)

* 6 bands, each: enable, Bell / Low Shelf / High Shelf / High Pass /
Low Pass / Notch, 20–20,000 Hz, ±15 dB, Q 0.1–10
* Per-band dynamics: threshold −60–0 dB, ratio 1–20:1, attack 0.1–100 ms,
release 10–1000 ms, stereo-linked detector, smoothed gain (no zipper noise)
* Global input gain, output gain, bypass
* 2× oversampling (polyphase half-band IIR) around the band chain, with
latency reported to the host
* Pre/post FFT analyzer (2048-pt, Hann, 256 log-spaced bins) with peak decay
* Pro-Q-style UI: logarithmic grid, summed static response curve, draggable
nodes (drag = freq/gain, Alt/right-drag = Q, double-click empty space =
enable a band there), per-band GR ring + readout

## Build (Windows + Visual Studio 2022)

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The VST3 is copied to the system VST3 folder automatically
(`COPY\_PLUGIN\_AFTER\_BUILD`). Requires internet on first configure
(FetchContent pulls JUCE 8.0.8).

Rename the plugin in exactly one place: the `RONINEQ\_NAME` cache variable
in `CMakeLists.txt` (it flows into the product name, the processor, and the
generated `PluginInfo.h`).

## Verify the DSP without JUCE

```bash
# Linux/macOS/WSL — also the exact command CI should run:
g++ -std=c++17 -Wall -Wextra -O2 \\
  Source/DSP/\*.cpp Source/Tests/dsp\_smoke.cpp \\
  -o /tmp/roneq\_smoke \&\& /tmp/roneq\_smoke

# ...or via CMake/CTest after configuring:
ctest --test-dir build -C Release --output-on-failure
```

The smoke test checks biquad magnitude responses against cookbook values and
drives a 3-band chain through quiet/loud/in-band/out-of-band bursts:
it must show a loud 1 kHz burst getting cut \~10 dB, a quiet one passing
untouched, a static +6 dB boost landing exactly, stereo-link engaging on a
left-only signal, and the dynamic gain releasing back to 0 dB in silence.

For a deeper host-level check, run the built VST3 through
[pluginval](https://github.com/Tracktion/pluginval) (strictness 5+).

## Architecture

```
Source/
  PluginProcessor.{h,cpp}   APVTS (63 params), oversample+band chain,
                             pre/post FFT analyzer -> lock-free FIFO
  PluginEditor.{h,cpp}       spectrum/grid/curve/nodes, band + dynamics
                             controls, attachment rewiring on band select
  DSP/
    Biquad.{h,cpp}           RBJ cookbook biquads (clean-room reimplementation)
    EnvelopeFollower.{h,cpp} peak attack/release ballistics
    DynamicBand.{h,cpp}      static filter + stereo-linked dynamic gain stage
  Tests/
    dsp\_smoke.cpp            standalone verification (no JUCE)
```

The audio path is: input gain → pre-analyzer tap → 2× upsample → 6 bands
→ downsample → post-analyzer tap → output gain. Dynamic gain is computed
per-sample inside each band; static parameters are read block-rate from the
APVTS. The editor never touches DSP state — it mirrors static parameters
from APVTS atomics for the curve display.

## Known v1 limitations (honest list)

* **Notch gain knob does nothing.** The RBJ notch has a fixed (theoretically
infinite) null; `gainDb` is unused for the Notch type. The editor disables
the gain knob when Notch is selected.
* **No dynamics on High Pass / Low Pass.** Those filters have no meaningful
gain-depth parameter, so the dynamic stage is inert for them (threshold
etc. are accepted but have no effect). Bell/Shelf/Notch-detector routing:
Bell/Notch use a band-pass detector, Low Shelf a low-pass detector,
High Shelf a high-pass detector.
* **Downward dynamics only** (cut on loud). No upward/expansion mode yet.
* **No mid/side or per-channel modes** — stereo-linked only.
* The display shows a dim **static** curve plus a bright **live** curve that
includes each band's current dynamic gain, so compression visibly pumps the
curve (FabFilter-style). The GR ring/readout remains as a per-band meter.

## Status

See `STATUS.md` for what's verified vs. what's written-but-not-yet-compiled.

