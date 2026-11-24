# NextVerse DAW — Production Development Plan (2026)

**Document type:** engineering plan + evidence-based gap assessment
**Date:** 2026-11-23
**Scope:** `ROOT\nextverse-platform\daw` and its `ai-assistant` / `quantum-core` integration points
**ROOT:** `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main`
**Path convention:** every path in this document is relative to ROOT unless written with a drive letter. Section 9 gives the absolute form of every cited file.
**Evidence basis:** direct source reading only. No C++ build was executed (see section 6.1).

---

## 0. Executive assessment

The NextVerse DAW is **two disconnected prototypes**, not an audio engine.

1. `nextverse-platform/daw/conceptual_logic` is a C++20 **data-model simulator**. Its own README states it "does not involve actual audio processing, GUI, or real-time audio I/O" (`nextverse-platform/daw/conceptual_logic/README.md:9`). `Transport::processRealTimeAudio` walks track/clip lists, logs, and feeds a 512-sample buffer of silence to effects (`nextverse-platform/daw/conceptual_logic/src/Transport.cpp:17-60`); playback advances with a 100 ms `sleep_for` in a blocking loop (`:62-93`).
2. `nextverse-platform/daw/source/NextVerseDAW` is a **JUCE application shell** that constructs a Tracktion `Edit`, adds one test track, and clears the output buffer on every audio callback (`nextverse-platform/daw/source/NextVerseDAW/Source/MainComponent.cpp:82-86`). The vendored Tracktion engine is a single-line placeholder comment (`nextverse-platform/daw/source/NextVerseDAW/modules/tracktion_engine/tracktion_engine/tracktion_engine.h:1`).

There is no plugin hosting, no real-time thread, no lock-free messaging, no automation, no MIDI input path, no AI composition call, and no quantum-core call anywhere in the DAW code. The only quantum reference is a reference snippet that holds a `QuantumVM&` and multiplies a buffer by 0.5 (`reference_code_snippets/cpp/daw_grok2.hpp:13-35`).

**Confidence: high (0.93)** on the "what exists today" claims — they are direct line-level source reads. **Confidence: medium (0.78)** on the competitor capability descriptions in section 2 — they are general public product knowledge, not benchmarked on this host (no commercial DAW is installed or measured here). The engineering design in sections 4–7 is a proposal, not a description of existing code.

The domain class model is a usable seed. The platform layer must be rebuilt around a real audio callback before any parity claim is defensible.

---

## 1. Files examined and method

Read in full for this plan:

- Conceptual logic headers: `AudioTrack.h`, `MidiTrack.h`, `Transport.h`, `AudioEffect.h`, `ReverbEffect.h`, `DelayEffect.h`, `EditSession.h`, `EventTypes.h` under `nextverse-platform/daw/conceptual_logic/include/`.
- Conceptual logic sources: the eight matching `.cpp` files plus `main.cpp` under `nextverse-platform/daw/conceptual_logic/src/`.
- Build/description: `nextverse-platform/daw/conceptual_logic/CMakeLists.txt`, `nextverse-platform/daw/conceptual_logic/README.md`, `nextverse-platform/daw/source/NextVerseDAW/CMakeLists.txt`, `nextverse-platform/daw/source/NextVerseDAW/NextVerseDAW.jucer`.
- JUCE app: `Source/Main.cpp`, `Source/MainComponent.h`, `Source/MainComponent.cpp` under `nextverse-platform/daw/source/NextVerseDAW/`.
- Engine placeholder: `nextverse-platform/daw/source/NextVerseDAW/modules/tracktion_engine/tracktion_engine/tracktion_engine.h`.
- Platform context: `README.md`, `nextverse-platform/docs/quantum_enhancements.md`, `Q&A_Log.md`, `beta files/juce_daw_best_practices.md`, `beta files/Breaking Barriers with Joshua Don Lee Formulas.md`.
- Integration modules: `nextverse-platform/ai-assistant/README.md`, `nextverse-platform/ai-assistant/api/chat_api.py` (first 80 lines), `nextverse-platform/quantum-core/quantum_core.py`, `reference_code_snippets/cpp/daw_grok2.hpp`.

Method: static source inspection plus host tool discovery. No claim about runtime behavior is made where the code was not executed; those claims are marked as static findings.

---

## 2. 2026 competitive gap analysis

### 2.1 Feature comparison

"Prototype" means the code as read on 2026-11-23. Competitor columns describe current major product lines and their documented general capabilities; they are not lab measurements from this host.

| Capability | Ableton Live 12 | Logic Pro | FL Studio | REAPER | Bitwig Studio | NextVerse prototype (source evidence) |
|---|---|---|---|---|---|---|
| Multitrack audio record/playback | Yes | Yes | Yes | Yes | Yes | **No audio at all**. Tracks hold metadata/clip records only (`conceptual_logic/README.md:9-15`); callback clears silence (`.../source/NextVerseDAW/Source/MainComponent.cpp:82-86`). |
| MIDI note editing / piano roll | Yes | Yes | Strongest-in-class roll | Yes | Yes | **No editor**. `MidiEvent` is `{startTime, noteNumber, velocity, duration}` with no channel/CC/bend (`.../include/EventTypes.h:15-23`); `MidiTrack` only has `addEvent/getEvents` (`.../src/MidiTrack.cpp:28-35`). |
| Clip / session launching | Signature feature | Arrangement + Live Loops | Playlist clips | Item-based | Clip launcher | **Absent**. No clip-launch state machine; clips are `AudioClip` records with `startTime/duration/filePath` (`.../include/AudioTrack.h:18-35`). |
| Audio time-stretch / warp | Signature feature | Flex Time | Time stretching | Elastique-style modes | Warp/clip timestretch | **Absent**. No time-stretch code exists in either tree. |
| VST3 hosting | Yes | AU-first + VST3 in current lines | Yes | Yes | Yes | **Absent in production code**. `juce_audio_processors` is declared in the `.jucer` (`.../NextVerseDAW.jucer:19`) but no `AudioPluginFormatManager` or `AudioPluginInstance` is instantiated. |
| AudioUnits hosting | No (macOS host only in Logic) | Yes | Yes | Yes | Yes | **Absent**; macOS-only target and would require Xcode. |
| CLAP hosting | Growing support | Varies by version | Varies | Yes | Native | **Absent**; no CLAP code or dependency. |
| Built-in instruments/synths | Wavetable, Operator, Drift | Alchemy, Retro Synth | 3x Osc, Massive-style | ReaSynth | Polysynth, Fabric | **Absent**. `MidiTrack::instrumentName` is a string defaulting to `"Default Instrument"` (`.../include/MidiTrack.h:68`). |
| Effect DSP | Rich stock set | Rich stock set | Rich stock set | Rich stock set | Rich stock set | **Placeholder only**. Reverb is a gain/wet-dry multiply (`.../src/ReverbEffect.cpp:36-49`); delay is a single scalar ring buffer (`.../src/DelayEffect.cpp:76-91`). |
| Mixer / routing / buses | Yes | Yes | Yes | Yes | Yes | **Absent**. Only per-track float fields `volume/pan/isMuted/isSoloed` (`.../include/AudioTrack.h:139-142`), never read by the audio callback. |
| Automation (sample-accurate) | Yes | Yes | Yes | Yes | Yes | **Absent**. No automation lane, envelope, or parameter-timeline type exists. |
| MPE / MIDI 2.0 expression | Yes | Yes | Partial | Yes | Strong | **Absent**. Event struct has no channel or per-note expression fields (`.../include/EventTypes.h:15-23`). |
| Scripting / extensibility | Max for Live | Alchemy scripts + Control surface API | Python-ish plugins | ReaScript (Lua/EEL/Python) | JSFX / modulators | **Absent in DAW**; platform has Python `ai-assistant` but no bridge. |
| AI-assisted composition | Increasing third-party | Increasing third-party | Third-party | Third-party | Third-party | **Aspirational**. Root README claims "AI composition" (`README.md:39`); no code path connects `ai-assistant` to the DAW. |
| Low-latency claim | Device-dependent | Device-dependent | Device-dependent | Excellent reputation | Device-dependent | **Unmeasured target**. "5ms latency" is a README claim (`README.md:39`); no measurement code exists. |
| Cross-platform | Win/macOS | macOS only | Win/macOS | Win/macOS/Linux | Win/macOS/Linux | .jucer exports **Linux Make only** (`.../NextVerseDAW.jucer:24`); CMake aims wider but is non-functional as written (section 3.4). |
| Source availability | Closed | Closed | Closed | Closed | Closed | **Fully local/source-available**; this is the real differentiator, not feature parity. |

### 2.2 Where the prototype actually stands

By feature count, the prototype is **pre-alpha**: it satisfies none of the baseline DAW rows above. The conceptual layer equals a university data-structures exercise with console logging; the JUCE layer equals the default Projucer "Audio App" template with a Tracktion `Edit` construction attempt. The public `beta files/juce_daw_best_practices.md:15` estimate of "hundreds of hours for a minimal viable product, thousands for a full-featured DAW" is the correct order of magnitude.

### 2.3 The only credible differentiation

Do not compete on stock effects, warp, or mixer depth. Compete on the three things this repository already owns:

1. Local-first, source-available operation (`README.md:19-26`).
2. A real AI assistant module with an NLU/ethics stack and a local Flask-SocketIO API (`nextverse-platform/ai-assistant/README.md:13-53`, `.../api/chat_api.py:33-64`).
3. A quantum-inspired simulation core with defined JQLD/LVVM metrics (`nextverse-platform/quantum-core/quantum_core.py:58-125`, `:167-240`) — used as an **offline scheduling estimator**, never as audio DSP (section 5.2).

---

## 3. Architecture mapped to the existing classes

### 3.1 Conceptual data model (`nextverse-platform/daw/conceptual_logic`)

| Class / type | Already implements (source evidence) | Missing for DAW parity |
|---|---|---|
| `AudioClip` (`include/AudioTrack.h:18-35`) | `id`, `name`, `startTime`, `duration`, `filePath` value struct. | No gain, fades, warp markers, looping, pitch, channel count, or sample-offset bounds. No decoded audio buffer. |
| `AudioTrack` (`include/AudioTrack.h:45-157`, `src/AudioTrack.cpp:8-126`) | Name get/set, `AudioEvent` accumulation, full clip CRUD (`addClip` validates empty name and non-positive duration, `getClip` throws `std::out_of_range`, `updateClip`, `removeClip`), effect chain, four public mix fields. | No audio graph node, no bus/send structure, no meters, no per-block processing method, no automation. Raw `std::vector<AudioEffect*>` with deleting destructor (`:12-19`) creates ownership hazard (section 3.3). |
| `MidiTrack` (`include/MidiTrack.h:17-95`, `src/MidiTrack.cpp:5-49`) | Name, `MidiEvent` accumulation, effect chain, `instrumentName`, `midiChannel` (1–16), mix fields. | No note CRUD, no clip/region concept, no quantize, no CC/pitch-bend/MPE, no synthesizer link. `getTotalDuration` deliberately ignores MIDI events (`src/EditSession.cpp:58-63`). |
| `Transport` (`include/Transport.h:16-96`, `src/Transport.cpp:17-151`) | `play/stop/record/seek`, `getCurrentPosition/isPlaying/isRecording`, conceptual per-slice processing. | `processRealTimeAudio` only logs and processes a 512-sample silent dummy buffer (`:17-60`). `play()` blocks the caller with 100 ms sleeps and 0.1 s position increments (`:72-86`). `seek()` stops playback and does not resume (`:120-139`). No sample-clock, no audio-device callback, no tempo/beat grid. |
| `AudioEffect` (`include/AudioEffect.h:18-76`, `src/AudioEffect.cpp:3-36`) | Abstract `process(float*, float*, int)`, name, enable flag, logging helpers. | Single mono float pointer interface; no sample rate, channel count, parameter model, state, latency report, or plugin ABI. `using AudioBuffer = std::vector<float>` at `:9` is unused by the interface. |
| `ReverbEffect` (`include/ReverbEffect.h:11-38`, `src/ReverbEffect.cpp:12-54`) | Parameters `roomSize/damping/wetLevel/dryLevel`, null/size guards, disabled pass-through. | **Not a reverb**. Wet signal is `drySignal * 0.5f` (`:38`) followed by scalar gain tweaks (`:41-48`). No comb/allpass network, no impulse response, no tail, no state. |
| `DelayEffect` (`include/DelayEffect.h:12-57`, `src/DelayEffect.cpp:5-95`) | Working single-tap feedback delay ring buffer; sample-rate-aware sizing; feedback and dry/wet mix. | **Unsafe for a real-time thread**: `updateDelayBufferSize()` calls `std::vector::assign` and can be invoked from inside `process()` (`:64-66`), which allocates. No interpolation, stereo, tempo sync, or parameter smoothing. |
| `EditSession` (`include/EditSession.h:16-104`, `src/EditSession.cpp:4-68`) | Name, add/get audio and MIDI track vectors, `getTotalDuration` = max clip end, default 10 s when empty. | Tracks are stored **by value copies** (`:21-29`); no track IDs, ordering, tempo, time signature, undo/redo, save/load, or project file format. |
| `AudioEvent`, `MidiEvent` (`include/EventTypes.h:5-23`) | Plain timing/value records. | No MIDI channel, CC, aftertouch, pitch bend, note ID, or sample-accurate timestamp. Comments at `:9-12` and `:20-22` list these as future ideas. |
| `main.cpp` test runner (`src/main.cpp:15-135`) | Exercises session construction, clip/event adds, transport state, effect enable, direct effect processing. | No assertions, no test framework, no expected values; it is a demonstration, not a test suite. |

### 3.2 JUCE application layer (`nextverse-platform/daw/source/NextVerseDAW`)

| Component | Already implements (source evidence) | Missing for DAW parity |
|---|---|---|
| `NextVerseDAWApplication` (`Source/Main.cpp:15-114`) | `JUCEApplication` lifecycle, `MainWindow` as `DocumentWindow`, 800×600 default, native title bar, `START_JUCE_APPLICATION`. | No command-line handling, no project file association, no persistence, no plugin scan at startup. |
| `MainComponent` (`Source/MainComponent.h:17-47`, `Source/MainComponent.cpp:14-120`) | `AudioAppComponent`; constructs a Tracktion `Engine` and `Edit` with `Edit::Options`, temp-file location, and undo manager; creates one audio track via `createAudioTrack`; requests record permission; opens `setAudioChannels(0, 2)`; paints edit name and track count. | `prepareToPlay` ignores both arguments (`:75-80`). `getNextAudioBlock` only calls `clearActiveBufferRegion()` (`:82-86`) — the application is silent. No connection between the Tracktion `Edit` and the JUCE audio callback. No UI beyond three text labels. |
| Build descriptors (`NextVerseDAW.jucer:1-26`, `CMakeLists.txt:1-79`) | Declares C++20, JUCE modules including `juce_audio_processors`, a local Tracktion path, and Linux Make export; CMake sets warning flags and searches for JUCE under `ThirdParty/JUCE`. | See section 3.4 — both descriptors contain blocking defects. |

### 3.3 Critical static findings in the existing code

These are not style notes; they are correctness blockers that must be resolved before feature work.

1. **Double-delete / dangling effect pointers.** `AudioTrack::~AudioTrack` deletes every `AudioEffect*` (`src/AudioTrack.cpp:12-19`), while `EditSession::addAudioTrack` stores a **copy** of the track (`src/EditSession.cpp:21-24`). The compiler-generated copy constructor copies the raw pointer vector (`include/AudioTrack.h:149`). A track that owns effects and is copied into a session leaves two owners of the same pointers. `MidiTrack` has the identical pattern (`src/MidiTrack.cpp:9-16` and `:39-45` vs `src/EditSession.cpp:26-29`). Risk: **critical** (CWE-416/CWE-415).
2. **Allocation on the audio path.** `DelayEffect::process` may resize its internal `std::vector` (`src/DelayEffect.cpp:64-66` → `updateDelayBufferSize` at `:25-35`). Risk: **high** (glitches).
3. **Blocking transport.** `Transport::play` sleeps and blocks (`src/Transport.cpp:72-86`); it cannot sit behind a UI button in a responsive app, let alone an audio callback. Risk: **high**.
4. **No audio output.** The JUCE callback produces silence (`Source/MainComponent.cpp:82-86`). Risk: **critical** for any claim of a DAW.
5. **Engine dependency is a stub.** `modules/tracktion_engine/tracktion_engine/tracktion_engine.h:1` contains only `// Tracktion Engine placeholder`, and `Source/MainComponent.h:14` includes `<tracktion_engine/tracktion_engine.h>`. A real build cannot compile this include from the vendored stub. Risk: **critical**.

### 3.4 Build-descriptor defects (static analysis, confidence 0.82)

- `NextVerseDAW.jucer:21` declares Tracktion at `../../ThirdParty/tracktion_engine`, which resolves from the `.jucer` directory to `nextverse-platform/daw/ThirdParty/tracktion_engine`. `CMakeLists.txt:14` looks for JUCE at `nextverse-platform/daw/source/NextVerseDAW/ThirdParty/JUCE`. The actual placeholder lives at `nextverse-platform/daw/source/NextVerseDAW/modules/tracktion_engine/tracktion_engine/`. **All three locations differ.**
- `CMakeLists.txt:44-49` calls `juce_add_juce_project_from_jucer`. JUCE's documented CMake API provides `juce_add_gui_app`, `juce_add_console_app`, `juce_add_plugin`, and `juce_add_module`; a general `.jucer`-to-CMake converter is not part of that API. Expect a configure-time failure once JUCE is present.
- `find_package(JUCE CONFIG REQUIRED ...)` passes JUCE module names as components (`CMakeLists.txt:19-35`); JUCE's package config does not define those as find-package components. Expect the same configure failure.
- Only `LINUX_MAKE` is exported (`NextVerseDAW.jucer:24`) despite Windows being the host.

---

## 4. Audio engine design

### 4.1 Layering

Keep the conceptual data model as the **domain layer**, hardened per section 3.3. Add three layers above it:

1. **Engine layer (new classes, to be introduced under `nextverse-platform/daw/source/NextVerseDAW/Source/`)** — an audio graph, a sample clock, a parameter/automation store, and a real-time-safe event bus.
2. **Platform layer** — JUCE `AudioDeviceManager`, `AudioProcessorGraph`, plugin format manager, and UI components. This is where the existing `MainComponent` evolves from "clear the buffer" to "pull one block from the graph."
3. **Integration layer** — AI composition and JQLD/LVVM estimation, strictly off the audio thread, communicating with the engine through the same lock-free control bus as the UI.

No quantum or AI call may ever appear in a `getNextAudioBlock`-derived path.

### 4.2 Plugin hosting (VST3 / CLAP / AU)

- Use JUCE's `AudioPluginFormatManager` with `VST3AudioPluginFormat` for Windows/Linux/macOS and `AudioUnitPluginFormat` for macOS. The required module `juce_audio_processors` is already declared (`NextVerseDAW.jucer:19`), although no hosting code exists yet.
- CLAP: verify the vendored JUCE version's CLAP-hosting support before committing. If the pinned JUCE predates it, either upgrade JUCE or integrate a maintained CLAP host bridge. Do not claim CLAP support until a scan test loads a real CLAP plugin.
- Host lifecycle: scan in a background thread at startup, cache a plugin description list, instantiate on the message thread, and connect into an `AudioProcessorGraph`. Never scan or instantiate inside the audio callback.
- Validation: run `pluginval` on every in-house plugin before integration, as already recommended in `beta files/juce_daw_best_practices.md:27`.
- Licensing: the same reference document requires reviewing Tracktion licensing before depending on it (`beta files/juce_daw_best_practices.md:11`). Resolve that decision in Phase 0.

### 4.3 Real-time thread safety

Non-negotiable rules for the audio callback and everything it calls:

- No heap allocation, no `new`/`delete`, no `std::vector::assign`, no `std::string` construction. This directly prohibits the current delay resize (`src/DelayEffect.cpp:64-66`) in the RT path.
- No locks, mutexes, or blocking I/O; no logging (`DBG` included) — `beta files/juce_daw_best_practices.md:25,28` already warns about both.
- All buffers pre-allocated and pre-sized at `prepareToPlay`; use planar, contiguous, 32/64-byte-aligned float blocks.
- UI reads state via atomic snapshots or a queue consumed on a `Timer` at 30–60 Hz, never by polling per sample.
- Parameter changes cross into the audio thread only through the lock-free queue described in 4.4, stamped with a sample offset.
- Debug builds should override global `operator new` to assert/record allocations while the audio callback is active; this turns "no allocation" from a claim into a test.

### 4.4 Lock-free queues

Design contract (new engine component):

- Single-producer/single-consumer ring buffers, power-of-two capacity, `std::atomic<size_t>` head/tail with `alignas(64)` to prevent false sharing.
- One queue for MIDI events, one for UI/control messages, one for parameter changes; monotonic sequence counters rather than wrap-to-zero indices to avoid ABA ambiguity.
- Each message carries `{sampleOffset, type, payload}` so the audio thread can apply it at an exact sample within the current block.
- Consumer drains the full queue once per block, sorts by `sampleOffset`, and applies changes in order. No dynamic memory in the drain path; payloads are fixed-size POD.
- Overflow policy: drop newest MIDI with a counter, never block; expose the dropped-count through the UI snapshot so it is visible, not silent.

### 4.5 SIMD DSP

- Use `juce::FloatVectorOperations` and `juce::SIMDRegister` where they map cleanly; the root technology stack already names SIMD and Eigen (`README.md:70`).
- Keep DSP in deinterleaved planar buffers so inner loops are contiguous; process in block sizes that match the device callback (typically 64–256 frames).
- Align scratch buffers and use runtime CPU dispatch for SSE/AVX/NEON rather than compile-time-only flags, so one binary remains portable.
- Measure before optimizing. The LSSS formula explicitly models overhead as `ι_audio` (`beta files/Breaking Barriers with Joshua Don Lee Formulas.md:127-137`); treat that term as a measured quantity from block-time histograms, not an assumption.

### 4.6 5 ms latency target and how to measure it

**Definition lock (Phase 0 deliverable).** Two different numbers exist in this repository:

- `README.md:39` claims "5ms latency" for the DAW module.
- `beta files/Breaking Barriers with Joshua Don Lee Formulas.md:131` claims "96 kHz sampling with 10 effects at <50ms latency" and `:137` says "test for 96 kHz at <50ms".

These are not the same metric. This plan defines them separately:

- **Engine callback budget (hard target): 5 ms maximum block processing time.** At 96 kHz that is 480 samples; a 128-frame block is 1.33 ms of audio, leaving substantial headroom. At 48 kHz a 128-frame block is 2.67 ms.
- **End-to-end round trip (historical LSSS claim): below 50 ms**, device plus OS plus application, to be re-baselined on real hardware rather than inherited from a beta document.

Measurement procedure:

1. **Static budget** — record `samplesPerBlockExpected` and `sampleRate` in `prepareToPlay` (currently discarded at `Source/MainComponent.cpp:75-80`). Derive the device's theoretical base latency from the JUCE device settings.
2. **Instrumented block timing** — wrap the graph pull with a high-resolution monotonic clock (`juce::Time::getHighResolutionTicks` or `std::chrono::steady_clock`). Collect at least 100,000 block durations; report p50, p99, and max. Pass criterion: p99 < 5 ms and max < 5 ms on the reference device at both 48 kHz and 96 kHz. The `Q&A_Log.md:568-582` framework using nanosecond timestamps is the project's own precedent for this style of measurement.
3. **Xrun counting** — hook the audio device's underrun/overrun callback and require zero overruns over a 10-minute sustained playback with 8 tracks and 4 plugins per track. `Q&A_Log.md:587` already names "measure buffer underruns" as the DAW validation method.
4. **Loopback confirmation** — on hardware with a physical output-to-input loopback, play a known impulse and cross-correlate the recording to obtain true round-trip latency. This is the only measurement that validates the "<50 ms" claim.
5. **Stress conditions** — repeat steps 2–4 under CPU load and sustained thermal load, per `Q&A_Log.md:591-595` (variance reporting, load testing, thermal impact).
6. **Publish the report** with device name, driver, sample rate, block size, and all four percentiles. An unlabelled latency number is not evidence.

### 4.7 Sample-accurate automation

- Store automation as sorted `(sampleIndex, value)` segments in a lane owned by the engine, not as floating-point setters called from the UI.
- On each block, the audio thread walks only the segments intersecting `[blockStart, blockEnd)`, applies linear or equal-power interpolation, and splits the block at segment boundaries where the parameter is not smoothly variable.
- UI edits enter through the parameter queue (4.4) with a `sampleOffset`; the consumer inserts them into the lane without allocation (pre-allocated segment pool).
- Plugin parameter writes use the host's parameter API at the exact sub-block position; avoid per-sample host notification callbacks.
- Acceptance test: a ramp from 0.0 to 1.0 across exactly 480 samples at 96 kHz produces a sample-by-sample output within a documented tolerance of the analytic ramp.

### 4.8 MIDI and MPE

- Extend the conceptual `MidiEvent` (`include/EventTypes.h:15-23`) to carry: channel (1–16), note number, 14-bit velocity, duration in samples, note ID, CC number/value, channel pressure, pitch bend, per-note pitch bend, per-note pressure, and timbre. Preserve the existing field names where possible so `src/main.cpp:47-50` still compiles.
- At the platform layer, translate JUCE `MidiMessage`/`MidiBuffer` into and from these records at block boundaries.
- MPE zone logic: route per-note bend/pressure/timbre only to plugins that declare MPE support; fall back to channel-based expression otherwise. Document the fallback explicitly in the UI.
- `MidiTrack::midiChannel` (`include/MidiTrack.h:69`) becomes the default zone/channel rather than a hard-coded single channel.
- Acceptance test: a 4-note MPE phrase with independent pitch bends round-trips through the queue, the graph, and a test plugin with no note-ID collision and no lost per-note events.

---

## 5. AI composition and quantum DSP integration

### 5.1 AI composition via `ai-assistant`

Existing capability (all real code): the assistant has an `AIBrain` orchestrator, NLU with transformer-based sentiment/NER/intent and rule-based fallbacks, a `DialogueManager` with SQLite long-term memory, an `EthicalFramework` computing a DESS score, and a mock QVM interface (`nextverse-platform/ai-assistant/README.md:13-53`). `api/chat_api.py:33-64` exposes a Flask-SocketIO service with a globally instantiated `AIBrain`.

Integration design:

1. Add a composition-only request shape on the existing local service: the DAW sends a structured session summary (key, scale, tempo, time signature, target bar count, existing chord/clip statistics) derived from `EditSession` and `MidiTrack`.
2. The assistant returns a **schema-validated note list** expressed in the existing conceptual event vocabulary (`AudioEvent`/`MidiEvent` extended per 4.8). Reject any response that fails schema validation or violates the declared key/scale.
3. **Preview before insert.** Generated events appear as a provisional region; the user accepts or discards. Never let a generative model write directly into the playing timeline.
4. Run the request on a background thread; the audio thread only receives the final accepted event batch through the control queue (4.4). No HTTP or model inference in the callback.
5. Keep the local-first invariant (`README.md:19-26`): no cloud dependency for composition; the existing fallback-to-placeholder behavior (`nextverse-platform/ai-assistant/README.md:7,26`) means the DAW must degrade to manual entry, not fail.
6. Route the request through the existing ethical gate (DESS, `nextverse-platform/ai-assistant/README.md:34-39`) and log the decision for audit.

### 5.2 Quantum-core JQLD / LVVM — where applicable, and where not

`nextverse-platform/quantum-core/quantum_core.py:1-11` states explicitly that all metrics are **simulations of theoretical performance** and that no physical quantum hardware is addressed. That constraint governs everything here.

- **Applicable use:** offline estimation. `QuantumComputationalCore.calculateJQLD_detailed` computes `Q = C * 2^sum(N_jq * eta_j * theta_j) / omega_q` (`quantum_core.py:88-108`); `QuantumSuperVM.calculateLVVM_detailed` computes `VM_eff = Q * (R_vm + psi_vm * (1 - mu_vm)) / (1 + tau_vm)` (`:210-230`). `compound_speedup_report()` (`:278-303`) can score how many composition candidates fit a background budget. `QuantumSuperVM.submitTask` uses a `queue.Queue` and a daemon worker thread (`:183-186`, `:250-271`) — a legitimate off-thread scheduler pattern.
- **Not applicable:** the audio callback, DSP maths, mixing, plugin parameter writes, or latency measurement. There is no quantum acceleration of audio in this repository; claiming otherwise would be false.
- **Current wiring:** none. The only DAW-adjacent quantum code is `reference_code_snippets/cpp/daw_grok2.hpp:13-35`, which stores a `QuantumVM&` and applies a 0.5 gain in `process_audio`; its `AudioDeviceManager` initialization is commented out at `:16-21`. It is a reference snippet, not a build target.
- **Design rule:** any future quantum-core call from the DAW must go through the same background integration layer as AI composition, carry a measured wall-clock budget, and label its output "simulated" in the UI.

### 5.3 LSSS as a performance model

`A_mix = P_core * Delta_audio * (1 + theta_audio) / (1 + iota_audio)` is defined in `beta files/Breaking Barriers with Joshua Don Lee Formulas.md:126-137`. Use it as a **reporting model** whose `iota_audio` is filled from measured block-time overhead (section 4.6), not as a magic gain stage. The root README maps DAW → Auditory Cortex → LSSS (`README.md:39`); keep that mapping in documentation but do not present the formula as implemented DSP until a measured validation exists.

---

## 6. Build and test strategy

### 6.1 Host verification matrix

Host discovery on 2026-11-23: `cmake.exe` is present at `C:\Program Files\CMake\bin\cmake.exe`; `python.exe` is present at `C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe`; **no `cl`, `g++`, or `clang++` is on PATH**. Therefore nothing in the C++ trees has been compiled or run for this plan.

| Check | Possible on this host now | What it proves | What it cannot prove | Basis |
|---|---|---|---|---|
| File existence, completeness, exact contents | Yes | Which classes and methods actually exist; the Tracktion stub; declared modules | That any of it compiles | Direct reads, section 1 |
| Source-level API and ownership audit | Yes | Raw-pointer double-delete, blocking loop, delay resize, silent callback | Runtime manifestation or crash reproduction | Section 3.3 |
| Build-descriptor consistency inspection | Yes | Path mismatch across `.jucer`, CMake, and actual module location; non-standard CMake command | That configure succeeds (needs JUCE + compiler) | Section 3.4 |
| Python module import / unit tests for `quantum-core` | Yes (Python 3.14 present) | JQLD/LVVM formulas and queue behavior on this host | Any DAW behavior; no DAW import exists | `quantum_core.py:1-11,278-303` |
| Python tests for `ai-assistant` | Yes, subject to its dependencies (`nextverse-platform/ai-assistant/README.md:82-90,160-166`) | Assistant fallback behavior | DAW integration; no bridge exists | Same |
| Compile `ConceptualDAW_App` | **No** | Would validate the domain model | Prohibited by missing compiler | `conceptual_logic/CMakeLists.txt:12-24` |
| Configure/build JUCE app | **No** | Would validate CMake/JUCE integration | No compiler; JUCE SDK and real Tracktion source absent | `NextVerseDAW/CMakeLists.txt:14-49` |
| DSP unit tests (impulse response, feedback decay) | **No** | Numerical correctness | Needs compiler + test framework | `src/ReverbEffect.cpp`, `src/DelayEffect.cpp` |
| Thread-sanitizer / allocation instrumentation | **No** | Real-time safety | Needs instrumented compiler build | Section 4.3 |
| Latency and xrun measurement | **No** | The 5 ms / 50 ms claims | Needs compiled app + audio device | Section 4.6 |
| `pluginval` validation | **No** | Plugin hosting correctness | Needs compiled binaries + host | `beta files/juce_daw_best_practices.md:27` |

**Conclusion:** this plan is statically grounded. The first executable verification happens in Phase 0 on a host or CI runner with a compiler.

### 6.2 Cross-platform build plan (CMake / JUCE)

1. **Repair descriptors first.** Replace the unsupported `.jucer`-to-CMake call (`NextVerseDAW/CMakeLists.txt:44-49`) with a standard JUCE CMake target (`juce_add_gui_app`), add JUCE modules explicitly, and set `juce::juce_recommended_config_flags`. Decide whether Tracktion is vendored at one agreed path or removed; align `NextVerseDAW.jucer:21`, `CMakeLists.txt:14`, and the real module location from section 3.4 to a single source of truth.
2. **Split targets.** Keep `ConceptualDAW_App` (`conceptual_logic/CMakeLists.txt:24`) as a small, dependency-free target that any compiler can build in seconds. Make the JUCE app a separate target. Add a third test target for domain and DSP unit tests.
3. **Presets.** Provide CMake presets for Windows/MSVC, macOS/Clang-Xcode, and Linux/GCC-or-Clang with ALSA/JACK development packages. C++20 is already the declared standard in both CMake files (`conceptual_logic/CMakeLists.txt:5`, `NextVerseDAW/CMakeLists.txt:4`); keep it.
4. **Warnings to errors in CI.** The conceptual CMake already sets `-Wall -Wextra -Wpedantic` and `/W4` (`conceptual_logic/CMakeLists.txt:28-40`); add `-Werror` / `/WX` only in CI after the existing warnings are cleared.
5. **CI matrix.** Three OS runners, each fetching a pinned JUCE release, building all targets, and running the unit tests. Add a dedicated audio-capable bench runner for the latency procedure in 4.6; do not fuse it into PR CI.
6. **Static analysis.** Add clang-tidy and a compile-command database export; the project's own enhancement notes already call for clang-tidy and GoogleTest integration (`nextverse-platform/docs/quantum_enhancements.md:22-25`).
7. **Packaging.** Produce a standalone app plus, later, a plugin build with `COPY_PLUGIN_AFTER_BUILD TRUE` as recommended in `beta files/juce_daw_best_practices.md:24`.

### 6.3 Test layers

| Layer | Scope | Requires compiler |
|---|---|---|
| Domain unit tests | Clip CRUD, event storage, session duration, ownership/copy semantics (fix from 3.3-1) | Yes |
| DSP numeric tests | Reverb replaced by real algorithm; delay feedback decay; no allocation assertion | Yes |
| Concurrency tests | SPSC queue stress, dropped-event counters, sample-offset ordering | Yes |
| Integration tests | Plugin scan/load on VST3 and AU where available; graph render non-silent | Yes |
| Hardware bench | Block-time histogram, xrun count, loopback latency under load | Yes |
| AI contract tests | Composition JSON schema validation, key/scale constraints, ethical gate | No for Python-side tests |
| Static audit | Ownership, RT-safety rules, path consistency in build descriptors | No |

---

## 7. Phased roadmap, acceptance criteria, and definition of done

### Phase 0 — Buildability and definition lock

Work: repair build descriptors; pick one Tracktion path or drop Tracktion; vendor a pinned JUCE; lock the latency definitions from 4.6; establish CI with a compiler.

Acceptance criteria:
- `cmake -S nextverse-platform/daw/conceptual_logic -B <build-dir>` configures and `ConceptualDAW_App` builds and exits 0 on at least one compiler host.
- The JUCE target configures with a standard JUCE CMake API; no `juce_add_juce_project_from_jucer` remains.
- A written decision on Tracktion (vendor real source at one path, or remove the dependency) is recorded in the repository documentation.
- The 5 ms engine-budget and <50 ms round-trip definitions are published in the DAW docs.

Definition of done: a clean checkout builds on Windows and Linux CI runners.

### Phase 1 — Domain hardening

Work: replace raw owning effect pointers with unique ownership and defined copy/move semantics; add stable track/clip IDs; extend `MidiEvent` per 4.8; add tempo/time signature to `EditSession`; replace the console demo in `main.cpp` with assertion-based tests.

Acceptance criteria:
- No `delete` of an effect pointer in a track destructor and no by-value track copy of an owning container; ownership model documented.
- A session with effects on two tracks can be destroyed with no double-free under a sanitizer build.
- New MIDI fields are exercised by unit tests; existing `src/main.cpp:47-50` initialization remains valid or is migrated.
- `getTotalDuration` accounts for MIDI event extents instead of the `(void)track` placeholder at `src/EditSession.cpp:58-63`.

Definition of done: domain unit tests pass on all three CI OSes.

### Phase 2 — Real-time audio core

Work: implement the graph and sample clock; make `prepareToPlay` use its arguments; replace the silent `getNextAudioBlock` with a graph pull from at least one audio clip; introduce the allocation-free parameter and MIDI queues; replace `ReverbEffect` with a real algorithm and remove the delay resize from `process()`.

Acceptance criteria:
- Non-silent, correct-length output from one loaded clip with no xruns.
- Block-time p99 and max below 5 ms over 100,000 blocks at 48 kHz and 96 kHz.
- Zero xruns over 10 minutes with 8 tracks.
- Debug allocation guard records zero allocations inside the audio callback.
- Delay and reverb pass numeric tests.

Definition of done: the latency report from 4.6 step 6 is published with hardware details.

### Phase 3 — Plugin hosting, mixing, and automation

Work: VST3 hosting everywhere; AU on macOS; CLAP only after version verification; mixer buses/sends; sample-accurate automation lanes per 4.7.

Acceptance criteria:
- Scan and instantiate at least three third-party VST3 plugins; load one AU on macOS if a runner is available.
- 8 tracks × 4 plugins render without xruns under the Phase 2 bench.
- The 480-sample ramp test in 4.7 passes within tolerance.
- Any in-house plugin passes `pluginval` before integration.

Definition of done: plugin scan, load, and automation results are captured in CI and bench reports.

### Phase 4 — MIDI, MPE, and editing workflows

Work: piano roll, note CRUD, quantize, CC/pitch-bend lanes, MPE zone routing; transport rewritten to be non-blocking and driven by the audio clock; undo/redo.

Acceptance criteria:
- The 4-note MPE round-trip test in 4.8 passes with no lost events.
- `Transport::play` no longer sleeps; play/stop/seek respond within one block.
- Undo/redo restores note, clip, and automation edits.

Definition of done: a user can record, edit, and play back an MPE performance without a UI freeze.

### Phase 5 — AI composition and quantum offload

Work: composition endpoint on the existing assistant service; schema-validated preview/accept flow; background scheduler using `compound_speedup_report()` only as an estimator; QA audit proving no AI or quantum call on the RT thread.

Acceptance criteria:
- Generated note list conforms to schema and declared key/scale in 100 test requests.
- The assistant running in fallback mode still leaves the DAW fully usable.
- Code audit plus instrumentation shows zero AI/quantum calls in the audio callback path.
- Every simulated metric is labelled "simulated" in the UI, consistent with `quantum_core.py:9-10`.

Definition of done: an offline, local-only composition session completes end to end with user preview.

### Phase 6 — Hardening and release

Work: cross-platform polish, accessibility, full documentation, performance report, plugin validation, and a release build.

Acceptance criteria:
- All CI matrices green; `pluginval` clean; published latency and xrun report; no open critical findings from section 3.3.
- Documentation updated to reflect the shipped feature set, with no claim that exceeds the evidence in section 2.

Definition of done: a versioned standalone release is produced for Windows, macOS, and Linux.

### Global definition of done

A phase is complete only when all of the following hold:

1. Every acceptance criterion above has a named check and an observed result.
2. Every result is tied to a file path, a test ID, or a measurement report.
3. No critical or high finding from section 3.3 remains open.
4. No unlabelled performance claim is published.
5. Audio-thread rules from 4.3 are verified by instrumentation, not assertion.
6. Local-first operation is preserved (`README.md:19-26`).
7. C2-VIR ethical gate and C13-WARDEN security gate are satisfied for any AI or networked feature.
8. Documentation matches code at the line level.

---

## 8. Risks and mitigations

| Risk | Evidence | Impact | Mitigation |
|---|---|---|---|
| No C++ compiler on the authoring host | `cmake.exe` and `python.exe` found; `cl`, `g++`, `clang++` absent (2026-11-23) | Nothing C++ can be executed locally; false "it works" claims | Static checks now; all compile/test gates delegated to Phase 0 CI runners with compilers |
| Tracktion engine is a one-line placeholder | `modules/tracktion_engine/tracktion_engine/tracktion_engine.h:1` | Build cannot resolve the include at `Source/MainComponent.h:14` | Vendor real engine at one path or remove dependency; decision is a Phase 0 gate |
| Build descriptors use a non-standard API and three different module paths | `NextVerseDAW/CMakeLists.txt:14,19-35,44-49`; `NextVerseDAW.jucer:21` | Configure-time failure even with JUCE installed | Replace with standard JUCE CMake targets; single source of truth for paths |
| Double-delete of effect pointers | `src/AudioTrack.cpp:12-19` + `src/EditSession.cpp:21-24`; same for MIDI | Memory corruption, crashes | Replace raw owning pointers with unique ownership; sanitizer test in Phase 1 |
| Reverb is only a gain stage | `src/ReverbEffect.cpp:36-49` | Any audio-quality claim is false | Implement a real comb/allpass or convolution reverb; numeric tests in Phase 2 |
| Delay allocates inside `process()` | `src/DelayEffect.cpp:64-66` → `:25-35` | Audio glitches / xruns | Pre-size at `prepareToPlay`; remove all resizing from RT path; allocation guard |
| Blocking transport loop | `src/Transport.cpp:72-86` | UI freeze; unusable transport | Drive transport from audio clock; non-blocking state machine in Phase 4 |
| 5 ms vs <50 ms target ambiguity | `README.md:39` vs `beta files/Breaking Barriers with Joshua Don Lee Formulas.md:131` | Unverifiable or contradictory marketing claim | Lock two distinct definitions in Phase 0; publish measured report |
| Quantum acceleration misrepresentation | `quantum_core.py:9-10` explicitly labels simulation | Ethical/reputational failure; false advertising | Offline estimator only; label "simulated" in UI; no RT quantum calls |
| AI-generated notes are musically or Ethically invalid | `nextverse-platform/ai-assistant/README.md:34-39` (DESS); fallback behavior at `:7,26` | Bad user output; dependency on model availability | Schema + key/scale validation, DESS gate, human preview, graceful fallback |
| macOS AU and Xcode cannot be tested on this Windows host | `NextVerseDAW.jucer:24` exports Linux Make only | AU support unverified | Dedicated macOS CI runner; mark AU unsupported until green |
| CLAP hosting support depends on JUCE version | No CLAP dependency exists in either build descriptor | Wasted work or broken plugin scan | Verify vendored JUCE capability in Phase 0; add only with a passing load test |
| Tracktion licensing | `beta files/juce_daw_best_practices.md:11` requires review | Legal blocker | Resolve in Phase 0 decision; prefer removal if terms conflict with local-first sovereignty |
| Scope exceeds available effort | `beta files/juce_daw_best_practices.md:15` ("hundreds to thousands of hours") | Stalled project | Ship the narrow MVP from Phases 0–3 first; AI/quantum differentiation in Phase 5 only |

---

## 9. Evidence index — absolute paths

Every source claim in this document resolves to one of these files.

### Conceptual logic
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\AudioTrack.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\AudioTrack.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\MidiTrack.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\MidiTrack.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\Transport.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\Transport.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\AudioEffect.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\AudioEffect.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\ReverbEffect.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\ReverbEffect.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\DelayEffect.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\DelayEffect.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\EditSession.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\EditSession.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\include\EventTypes.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\src\main.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\CMakeLists.txt`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\conceptual_logic\README.md`

### JUCE application and build descriptors
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\source\NextVerseDAW\Source\Main.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\source\NextVerseDAW\Source\MainComponent.h`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\source\NextVerseDAW\Source\MainComponent.cpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\source\NextVerseDAW\NextVerseDAW.jucer`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\source\NextVerseDAW\CMakeLists.txt`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\daw\source\NextVerseDAW\modules\tracktion_engine\tracktion_engine\tracktion_engine.h`

### Integration modules and platform context
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\README.md`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\Q&A_Log.md`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\reference_code_snippets\cpp\daw_grok2.hpp`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\ai-assistant\README.md`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\ai-assistant\api\chat_api.py`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\quantum-core\quantum_core.py`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\nextverse-platform\docs\quantum_enhancements.md`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\beta files\juce_daw_best_practices.md`
- `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main\beta files\Breaking Barriers with Joshua Don Lee Formulas.md`

---

## 11. 2026-11-23 milestone: real JUCE audio graph (verified)

A real JUCE 8.0.8 source checkout was downloaded and wired into the build with
`-DJUCE_SOURCE_DIR=<path>`. The invalid `juce_add_juce_project_from_jucer` call was replaced
with standard `juce_add_gui_app` / `juce_add_console_app`. A JUCE-only audio graph was added:

- `Source/NextVerseToneGenerator.h` — deterministic 440 Hz stereo signal source (`AudioProcessor`).
- `tests/juce_audio_graph_test.cpp` — connects the generator to the graph output node and
  renders a 512-frame block offline.
- CMake target `NextVerseDAW_AudioGraphTest`, registered with CTest.

Verified with MSVC Release:

```
1: [PASS] JUCE audio graph connected 1 generator to 2 outputs
1: [PASS] JUCE audio graph offline render peak=0.249999
1: [PASS] JUCE audio graph verification complete
1/1 Test #1: NextVerseDAW_AudioGraphTest ......   Passed
```

Additional verified plugin contract (same JUCE target): `Source/NextVerseSynthProcessor.h`
exposes an automatable gain parameter and accepts MIDI note on/off. Offline verification proves
gain 0.5 produces peak 0.125, gain 0 mutes the voice, note-on starts it and note-off stops it.

```
1: [PASS] JUCE plugin parameter automation (gain 0.5 -> peak 0.125, gain 0 -> silence)
1: [PASS] JUCE MIDI note on/off gating verified
```

The GUI target remains gated on a real Tracktion Engine checkout: the vendored header is still
a one-line placeholder, so CMake warns and skips `NextVerseDAW` instead of failing cryptically.
This is a material step beyond the old silent simulator — the audio graph now carries real
samples under JUCE — but it is not competition parity: plugin hosting, automation, MIDI/MPE
and a shipped GUI are still outstanding.

## 12. Closing note

The fastest credible path is not "add features to the prototype." It is: make the repository buildable (Phase 0), make the data model safe (Phase 1), and produce one audible, measured block of audio (Phase 2). Everything else — plugins, MPE, AI composition, quantum-inspired scheduling — is only meaningful after those three gates pass. Engineering judgment: do not schedule feature work until a compiler has validated the foundation.
