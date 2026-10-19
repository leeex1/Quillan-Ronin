# Game Audio in the Browser (Web Audio API, mobile unlock, Howler.js, gain buses, spatial, latency)

Consolidated from MDN Web Audio docs and Howler.js docs (see Sources). Focus: what an AI builder needs so audio **actually plays on mobile, stays low-latency, and mixes cleanly** — the #1 audio bug in browser games is "no sound on iOS."

---

## 1. The autoplay unlock — the single most important rule

Browsers block audio until the user interacts with the page. An `AudioContext` created on page load starts in the **`"suspended"`** state and will silently play nothing until resumed.

**Rule: create/resume the AudioContext (or unlock your audio lib) from inside the FIRST real user gesture** (`click`/`touchend`/`keydown`), and call `resume()` **synchronously** in that handler's call stack (iOS Safari is strict — an `await` before `resume()` can break the gesture chain).

```js
let audioCtx;
function unlockAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume(); // call synchronously in the gesture
}
// Attach once; remove after first success. `pointerdown`/`touchend`/`keydown` all count.
window.addEventListener('pointerdown', unlockAudio, { once: true });
window.addEventListener('keydown', unlockAudio, { once: true });
```

- Show a "tap to start" / "click to play" screen so the first gesture reliably unlocks audio (and any pointer lock / fullscreen you need). Games should not expect audio before the player clicks in.
- Also **resume on `visibilitychange`/focus** — contexts can get re-suspended when a tab is backgrounded on mobile.
- Listen to the `statechange` event if you need to reflect mute/paused UI.

---

## 2. Web Audio API vs `<audio>` vs Howler

- **Don't use `<audio>` / `Audio()` elements for game SFX.** They have high latency, limited simultaneous playback, and inconsistent mobile behavior. Fine for background music streaming only.
- **Web Audio API** is the correct foundation: decode once into an `AudioBuffer`, then fire many low-latency, overlapping `AudioBufferSourceNode`s. Gives you precise scheduling, gain buses, and spatialization.
- **Howler.js** is the recommended default library — it wraps Web Audio (with `<audio>` fallback), auto-handles the **mobile unlock**, sprite support, fades, spatial audio, and pooling. Use it unless you specifically need raw Web Audio control. (Note: `AudioBufferSourceNode` is one-shot — you create a new source per play; that's normal and cheap. Howler manages this for you.)

**Preload & decode up front:** `fetch → arrayBuffer → audioCtx.decodeAudioData` during a loading screen. Decoding on first play causes an audible hitch.

---

## 3. Gain buses (music / SFX / master) — do this from the start

Route everything through a small mixer graph so you can control category volumes and a master mute:

```
source → (per-sound gain) → sfxBus ─┐
music  → musicGain ────────────────┼→ masterGain → audioCtx.destination
```

```js
const master = audioCtx.createGain();
const musicBus = audioCtx.createGain();
const sfxBus   = audioCtx.createGain();
musicBus.connect(master); sfxBus.connect(master); master.connect(audioCtx.destination);
```

- Set volumes with **`gain.setTargetAtTime(value, audioCtx.currentTime, 0.02)`** or `setValueAtTime`, **not** a raw `gain.value =` mid-play — abrupt jumps cause clicks/pops. Ramp fades over ~10–50ms.
- Perceived loudness is logarithmic: map a 0–1 slider to gain with a curve (e.g. `gain = slider²`) so sliders feel linear.
- Howler equivalent: per-sound `volume`, category via `Howler` groups / separate `Howl` instances, and `Howler.volume()` for master.

---

## 4. Latency (keep audio tight)

- Prefer **Web Audio buffer sources** over `<audio>` for anything reactive — element playback latency ruins hit feedback.
- Optionally construct the context with **`{ latencyHint: 'interactive' }`** (default) for lowest latency; use `'playback'` only for pure music.
- **Reuse decoded buffers**; never `decodeAudioData` in the gameplay hot path.
- Schedule to `audioCtx.currentTime` for sample-accurate timing (e.g. rhythm games): `source.start(when)`. Don't rely on `setTimeout` for musical timing — use the Web Audio clock ("A Tale of Two Clocks").
- Avoid creating hundreds of nodes per frame; cap concurrent voices and stop/disconnect finished sources (set `source.onended` to disconnect) to avoid graph buildup.

---

## 5. Variation & layering (feel)

- **Pitch-randomize** repeated SFX so footsteps/gunfire don't sound robotic: set `source.playbackRate.value = 1 + (Math.random()*2-1)*0.1` (Howler: `rate`). Randomize volume slightly too.
- **Layer** a sound from multiple samples (thump + body + transient) — see `game-feel-juice.md`.
- **Round-robin / random pick** from a few variants of the same sound.
- Use **audio sprites** (one file, offset+duration segments) to cut HTTP requests and decode overhead — Howler supports `sprite` natively.

---

## 6. Spatial / positional audio (for 3D or top-down games)

- **`PannerNode`** positions a source in 3D relative to `audioCtx.listener`. Connect `source → panner → bus`. Update `panner.positionX/Y/Z` and `listener.positionX/Y/Z` (+ listener `forward`/`up`) each frame to match camera/player.
- `panningModel`: `'HRTF'` (realistic, headphone-friendly, costlier) or `'equalpower'` (cheap stereo). `distanceModel`: `'inverse'`/`'linear'`/`'exponential'` with `refDistance`, `maxDistance`, `rolloffFactor` for falloff.
- For simple **2D stereo pan** (left/right by screen x), a `StereoPannerNode` (pan −1..1) is cheaper and sufficient — don't reach for HRTF in a 2D game.
- Howler exposes `pos()`, `orientation()`, and `pannerAttr()` for this.
- Update positions in the render loop; set the position AudioParams via `.value` (or `setValueAtTime`) to avoid zipper noise on fast movement.

---

## 7. Formats & assets
- Use **compressed formats**: `.webm`/`.ogg` (Opus/Vorbis) with an `.mp3`/`.m4a` fallback for Safari. Howler takes an array of sources and picks a supported one.
- Keep SFX short and mono (mono halves size and works with panners); keep music stereo but compressed.
- Watch total download; long music tracks can stream via `html5: true` in Howler (uses `<audio>`, saves memory, slightly higher latency — fine for music).

---

## 8. Bug-prevention checklist
- **No sound on mobile/iOS** → didn't resume `AudioContext` synchronously inside a user gesture; add a tap-to-start unlock.
- **Silence after backgrounding tab** → context re-suspended; resume on `visibilitychange`/focus.
- **Clicks/pops on volume change or start/stop** → setting `gain.value` abruptly; ramp with `setTargetAtTime` and fade in/out a few ms.
- **Laggy hit sounds** → using `<audio>` elements; use Web Audio buffer sources.
- **Hitch on first play of a sound** → decoding at play time; preload + `decodeAudioData` during loading.
- **Robotic repeated sounds** → no pitch/volume variation; randomize `playbackRate`.
- **Growing memory / stuck voices** → not disconnecting finished sources or capping concurrent voices.
- **Works in Chrome, silent in Safari** → format not supported (provide ogg/opus **and** mp3/m4a) or async break before `resume()`.
- **No volume controls** → always expose master/music/sfx sliders + mute (accessibility + player expectation).

---

## Defaults to apply
- **Default to Howler.js** for generated games (handles unlock, sprites, fades, spatial, pooling) — fall back to raw Web Audio only when we need custom DSP/scheduling.
- **Always wire the mixer graph**: master + music + sfx gain buses with sliders and a mute, from the first commit. Map sliders through a `x²` curve.
- **Always add a "tap to start" gate** that unlocks audio (and pointer lock/fullscreen) on the first gesture, and re-resume on visibility change — kills the #1 "no audio on mobile" bug.
- **Preload + decode on the loading screen**, reuse buffers, pitch-randomize repeated SFX, ramp all gain changes. Provide ogg/opus + mp3 fallbacks.

---

## Sources
- MDN — Web Audio API Best Practices (autoplay, gesture, gain, controls): https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices
- MDN — `AudioContext.resume()`: https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/resume
- MDN — `BaseAudioContext.decodeAudioData()`: https://developer.mozilla.org/en-US/docs/Web/API/BaseAudioContext/decodeAudioData
- MDN — `GainNode` / `AudioParam.setTargetAtTime`: https://developer.mozilla.org/en-US/docs/Web/API/GainNode , https://developer.mozilla.org/en-US/docs/Web/API/AudioParam/setTargetAtTime
- MDN — `PannerNode` + Web audio spatialization basics: https://developer.mozilla.org/en-US/docs/Web/API/PannerNode , https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Web_audio_spatialization_basics
- MDN — `StereoPannerNode`: https://developer.mozilla.org/en-US/docs/Web/API/StereoPannerNode
- MDN — Autoplay guide for media & Web Audio: https://developer.mozilla.org/en-US/docs/Web/Media/Autoplay_guide
- Chris Wilson — "A Tale of Two Clocks" (Web Audio scheduling): https://web.dev/articles/audio-scheduling
- Howler.js — docs & repo: https://howlerjs.com/ , https://github.com/goldfire/howler.js
- Chrome autoplay policy for Web Audio: https://developer.chrome.com/blog/autoplay/#webaudio
