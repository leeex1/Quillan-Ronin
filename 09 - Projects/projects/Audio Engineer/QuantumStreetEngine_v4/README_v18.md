# Quantum Street Studio v18 — LLM Ears integrated

File: `Quantum_Street_Studio_v18.html`.

## What's new

Your `LLM Ears.py` SynesthesiaEngine is now reachable from the browser through
a small local HTTP bridge:

- `llm_ears_bridge.py` — stdlib server on `127.0.0.1:8790`, CORS enabled,
  `GET /analyze?path=<file>` → `{ ok, bpm, texture, centroid_hz, segments }`.
- `LLM_Ears_compat.py` — loads acoustics on **CPU via librosa** (no Whisper at
  startup) and lazily loads Whisper **forced to CPU** for optional transcription.

The GTX 1050 (sm_61) has no CUDA kernel in the installed torch build, so GPU
Whisper failed with `no kernel image available`; forcing CPU fixed it.

**`LLM EARS · LISTEN`** button: analyse the file path in the input box, then:

- **BPM** from librosa replaces the in-app estimator for compressor release
  timing (`60000/BPM × 0.35` for comp A, `× 0.9` for comp B).
- **Spectral centroid / texture** sets de-ess, de-harsh, air boost and reverb LP.
  These survive subsequent deep-tune passes (previously they were overwritten).
- Results show in the ears readout and status bar.

## Why it mattered

On your track, the in-app BPM estimator reported **97 BPM**; LLM Ears/librosa
reported **144.23 BPM** with a centroid of **4258.8 Hz** ("bright / treble-
forward"). Compressor release timing was therefore wrong before — now it is
derived from the real tempo.

## Verified

- Bridge: HTTP 200 with `{bpm:144.23, texture:'bright...', centroid_hz:4258.8}`.
- Full mix after ears: deSess 2.5, highSmooth 30, reverb LP 6000, comp release
  175 ms — all preserved through deep tune; 8/12 pass; 0 console errors.
- All v17 features intact: 28-band LTAS, second spectrum analyzer, coupled
  controls, 12-section daemon, 38-control deep tune, post-export verification,
  TPDF dither, level-matched A/B.

## Start the bridge

```
python "C:\02_QUILLAN\09 - Projects\projects\Audio Engineer\QuantumStreetEngine_v4\llm_ears_bridge.py"
```

If the bridge is not running, the button reports it and the rest of the app
keeps working with its own measurements.

## Honest limit

librosa's BPM/centroid are acoustic estimates, not human judgment. "Bright" can
be intentional. The app reports the raw numbers so you can override them.
