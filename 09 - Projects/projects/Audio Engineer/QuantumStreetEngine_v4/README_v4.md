# Quantum Street Engine / Studio — v4 build

Built from the sealed `Quantum_Street_Engine_v3.fmx` bundle and the Google Flow
`QuantumMasteringConsole` source you supplied, plus the *Quantum Street Anthem*
article. Everything here is verified — see §4.

## 1. What's in this folder

| File | Purpose |
|---|---|
| `Quantum_Street_Engine_v4.fmx` | **Drop-in VST effect bundle** for Cakewalk. 8 host params, new DSP + UI. |
| `Quantum_Street_Studio_v4.html` | **Full 24-knob mastering console** with the Smart Mix Assistant. Self-contained; open in any browser. |
| `qse_dsp.c` / `qse_dsp.h` | C DSP core (no libc). Compiles to wasm32. |
| `qse_wasm.c` | ABI adapter matching the original module exactly. |
| `engine_v4.wasm` | Compiled replacement DSP (8,278 B). |
| `metadata_v4.json` | Corrected parameter defaults/ranges. |
| `build_v4.py` | Reproducible bundle assembler. |
| `DSP_SPEC.md` | 24→8 parameter mapping and preset table. |
| `test_mix.wav` | Synthetic test signal used for verification. |

The original v3 bundle is untouched at
`09 - Projects\projects\Audio Engineer\Quantum_Street_Engine_v3.fmx`.

## 2. What changed

### VST bundle (`v3` → `v4`)
- **DSP replaced.** The opaque AssemblyScript engine (19 KB) is replaced by a
  6 KB C core: rumble HP → 808 sub saturation → de-mud/box/presence → high shelf
  → Shor exciter → superposed ghost taps → M/S entanglement → entropy dither →
  glue compression → brickwall soft clip → output trim.
- **Stability bugs fixed** in the original: NaN blow-up at high output gain,
  unguarded denormals, partial `reset()`, hard-clip distortion.
- **Parameters recalibrated** to musical defaults (Mix 85 % → 55 %, entropy
  18 % → 12 %, cutoff 4500 → 3400 Hz, etc.) — see `DSP_SPEC.md`.
- **UI:** AUTO mode (BPM-locked evolving collapse) + 5 genre presets applied
  through real `SPVFUI`/`BPCFUI`/`EPCFUI` automation messages; dB-scaled meters
  with peak hold; buffer updates coalesced to one per animation frame.
- **ABI preserved exactly:** `init(f32)`, `getNumParams()=8`, `reset()`,
  `process(inL,inR,outL,outR,frames)`, `getAbiVersion()=1`, exported memory,
  params at `0x10000`, `meters`/`qTelemetry` buffers exposed at init.

### Web console (`Quantum_Street_Studio_v4.html`)
- All **24 controls** wired (the v3 source left `subTightness`, `mudQ`,
  `highSmooth`, `masterOutGain` and `softClipKnee` unused). Attack/release,
  crossover, Q and ceiling now actually change the audio.
- **Smart Mix Assistant** analyzes the loaded track (LUFS estimate, true peak,
  sub/mud/presence/air band energy, stereo correlation) and applies targeted
  parameter changes with an on-screen explanation.
- 5 genre presets, A/B bypass, real spectrum + peak-hold meters, and WAV export
  that renders the *same* graph offline.

## 3. How to test in Cakewalk

1. Copy `Quantum_Street_Engine_v4.fmx` into the location where your `.fmx`
   host wrapper expects it (same place you had v3).
2. In Cakewalk, rescan VST3 / reload the wrapper, then insert
   **Quantum Street Engine**.
3. Start with the `808 Heavy / Trap` preset, hit **AUTO**, and play a track.
4. For the full 24-knob experience, open `Quantum_Street_Studio_v4.html` in a
   browser, load a WAV/MP3, press **SMART MIX**, then **EXPORT WAV** and compare.

## 4. Verification performed (not claims — measured)

- WASM: `node test_engine.mjs` → exports correct, `getNumParams=8`,
  `getAbiVersion=1`, both buffers exposed, output finite, peak 0.3999,
  NaN guard passes with `OutputGain=500`, `reset()` OK.
- DSP stability: intermediate stage probe caught and fixed an unstable RBJ
  peaking-EQ denominator (poles on the unit circle); post-fix pre-compressor
  level 0.083 for a 0.3 input.
- UI: rendered in Chromium → 24 sliders, 7 panels, 5 presets, **0 console
  errors**. AUTO, preset cycling and SPVFUI/EPCFUI automation verified.
- Bundle: structure validator → valid, 8 params, correct magic, version 1.4.0.
- Web console: rendered with 0 errors; Smart Mix on a synthetic mix produced 8
  reasoned adjustments; WAV export produced a valid 705,644-byte file with
  signal present; gain staging corrected from +14 dB/clipped to +5 dB with
  0.54 peak (no clipping).

## 5. Caveats — read before relying on it

1. **Signature.** The `signature` field is preserved verbatim from v3. It is a
   128-hex-char provenance stamp from the generator (consistent with an
   Ed25519 signature) and cannot be regenerated without the generator's
   private key. If your wrapper *verifies* it, the v4 bundle will be rejected —
   tell me and I'll adapt (e.g. keep v3's payload layout and patch only the UI,
   or regenerate through the original tool). If it only checks presence/length,
   it will load. This is the single highest-risk item.
2. `process()` is capped at 4096 frames per call, matching v3 — your host must
   chunk blocks. (Standard VST hosts do.)
3. The web console uses Web Audio's `DynamicsCompressorNode`, which applies an
   undocumented makeup gain; v4 cancels it with fixed 0.596 compensation gains.
   Attack/release still work; makeup compensation is approximate.
4. The LUFS figure is a calibrated RMS estimate, not a full EBU R128 meter.

## 6. Rebuild

```powershell
$clang = "C:\Program Files\LLVM\bin\clang.exe"
$dir   = "C:\02_QUILLAN\09 - Projects\projects\Audio Engineer\QuantumStreetEngine_v4"
$args  = @('--target=wasm32','-O2','-nostdlib','-Wl,--no-entry',
          '-Wl,--export=init','-Wl,--export=getNumParams','-Wl,--export=reset',
          '-Wl,--export=process','-Wl,--export=getAbiVersion','-Wl,--export=memory',
          '-Wl,--allow-undefined','-Wl,--global-base=1048576',
          '-Wl,--initial-memory=2097152','-Wl,--max-memory=4194304',
          '-o',"$dir\engine_v4.wasm","$dir\qse_wasm.c","$dir\qse_dsp.c")
& $clang @args
python "$dir\build_v4.py"
```
