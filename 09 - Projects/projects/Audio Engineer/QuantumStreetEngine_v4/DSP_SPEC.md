# Quantum Street Engine v4 — DSP spec & preset mapping

Derived from the Google Flow source you supplied (`QuantumMasteringConsole`, the
24-control mastering console) + the *Quantum Street Anthem* article. The VST
exposes only 8 host parameters, so the 24-control chain is folded into those 8.

## Signal chain (from the source, kept in this order)

1. Input trim + rumble high-pass (20–45 Hz, Butterworth Q=0.707)
2. **808 Tunneling**: sub band split @ 110 Hz -> lowpass -> `tanh(x*(1+drive*0.05))` saturation -> sub trim
3. **De-mud**: peaking @ 280 Hz (-3.5 dB, Q=1.8) + boxiness cut @ 520 Hz (-2 dB, Q=2)
4. **Presence**: peaking @ 3.4 kHz (+2 dB, Q=1.2)
5. **Shor Air**: high shelf @ 11 kHz (+3.5 dB) -> `x + 0.15*x^3` exciter
6. **Glue bus compressor**: threshold -14 dB, ratio 3.2:1, attack 25 ms, release 120 ms
7. **Brickwall**: tanh soft clip (4x oversampled in browser) -> limiter, ceiling -0.8 dBTP

## 8-parameter mapping (the plugin's host contract)

| id | Param | Drives | v3 default | v4 default | v4 rationale |
|---|---|---|---|---|---|
| 0 | Tunneling % | sub-band drive + sub trim | 40 | **55** | article: 808 must "tunnel through" phase cancellation |
| 1 | SuperposTime ms | ghost-transient tap spacing (superposition) | 125 | **125** | 1/8 note @ 120 BPM, musically neutral |
| 2 | Probability % | ghost-tap density / observer collapse | 65 | **60** | leaves headroom before the mix gets busy |
| 3 | EntangleCutoff Hz | M/S entanglement crossover (bass<->shimmer) | 4500 | **3400** | targets presence/hat region, not the whole top end |
| 4 | EntropyNoise % | Shor exciter + dithered noise floor | 18 | **12** | "quantum soul" should be a seasoning, not hiss |
| 5 | WaveDecay % | ghost-tap decay + glue release | 55 | **58** | keeps groove tails alive without smearing |
| 6 | OutputGain dB | final trim after limiter | 0 | **0** | never change level silently |
| 7 | Mix % | dry/wet of the whole chain | 85 | **55** | 85 % wet is destructive on a master bus |

## Genre presets (for the UI AUTO / PRESET buttons)

| Preset | Tunnel | Time | Prob | Cutoff | Entropy | Decay | Gain | Mix |
|---|---|---|---|---|---|---|---|---|
| 808 Heavy / Trap | 70 | 125 | 72 | 3400 | 40 | 60 | +1.5 | 75 |
| Vocal Pop & R&B | 25 | 187 | 35 | 4200 | 30 | 45 | 0.0 | 55 |
| Texas Rock | 35 | 95 | 45 | 2800 | 25 | 55 | +1.0 | 65 |
| EDM / Club | 80 | 63 | 85 | 5000 | 50 | 70 | +2.0 | 85 |
| Transparent Mastering | 12 | 125 | 15 | 3000 | 10 | 40 | 0.0 | 35 |

## AUTO mode (the "intelligence" the article asks for)

Not a static LFO. AUTO holds the chosen genre preset as a center point and
evolves the same parameters that the article says the track must decide for
itself:

- `Probability`: beat-locked slow drift, 0.30..0.80 normalized
- `EntropyNoise`: small drift, 0.15..0.45 normalized
- `SuperposTime`: locked to 1/8 note from the host BPM (`60000/BPM/2` ms)
- `Tunneling`: gentle pulse on strong beats

All moves are wrapped in `BPCFUI`/`EPCFUI` so Cakewalk sees one automation
gesture, not a stream of unrelated writes. Toggling AUTO off restores the
snapshot taken when it was enabled.
