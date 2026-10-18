---
file_type: audio_engineering_guide
file_id: 33
domain: audio_production
status: active
tags: [audio, mixing, mastering, bandlab, vocal_chain, jdxx, studio_grade, signal_flow, psychoacoustics]
---

# 🎛️ JDXX Audio Engineering: Master Mixing & Mastering Signal Chains

Authoritative, studio-grade signal chains, exact numeric parameters, and psychoacoustic engineering rationales extracted from BandLab Studio.

---

## 🎙️ 1. Lead Vocal Tracking & Shaping Chain: `recording JDXX`

```
  [Analog Mic Input]
         │
         ▼
  [1. Noise Gate]           --> Eliminates room ambience, preamp hiss & floor rumble
         │
         ▼
  [2. Multi-Filter (HPF)]    --> Strips sub-sonic DC offset & plosives (<90 Hz)
         │
         ▼
  [3. Tape Simulator]       --> Analog odd/even harmonic saturation & transient rounding
         │
         ▼
  [4. De-Esser]             --> Pre-compression surgical sibilance suppression (6.8 kHz)
         │
         ▼
  [5. DIGI Comp (Stage 1)]  --> Primary dynamic control & peak containment (3.5:1 ratio)
         │
         ▼
  [6. Graphic / Studio EQ]  --> Frequency pocketing & vocal air shelf (+2.5 dB @ 6.4 kHz)
         │
         ▼
  [7. Studio Reverb]        --> Psychoacoustic space & dimensional depth (20% wet mix)
         │
         ▼
  [8. DIGI Comp (Stage 2)]  --> Smoothing glue & RMS leveler
         │
         ▼
    [Track Output]
```

### Module Parameters, Physical "WHAT", & Psychoacoustic "WHY":

| Order | Module | Status | Exact Parameter Readouts | Physical "WHAT" (Signal Action) | Psychoacoustic "WHY" (Engineering Rationale) |
|:---|:---|:---|:---|:---|:---|
| **1** | **Noise Gate** | **ON** | • Threshold: `-35.9 dB`<br>• Attack: `20 ms`<br>• Release: `250 ms` | Closes the audio path when signal drops below $-35.9\text{ dBFS}$, attenuating background noise by $-60\text{ dB}$. | **Why Position #1**: Must be placed first before compressors amplify low-level room reflections, PC fan noise, or headphone bleed. A 20ms attack prevents cutting off initial vocal consonants (plosives/fricatives), while a 250ms release prevents audible gating chatter at the tail of spoken syllables. |
| **2** | **Multi-Filter** | **ON** | • Type: `High-Pass (12 dB/oct)`<br>• Cutoff Freq: `85 Hz`<br>• Resonance ($Q$): `0.1`<br>• Gain: `0.0 dB` | Filters out all sub-audible frequencies below $85\text{ Hz}$ with a smooth second-order Butterworth slope. | **Why Position #2**: Eliminates mic handling thumps, AC hum, and proximity effect. High-passing *before* the compressor prevents low-end sub frequencies from triggering false compressor gain reduction, leaving the compressor to react purely to musical vocal frequencies. |
| **3** | **Tape Simulator** | **ON** | • Drive: `50%`<br>• Bias: `50%`<br>• Speed: `15 ips`<br>• Flutter: `0% (Clean)`<br>• Output Vol: `50%` | Generates 2nd and 3rd order harmonic overtones and applies gentle high-frequency tape compression. | **Why Position #3**: Digital recordings often sound sterile and harsh. Saturating the clean high-passed signal glues the vocal formants and rounds harsh transient spikes in the $3\text{--}5\text{ kHz}$ region, making the voice sound rich and analog before dynamic compression. |
| **4** | **De-Esser** | **ON** | • Target Freq: `6,825 Hz`<br>• Threshold: `-17.6 dB`<br>• Range / Reduction: `-6.0 dB`<br>• Mode: `Split-Band` | Dynamically attenuates harsh sibilant spikes ('s', 't', 'sh', 'ch') specifically around $6.8\text{ kHz}$. | **Why Position #4**: Sibilance must be controlled *before* EQ and primary compression. If uncompressed sibilance hits a compressor, the compressor squashes the entire vocal; if placed after an air-boost EQ, sibilance becomes ear-piercing. |
| **5** | **DIGI Comp (Stage 1)** | **ON** | • Threshold: `-18.0 dB`<br>• Ratio: `3.5:1`<br>• Attack: `15 ms`<br>• Release: `120 ms`<br>• Knee: `Soft (0.5)` | Reduces transient peaks by $3\text{--}5\text{ dB}$ with a soft knee and moderate VCA response curve. | **Why Position #5**: Provides fast, transparent peak containment. The 15ms attack lets the initial vocal attack punch through for clarity, while the 120ms release recovers naturally between syllable pauses. |
| **6** | **Graphic / Studio EQ** | **ON** | • 100 Hz: `0.0 dB`<br>• 200 Hz: `-1.5 dB`<br>• 400 Hz: `-1.0 dB`<br>• 800 Hz: `+1.0 dB`<br>• 1.6 kHz: `+1.5 dB`<br>• 3.2 kHz: `+2.0 dB`<br>• 6.4 kHz: `+2.5 dB` | Attenuates mud frequencies and boosts presence, intelligibility, and high-end air sheen. | **Why Position #6**: Carving $-1.5\text{ dB}$ at $200\text{--}400\text{ Hz}$ removes boxiness and prevents vocal masking over guitars/keys. Boosting $1.6\text{--}3.2\text{ kHz}$ enhances lyric intelligibility, while $+2.5\text{ dB}$ at $6.4\text{ kHz+}$ imparts modern polished vocal sheen. |
| **7** | **Studio Reverb** | **ON** | • Wet Mix: `20%`<br>• Room Size: `40% (Medium Room)`<br>• Tone: `60% (Damped Highs)`<br>• Pre-Delay: `25 ms` | Generates early reflections and a dense acoustic tail mimicking a treated vocal live room. | **Why Position #7**: Reverb places the dry vocal in a realistic 3D acoustic space. A 25ms pre-delay ensures the dry vocal hits the listener's ear first before the reverberation starts, keeping the vocal right in the front of the mix. |
| **8** | **DIGI Comp (Stage 2)** | **ON** | • Threshold: `-12.0 dB`<br>• Ratio: `2.0:1`<br>• Attack: `30 ms`<br>• Release: `200 ms`<br>• Knee: `Soft` | Applies gentle opto-style smoothing to the combined vocal and early reverb reflections. | **Why Position #8**: "Serial Compression" — instead of one compressor working aggressively (which sounds pumped and squashed), two gentle compressors share the load. Stage 2 glues the reverb tail to the vocal performance. |

---

## 🌌 2. Ambient Vocal Intro & Atmospheric Texture: `intro`

```
  [Vocal Input]
        │
        ▼
  [1. De-Esser]             --> Tames sibilance before long reverb/delays
        │
        ▼
  [2. Dimension B Chorus]   --> BBD analog stereo widening (Preset 03)
        │
        ▼
  [3. Studio Reverb]        --> Massive 45% wet cathedral tail
        │
        ▼
  [4. DIGI Comp (Stage 1)]  --> Heavy compression of wet wash (4:1 ratio)
        │
        ▼
  [5. D-Delay]              --> Tempo-synced ping-pong echoes (35% feedback)
        │
        ▼
  [6. DIGI Comp (Stage 2)]  --> Leveler to sustain ambient pad envelope
        │
        ▼
   [Track Output]
```

### Module Parameters & Engineering Rationale:
* **Dimension B (Preset 03)**: Uses bucket-brigade device (BBD) phase modulation to spread the mono vocal across $180^\circ$ of the stereo panorama without causing mono cancellation.
* **Studio Reverb (Mix: 45%, Size: 75%, Tone: 70%)**: Pushes the vocal deep into the acoustic background, creating a cinematic mood.
* **DIGI Comp Stage 1 (Thresh: -10dB, Ratio: 4.0:1, Attack: 10ms)**: Compresses the reverb tail so quiet vocal notes sustain into an ambient synthesizer-like pad.
* **D-Delay (Ping-Pong, 1/4 Dotted Sync, Feedback: 35%, High-Pass: 250Hz, Low-Pass: 5kHz)**: Bandpasses the echo reflections (Abbey Road delay trick) so delay repeats never collide with the main lead vocal frequency range.

---

## 💎 3. Master Bus Finalizer: `Mastering JDXX`

```
  [Mixdown Bus Sum]
         │
         ▼
  [1. Master Studio EQ]     --> Sub-sonic cleanup (<28Hz), low-end punch & top air
         │
         ▼
  [2. Exciter / Tape Sat]   --> Odd/even harmonic density & stereowise spread
         │
         ▼
  [3. Studio Reverb (Amb)]  --> Micro-glue acoustic room (14% wet, 15% size, 90% bright)
         │
         ▼
  [4. DIGI Comp (Bus)]      --> Mix glue VCA compressor (2:1 ratio, 30ms attack)
         │
         ▼
  [5. Vintage Limiter]      --> Brickwall True Peak loudness ceiling (-0.3 dB TP)
         │
         ▼
   [Commercial Master]
```

### Module Calibration & Commercial Mastering Standards:

| Module | Exact Settings | Purpose & Psychoacoustic Function |
|:---|:---|:---|
| **Master Studio EQ** | • HPF: `28 Hz (24 dB/oct)`<br>• Low Shelf: `+0.8 dB @ 60 Hz`<br>• Mid Notch: `-1.0 dB @ 320 Hz`<br>• Air Shelf: `+1.2 dB @ 12 kHz` | **Sub-bass cleaning & tonal balance**: Eliminates DC rumble that wastes limiter headroom. Tightens sub-bass punch, cleans out mid-bass boxiness, and opens up the master with a silky air shelf. |
| **Exciter / Saturator** | • Harmonics: `50%`<br>• Tune: `3.5 kHz`<br>• Stereowise Spread: `40%`<br>• Low/High Balance: `50/50` | **Harmonic Enrichment**: Synthesizes pleasant musical overtones in the upper-mid range, increasing perceived loudness across phone and laptop speakers without raising peak levels. |
| **Studio Reverb (Acoustic Glue)**| • Mix: `14%`<br>• Size: `15% (Micro Room)`<br>• Tone: `90% (Bright)` | **Master Bus Spatial Glue**: A subtle micro-room reverb creates a unified acoustic space for the disparate stems (drums, bass, vocals, synths), making the track feel like it was recorded in the same acoustic environment. |
| **DIGI Comp (Master Glue)** | • Threshold: `-14 dB`<br>• Ratio: `2.0:1`<br>• Attack: `30 ms`<br>• Release: `100 ms` | **Dynamic Glue**: Catches $1\text{--}2\text{ dB}$ of master peaks with a slow 30ms attack to preserve drum transients while bonding the instrumental and vocal tracks. |
| **Vintage Brickwall Limiter** | • Ceiling: `-0.3 dB True Peak`<br>• Enhance: `50%`<br>• Volume / Threshold: `-4.5 dB` | **Loudness Maximization & ISP Protection**: Brings average loudness to commercial target ($-14\text{ to }-10\text{ LUFS}$) while ensuring True Peak remains below $-0.3\text{ dBFS}$ to prevent clipping when streaming services convert to AAC/MP3. |

---

## 🎧 4. Specialized Stems & Track Calibration

### A. Stereo Spreading Chain (`Pan`)
* **Dimension B + Little Rock Saturator**: Adds gritty analog drive and stereo motion.
* **D-Delay + Gain Stage**: Level-matches wide panned elements so they do not overpower center-panned lead vocals.

### B. Adlib & Backing Vocal FX (`adlib`)
* **Graphic EQ**: High-pass @ $300\text{ Hz}$ and low-pass @ $5\text{ kHz}$ ("Telephone EQ").
* **FBK Compressor + VibeToo**: Aggressive compression with optical modulation for punchy, distinct backing shouts.

### C. Drums & Low-End Separation (`kick` & `hihats`)
* **Hi-Hats**: High-pass filtered at $450\text{ Hz}$, high shelf at $10\text{ kHz}$, and stereo micro-delay.
* **Kick / 808**: Sub-bass mono summation below $120\text{ Hz}$, surgical notch at $250\text{ Hz}$ to leave room for snare body and vocal fundamentals.
