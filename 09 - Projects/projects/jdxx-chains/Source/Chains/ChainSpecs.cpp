/*
    JDXX Chains — the five chains (JDXX spec 2026-11-02)

    JDXX_CHAIN_2026_11_02. Sections map 1:1 onto the gameplan "4.2 .. 4.6".

    --- Assumptions ledger (verify by ear / sign-off list) ---
    A1 Gate:      attenuation floor fixed at -80 dB; downward gate (not
                  expander); hysteresis 6 dB; detector = max(|L|,|R|).
    A2 TapeSim:   drive 1.764 -> ~+4.9 dB into tanh; bias 50 -> centred;
                  wow rate = speed/24.94 * 0.55 Hz; wow depth 5 -> ~0.25%
                  pitch wobble (subtle); HF rounding fixed one-pole @ 16 kHz.
    A3 GraphicEQ7: fixed RBJ bells, Q = 1.0 (≈ one octave bandwidth).
    A4 DeEsser:   sidechain bandpass Q = 2.0; whole-band gain reduction
                  (not split-band); fixed 4:1; attack 1.5 ms; release 120 ms.
    A5 Compressor(DIGI): feedforward; soft knee = kneeDb; detector max(|L|,|R|).
       FBK Comp: feedback detector (one-sample lag on output); threshold
                  "20" interpreted as -20 dBFS (ASSUMPTION — BandLab FBK
                  displays positive units); makeup 8.6 dB; ratio 20:1.
    A6 Studio Reverb: comb sizes scale with `size` (0.5..2.0x); damping from
                  `tone`; stereo spread 23 samples; wet/dry as printed.
    A7 D-Delay:   time in SECONDS (tempo-independent); feedback (regen) HP/LP
                  one-pole filters inside the loop; out level per preset.
    A8 Dimension B "02"/"03": two fixed voicings (ASSUMPTION):
                  02 = 0.55 Hz, depth 1.6 ms, wet 35 %
                  03 = 0.80 Hz, depth 2.5 ms, wet 40 %
    A9 VibeToo:   100 % wet modulation (vibrato); amount -> delay depth.
    A10 Small Stone: 4 x 1st-order allpass; resonance lo/hi = 0.35/0.60 fb.
    A11 Stereowise: M/S widener; lower/upper intensity adds to S; crossover
                  2517.85 Hz; Butterworth cascade (~LR4 slope).
    A12 Exciter:  HP at `tune` -> tanh harmonics; threshold gate with 12 dB
                  smoothstep; mix in dB.
    A13 Vintage Limiter: zero-lookahead brickwall; internal threshold fixed
                  at -1 dBFS; Enhance = +3 dB shelf @ 3.5 kHz; Master Level
                  = output gain; Soft Knee = knee width. NO LUFS auto-gain.
    A14 Gain:     straight dB.
    A15 Pan chain: Tremolo spread = degrees of phase offset between L/R.
*/
#include "ChainSpecs.h"

namespace JdxxChains
{

// ---------------------------------------------------------------------------
// recording JDXX ("vocal presets for low end mic") — JDXX_CHAIN_2026_11_02 §4.2
// ---------------------------------------------------------------------------
static const ModuleSpec kVocalModules[] = {
    { ModuleType::Gate,        "NoiseGate", { -35.9f, 20.0f, 250.0f } },
    { ModuleType::HighPass,    "HPF",       { 100.0f, 0.2f } },
    { ModuleType::TapeSim,     "Tape",      { 1.764f, 50.0f, 0.0f, 24.9434f, 5.0f, 16000.0f } },
    { ModuleType::GraphicEQ7,  "EQ1",       { -1.0f, -3.0f, -3.3f, -1.0f, 0.5f, 3.0f, 2.0f, 0.0f } },
    { ModuleType::DeEsser,     "DeEsser",   { 6825.0f, -17.6f } },
    { ModuleType::CompressorFF,"Comp1",     { 14.0f, 3.0f, 4.0f, 150.0f, -15.0f, 0.0f, 0.0f } },
    { ModuleType::GraphicEQ7,  "EQ2",       { 0.0f, 0.0f, -2.2f, -0.3f, -2.1f, 0.0f, 0.0f, 0.0f } },
    { ModuleType::Reverb,      "Reverb",    { 20.0f, 40.0f, 60.0f } },   // raw mix 2, size 4, color 6
    { ModuleType::CompressorFF,"Comp2",     { 10.0f, 3.0f, 4.0f, 100.0f, -18.0f, 0.0f, 0.0f } },
};

// ---------------------------------------------------------------------------
// adlib — JDXX_CHAIN_2026_11_02 §4.3
// ---------------------------------------------------------------------------
static const ModuleSpec kAdlibModules[] = {
    { ModuleType::Gate,        "NoiseGate", { -44.0f, 20.0f, 200.0f } },
    { ModuleType::GraphicEQ7,  "EQ",        { -15.0f, -8.0f, 3.0f, 8.0f, 12.0f, 7.0f, 0.0f, 0.0f } },
    { ModuleType::CompressorFB,"FBKComp",   { 33.0f, 0.0f, 20.0f, 546.0f, -20.0f /* ASSUMPTION */, 8.6f, 1.0f } },
    { ModuleType::ChorusBBD,   "DimB03",    { 3.0f } },
    { ModuleType::DeEsser,     "DeEsser",   { 9865.0f, -25.6f } },
    { ModuleType::Vibrato,     "VibeToo",   { 57.0f, 1.19f } },
    { ModuleType::StereoDelay, "DDelay",    { 0.0f, 20.0f, 500.0f, 2000.0f, 5.0f, 1.29f, -3.0f } },
    { ModuleType::CompressorFF,"Comp",      { 18.0f, 6.0f, 4.0f, 222.0f, -51.0f, 0.0f, 0.0f } },
};

// ---------------------------------------------------------------------------
// intro (hypeVox basis) — JDXX_CHAIN_2026_11_02 §4.4
// ---------------------------------------------------------------------------
static const ModuleSpec kIntroModules[] = {
    { ModuleType::Gate,        "NoiseGate", { -44.0f, 20.0f, 200.0f } },
    { ModuleType::GraphicEQ7,  "EQ",        { -6.0f, -3.0f, 0.0f, 3.0f, 0.0f, -3.0f, -6.0f, 0.0f } },
    { ModuleType::DeEsser,     "DeEsser",   { 9865.0f, -25.6f } },
    { ModuleType::ChorusBBD,   "DimB03",    { 3.0f } },
    { ModuleType::Reverb,      "Reverb",    { 15.0f, 30.0f, 50.0f } },   // raw mix 1.5, size 3, color 5
    { ModuleType::CompressorFF,"Comp1",     { 10.0f, 5.0f, 4.0f, 200.0f, -10.0f, 0.0f, 0.0f } },
    { ModuleType::StereoDelay, "DDelay",    { 50.0f, 20.0f, 100.0f, 8000.0f, 15.0f, 1.4f, 0.0f } },
    { ModuleType::CompressorFF,"Comp2",     { 10.0f, 4.0f, 2.0f, 300.0f, -5.0f, 0.0f, 0.0f } },
};

// ---------------------------------------------------------------------------
// Pan (slowPan basis) — JDXX_CHAIN_2026_11_02 §4.5
// ---------------------------------------------------------------------------
static const ModuleSpec kPanModules[] = {
    { ModuleType::Tremolo,     "Tremolo",   { 0.89f, 0.209f, 140.0f } },
    { ModuleType::GraphicEQ7,  "EQ",        { -6.0f, -3.1f, -1.6f, -1.5f, -2.0f, -2.0f, 2.0f, 0.0f } },
    { ModuleType::Reverb,      "Reverb",    { 15.0f, 30.0f, 50.0f } },   // raw mix 1.5, size 3, color 5
    { ModuleType::ChorusBBD,   "DimB02",    { 2.0f } },
    { ModuleType::Phaser,      "SmallStone",{ 0.596f, 0.0f /* resonance low */ } },
    { ModuleType::StereoDelay, "DDelay",    { 70.0f, 20.0f, 100.0f, 8000.0f, 20.0f, 1.29f, 3.0f } },
    { ModuleType::GainStage,   "Gain",      { 6.0f } },
};

// ---------------------------------------------------------------------------
// Mastering JDXX — JDXX_CHAIN_2026_11_02 §4.6
// ---------------------------------------------------------------------------
static const ModuleSpec kMasteringModules[] = {
    { ModuleType::GraphicEQ7,  "EQ1",       { 0.2f, 0.4f, -1.0f, -0.6f, -1.0f, -0.4f, 1.0f, 0.0f } },
    { ModuleType::CompressorFF,"Comp1",     { 20.0f, 6.0f, 4.0f, 150.0f, -4.0f, 0.0f, 0.0f } },
    { ModuleType::Exciter,     "Exciter",   { 4000.0f, 10.0f, -4.0f, -15.5f } },
    { ModuleType::Widener,     "Stereowise",{ 0.21f, 2517.85f, 0.2f } },
    { ModuleType::GraphicEQ7,  "EQ2",       { 0.3f, -2.0f, 0.3f, 0.0f, 1.0f, 1.0f, 0.3f, 0.0f } },
    { ModuleType::Reverb,      "Reverb",    { 14.0f, 15.0f, 90.0f } },   // raw mix 1.4, size 1.5, color 9
    { ModuleType::Limiter,     "VintageLim",{ 0.65f, 762.5f, 7.5f, 3.0f, 3.0f, -1.0f } },
    { ModuleType::CompressorFF,"Comp2",     { 10.0f, 6.0f, 4.0f, 387.0f, -6.5f, 0.0f, 0.0f } },
};

static const ChainSpec kChainSpecs[] = {
    { "recording JDXX", "Vocals main 1 (and main 2 for adlibs — reused)", kVocalModules,     (int) (sizeof (kVocalModules)     / sizeof (kVocalModules[0])) },
    { "adlib",          "Vocals adlib",                                    kAdlibModules,     (int) (sizeof (kAdlibModules)     / sizeof (kAdlibModules[0])) },
    { "intro",          "Vocals intro",                                    kIntroModules,     (int) (sizeof (kIntroModules)     / sizeof (kIntroModules[0])) },
    { "Pan",            "Vocals pan",                                      kPanModules,       (int) (sizeof (kPanModules)       / sizeof (kPanModules[0])) },
    { "Mastering JDXX", "Master bus",                                      kMasteringModules, (int) (sizeof (kMasteringModules) / sizeof (kMasteringModules[0])) },
};

const ChainSpec& getChainSpec (ChainMode mode) noexcept
{
    int idx = (int) mode;
    if (idx < 0 || idx >= (int) ChainMode::NumModes) idx = 0;
    return kChainSpecs[(size_t) idx];
}

} // namespace JdxxChains
