/*
    JDXX Chains â€” clean-room DSP modules for the five JDXX chains.

    AGPL-3.0. Derivative of RoninEQ (AGPL-3.0). Every module here is an
    original implementation of the behavior described by the JDXX chain
    specification (2026-11-02 extraction of the live presets); no BandLab
    code or internal model is used or copied.

    Pure C++17, no JUCE dependency. Zero oversampling anywhere (the JDXX
    chains do not need it): the plugin reports 0 samples of latency so the
    host's delay compensation is unaffected.
*/
#pragma once

#include "DSP/Biquad.h"
#include "DSP/EnvelopeFollower.h"
#include <array>
#include <cmath>
#include <vector>
#include <string>

namespace JdxxChains
{

enum class ModuleType
{
    Gate,          // Noise Gate â€” threshold/attack/release, deep attenuation
    HighPass,      // RBJ high-pass biquad
    TapeSim,       // Soft saturation + HF rounding + gentle wow/flutter
    GraphicEQ7,    // Fixed 7-band peaking EQ (100..6400 Hz) + output level
    DeEsser,       // Sidechain bandpass detection + downward GR
    CompressorFF,  // Feedforward compressor (DIGI Comp)
    CompressorFB,  // Feedback compressor (FBK)
    Reverb,        // Schroeder/Freeverb-style stereo reverb
    StereoDelay,   // Stereo delay w/ feedback HP/LP filters
    Tremolo,       // Amplitude modulation, stereo spread
    ChorusBBD,     // "Dimension B" â€” two fixed BBD voicings (02/03)
    Vibrato,       // "VibeToo" â€” pitch modulation
    Phaser,        // "Small Stone" â€” 4-stage phaser, resonance low/high
    Widener,       // "Stereowise" â€” M/S intensity split at upperCrossoverHz
    Exciter,       // Harmonic generator around `tuneHz`, above-threshold only
    Limiter,       // Brickwall-style soft limiter with presence shelf ("Enhance")
    GainStage      // Plain output gain
};

// Uniform parameter carrier. Field meanings per ModuleType are documented in
// docs/JDXX_CHAIN_SPEC.md and next to each class below. Unused fields are 0.
struct ModuleParams
{
    float a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f, e = 0.0f, f = 0.0f, g = 0.0f;
    float h = 0.0f, i = 0.0f, j = 0.0f, k = 0.0f, l = 0.0f, m = 0.0f, n = 0.0f;
};

struct ModuleSpec
{
    ModuleType type;
    const char* name;    // stable instance name, e.g. "EQ1", "Comp2"
    ModuleParams p;
};

struct Module
{
    virtual ~Module() = default;
    ModuleType type = ModuleType::GainStage;
    std::string name;

    // Host-visible bypass (smoothed). When false, module is skipped with a
    // short fade to avoid clicks.
    bool enabled = true;

    // Stash parameter values from a spec. Called before prepare() at build
    // time, and again whenever a host parameter changes after that.
    // Implementations only copy values; all coefficient math happens in
    // prepare() / updateCoeffs() so changes are always consistent.
    virtual void configure (const ModuleParams& p) noexcept = 0;

    virtual void prepare (double sampleRate) = 0;
    virtual void reset() noexcept = 0;
    // In-place, interleaved L/R planes. nFrames == number of samples per channel.
    virtual void processBlock (float* L, float* R, int nFrames) noexcept = 0;
};

// Factory: builds a prepared module from a spec.
Module* createModule (const ModuleSpec& spec, double sampleRate) noexcept;

// Destroys a module created by the factory.
void destroyModule (Module* m) noexcept;

// ---------------------------------------------------------------------------
// Individual modules (public so tests can poke internals if needed)
// ---------------------------------------------------------------------------

// Noise Gate: p.a = threshold dB, p.b = attack ms, p.c = release ms
class NoiseGate final : public Module
{
public:
    NoiseGate() { type = ModuleType::Gate; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float thresholdDb = -35.9f, attackMs = 20.0f, releaseMs = 250.0f;
    double sr = 48000.0;
    EnvelopeFollower env;
    float gain = 1.0f;             // smoothed gate gain
    float gainAttackCoef = 0.0f, gainReleaseCoef = 0.0f;
    bool open = true;              // hysteresis state
    float floorDb = -80.0f;        // attenuation floor when fully closed
};

// High-pass: p.a = freq Hz, p.b = Q
class HighPass final : public Module
{
public:
    HighPass() { type = ModuleType::HighPass; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override { hpL.reset(); hpR.reset(); }
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float freqHz = 100.0f, q = 0.2f;
    double sr = 48000.0;
    Biquad hpL, hpR;
};

// TapeSim: p.a = drive (x gain into tanh), p.b = bias (-1..1 asymmetry),
//          p.c = out level dB, p.d = speed scale (affects wow rate; 24.94 â‰ˆ 1x),
//          p.e = wow depth (5 â‰ˆ 5 % detune), p.f = HF rounding cutoff Hz
class TapeSim final : public Module
{
public:
    TapeSim() { type = ModuleType::TapeSim; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float drive = 1.764f, bias = 0.5f, outLevelDb = 0.0f;
    float speed = 24.94f, wowDepth = 5.0f, hfCutHz = 16000.0f;
    double sr = 48000.0;
    double phase = 0.0;            // wow LFO phase
    float lp1L = 0.0f, lp1R = 0.0f;         // HF rounding one-pole
    std::vector<float> delayL, delayR;      // wow delay lines
    int delaySize = 0, delayPos = 0;
    float maxDelayMs = 4.0f;
};

// Graphic EQ 7 bands, RBJ peaking bells + level.
// p.a..p.g = gains dB @ 100,200,400,800,1600,3200,6400 Hz; p.h = level dB
class GraphicEQ7 final : public Module
{
public:
    GraphicEQ7() { type = ModuleType::GraphicEQ7; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    std::array<float, 7> gains{};
    float levelDb = 0.0f;
    double sr = 48000.0;
    static constexpr std::array<float, 7> freqs { 100.0f, 200.0f, 400.0f, 800.0f,
                                                  1600.0f, 3200.0f, 6400.0f };
    std::array<Biquad, 7> bandsL, bandsR;
    float levelGain = 1.0f;
    void updateCoeffs();
};

// DeEsser: p.a = freq Hz, p.b = threshold dB; fixed 4:1, fast attack, 100 ms rel.
class DeEsser final : public Module
{
public:
    DeEsser() { type = ModuleType::DeEsser; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float freqHz = 6825.0f, thresholdDb = -17.6f;
    double sr = 48000.0;
    Biquad bpL, bpR;               // constant-0 dB peak sidechain
    EnvelopeFollower env;
    float grDb = 0.0f;             // current gain reduction (positive dB)
    float atkCoef = 0.0f, relCoef = 0.0f;
    float applyGrL = 1.0f, applyGrR = 1.0f;
    float smooth = 0.0f;
};

// Compressor, shared FF/FB implementation.
// p.a = attack ms, p.b = knee dB, p.c = ratio, p.d = release ms,
// p.e = threshold dB, p.f = makeup dB, p.g = mode (0 = feedforward, 1 = feedback)
class Compressor final : public Module
{
public:
    Compressor() = default;
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float attackMs = 14.0f, kneeDb = 3.0f, ratio = 4.0f;
    float releaseMs = 150.0f, thresholdDb = -15.0f, makeupDb = 0.0f, mode = 0.0f;
    double sr = 48000.0;
    EnvelopeFollower env;
    float gainSmooth = 1.0f;
    float prevOutL = 0.0f, prevOutR = 0.0f;   // FB detector feed (one-sample lag)
};

// Stereo reverb, Schroeder/Freeverb style.
// p.a = mix %, p.b = size %, p.c = tone %
class Reverb final : public Module
{
public:
    Reverb() { type = ModuleType::Reverb; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float mixPct = 15.0f, sizePct = 30.0f, tonePct = 50.0f;
    double sr = 48000.0;
    // 4 combs + 2 allpass per channel, scaled by size; damping from tone
    static constexpr int kCombs = 4, kAllpass = 2;
    std::array<std::vector<float>, kCombs> combL, combR;
    std::array<int, kCombs> combPosL{}, combPosR{};
    std::array<float, kCombs> combFbL{}, combFbR{};
    std::array<float, kCombs> combDampL{}, combDampR{};
    std::array<float, kCombs> combLpL{}, combLpR{};
    std::array<std::vector<float>, kAllpass> apL, apR;
    std::array<int, kAllpass> apPosL{}, apPosR{};
    float wetGain = 0.2f, dryGain = 0.8f;
    void updateParams();
};

// Stereo delay w/ feedback HP/LP.
// p.a = dry %, p.b = wet %, p.c = HP Hz, p.d = LP Hz, p.e = feedback %,
// p.f = time s, p.g = out level dB
class StereoDelay final : public Module
{
public:
    StereoDelay() { type = ModuleType::StereoDelay; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float dryPct = 0.0f, wetPct = 20.0f, hpHz = 500.0f, lpHz = 2000.0f;
    float feedbackPct = 5.0f, timeSec = 1.29f, outLevelDb = -3.0f;
    double sr = 48000.0;
    std::vector<float> bufL, bufR;
    int bufSize = 0, pos = 0;
    float fbHpL = 0.0f, fbHpR = 0.0f, fbLpL = 0.0f, fbLpR = 0.0f;
    float xPrevL = 0.0f, xPrevR = 0.0f;
    float dryG = 0.0f, wetG = 0.2f, outG = 0.7f;
    float hpCoef = 0.0f, lpCoef = 0.0f;
};

// Tremolo: p.a = depth (0..1), p.b = rate Hz, p.c = spread deg (stereo phase)
class Tremolo final : public Module
{
public:
    Tremolo() { type = ModuleType::Tremolo; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override { phase = 0.0; }
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float depth = 0.89f, rateHz = 0.209f, spreadDeg = 140.0f;
    double sr = 48000.0, phase = 0.0;
};

// "Dimension B": p.a = preset index (2 or 3). Two fixed voicings,
// stereo chorus, no feedback, BBD-style LP in the wet path.
class ChorusBBD final : public Module
{
public:
    ChorusBBD() { type = ModuleType::ChorusBBD; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    int preset = 3;
    double sr = 48000.0, lfoPhase = 0.0;
    float rateHz = 0.8f, depthMs = 2.5f, wetMix = 0.4f;
    std::vector<float> bufL, bufR;
    int bufSize = 0, pos = 0;
    float lpL = 0.0f, lpR = 0.0f;
    void applyPreset();
};

// "VibeToo": pitch modulation via delay vibrato, 100 % wet.
// p.a = amount %, p.b = speed Hz
class Vibrato final : public Module
{
public:
    Vibrato() { type = ModuleType::Vibrato; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float amount = 57.0f, speedHz = 1.19f;
    double sr = 48000.0, phase = 0.0;
    std::vector<float> bufL, bufR;
    int bufSize = 0, pos = 0;
};

// "Small Stone" phaser: 4 allpass stages, LFO sweep, feedback resonance.
// p.a = speed Hz, p.b = resonance switch (0 = low, 1 = high)
class Phaser final : public Module
{
public:
    Phaser() { type = ModuleType::Phaser; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float speedHz = 0.596f;
    int resonance = 0;             // 0 low, 1 high
    double sr = 48000.0, lfoPhase = 0.0;
    float fb = 0.3f;
    std::array<float, 4> x1L{}, y1L{}, x1R{}, y1R{};
    float fbL = 0.0f, fbR = 0.0f;
};

// "Stereowise": M/S intensity below/above crossover.
// p.a = lowerIntensity (0..1 add on S), p.b = upperFreqHz, p.c = upperIntensity
class Widener final : public Module
{
public:
    Widener() { type = ModuleType::Widener; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float lowerInt = 0.21f, upperHz = 2517.85f, upperInt = 0.20f;
    double sr = 48000.0;
    // LR4-ish crossover: two cascaded 2nd-order Butterworth pairs via Biquads
    Biquad lo1L, lo1R, lo2L, lo2R, hi1L, hi1R, hi2L, hi2R;
};

// Exciter: HP(p.a) â†’ threshold-gated tanh harmonics â†’ mix p.b dB.
// p.a = tune Hz, p.b = harmonics amount (x drive), p.c = threshold dB, p.d = mix dB
class Exciter final : public Module
{
public:
    Exciter() { type = ModuleType::Exciter; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float tuneHz = 4000.0f, harmonics = 10.0f, thresholdDb = -4.0f, mixDb = -15.5f;
    double sr = 48000.0;
    Biquad hpL, hpR;
    EnvelopeFollower env;
    float mixG = 0.19f;
    float drive = 1.0f;
};

// "Vintage Limiter": zero-lookahead soft brickwall.
// p.a = attack ms, p.b = release ms, p.c = softKnee dB (width),
// p.d = enhance dB (presence shelf ~3.5 kHz), p.e = master level dB,
// p.f = threshold dB (internal; default -1)
class Limiter final : public Module
{
public:
    Limiter() { type = ModuleType::Limiter; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double sr) override;
    void reset() noexcept override;
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float attackMs = 0.65f, releaseMs = 762.5f, softKneeDb = 7.5f;
    float enhanceDb = 3.0f, masterLevelDb = 3.0f, thresholdDb = -1.0f;
    double sr = 48000.0;
    EnvelopeFollower env;
    float gainSmooth = 1.0f;
    Biquad shelfL, shelfR;         // Enhance presence shelf
    float outG = 1.0f;
};

// Plain gain: p.a = gain dB
class GainStage final : public Module
{
public:
    GainStage() { type = ModuleType::GainStage; }
    void configure (const ModuleParams& p) noexcept override;
    void prepare (double) override { update(); }
    void reset() noexcept override {}
    void processBlock (float* L, float* R, int n) noexcept override;
private:
    float gainDb = 0.0f, g = 1.0f;
    void update() { g = std::pow (10.0f, gainDb / 20.0f); }
};

} // namespace JdxxChains
