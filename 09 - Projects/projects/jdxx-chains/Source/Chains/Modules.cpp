/*
    JDXX Chains — module implementations (clean-room).

    JDXX_CHAIN_2026_11_02: every coefficient in the chain specs comes from the
    live BandLab preset tables extracted 2026-11-02. Where BandLab shows an
    ambiguous scale, the assumption used is documented inline AND flagged in
    docs/JDXX_CHAIN_SPEC.md (single source of truth for sign-off).

    AGPL-3.0. See LICENSE.
*/
#include "Modules.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace JdxxChains
{

static constexpr float DB_TO_LIN (float db) noexcept { return std::pow (10.0f, db / 20.0f); }
static constexpr float LIN_TO_DB (float lin) noexcept { return 20.0f * std::log10 (std::max (lin, 1e-9f)); }
static constexpr float clampf (float v, float lo, float hi) noexcept { return std::max (lo, std::min (hi, v)); }

// ===========================================================================
// Noise Gate
// ===========================================================================
void NoiseGate::prepare (double sampleRate)
{
    sr = sampleRate;
    env.setSampleRate ((float) sr);
    env.setAttackRelease (attackMs, releaseMs);
    // Gain ballistics: opening uses attack, closing uses release.
    const float atkSec = std::max (0.001f, attackMs * 0.001f);
    const float relSec = std::max (0.001f, releaseMs * 0.001f);
    gainAttackCoef  = std::exp (-1.0f / (atkSec * (float) sr));
    gainReleaseCoef = std::exp (-1.0f / (relSec * (float) sr));
    gain = 1.0f;
    open = true;
}

void NoiseGate::reset() noexcept
{
    env.reset();
    gain = 1.0f;
    open = true;
}

void NoiseGate::processBlock (float* L, float* R, int n) noexcept
{
    const float thrLin = DB_TO_LIN (thresholdDb);
    const float floorLin = DB_TO_LIN (floorDb);
    // Hysteresis: open strictly above threshold, close 6 dB below it.
    const float closeThr = DB_TO_LIN (thresholdDb - 6.0f);

    for (int i = 0; i < n; ++i)
    {
        const float det = std::max (std::fabs (L[i]), std::fabs (R[i]));
        env.process (det);
        const float e = env.getLevel();

        if (open && e < closeThr) open = false;
        else if (!open && e >= thrLin) open = true;

        const float target = open ? 1.0f : floorLin;
        const float coef = open ? gainAttackCoef : gainReleaseCoef;
        gain = coef * gain + (1.0f - coef) * target;

        L[i] *= gain;
        R[i] *= gain;
    }
}

// ===========================================================================
// HighPass (RBJ cookbook, 2nd order Butterworth when Q = 0.7071)
// ===========================================================================
void HighPass::prepare (double sampleRate)
{
    sr = sampleRate;
    hpL.setParams (Biquad::Type::HighPass, freqHz, q, 0.0f, (float) sr);
    hpR.setParams (Biquad::Type::HighPass, freqHz, q, 0.0f, (float) sr);
}

void HighPass::processBlock (float* L, float* R, int n) noexcept
{
    for (int i = 0; i < n; ++i) { L[i] = hpL.processSample (L[i]); R[i] = hpR.processSample (R[i]); }
}

// ===========================================================================
// Tape Simulator
// Assumptions (flagged, verify by ear):
//   - drive 1.764 → ~+4.9 dB into tanh,
//   - bias  50    → centered (offset 0); usable 0..100 maps to ±0.2 DC-ish offset,
//   - speed 24.94 → wow LFO rate ≈ 0.55 Hz (24.94 → 1x),
//   - wow   5     → delay-time modulation depth ≈ subtle ±0.25 % pitch wobble,
//   - HF rounding fixed one-pole lowpass at 16 kHz.
// ===========================================================================
void TapeSim::prepare (double sampleRate)
{
    sr = sampleRate;
    delaySize = (int) std::ceil ((maxDelayMs * 0.001) * sr) + 2;
    delayL.assign ((size_t) delaySize, 0.0f);
    delayR.assign ((size_t) delaySize, 0.0f);
    delayPos = 0;
    phase = 0.0;
    // One-pole HF-rounding coefficient
    lp1L = 0.0f; lp1R = 0.0f;
}

void TapeSim::reset() noexcept
{
    std::fill (delayL.begin(), delayL.end(), 0.0f);
    std::fill (delayR.begin(), delayR.end(), 0.0f);
    delayPos = 0; phase = 0.0; lp1L = lp1R = 0.0f;
}

void TapeSim::processBlock (float* L, float* R, int n) noexcept
{
    const float biasOff = (bias - 50.0f) * 0.004f;         // 0 for bias=50
    const float tanhBias = std::tanh (drive * biasOff);
    const float wobbleFrac = wowDepth * 0.0005f;           // 5 -> 0.0025 (subtle)
    const float wowHz = std::max (0.05f, speed / 24.94f * 0.55f);
    const float baseDelayMs = 1.25f;
    const float baseSamples = baseDelayMs * 0.001f * (float) sr;
    const float wowRangeSamp = baseSamples * wobbleFrac;   // ± samples of swing
    const float hfCoef = std::exp (-2.0f * 3.14159265f * hfCutHz / (float) sr);
    const float outG = DB_TO_LIN (outLevelDb);

    for (int i = 0; i < n; ++i)
    {
        const float lfo = std::sin ((float) phase);
        phase += 2.0 * 3.14159265 * wowHz / sr;
        if (phase > 2.0 * 3.14159265) phase -= 2.0 * 3.14159265;
        const float delaySamples = baseSamples + wowRangeSamp * lfo;

        // Saturate first (drive into tanh), then HF round.
        float xL = L[i], xR = R[i];
        float sL = std::tanh (drive * (xL + biasOff)) - tanhBias;
        float sR = std::tanh (drive * (xR + biasOff)) - tanhBias;

        lp1L = (1.0f - hfCoef) * sL + hfCoef * lp1L; sL = lp1L;
        lp1R = (1.0f - hfCoef) * sR + hfCoef * lp1R; sR = lp1R;

        // Wow: read from delay line at modulated distance.
        delayL[(size_t) delayPos] = sL;
        delayR[(size_t) delayPos] = sR;
        int rd = delayPos - (int) std::lround (delaySamples);
        while (rd < 0) rd += delaySize;
        const float yL = delayL[(size_t) rd];
        const float yR = delayR[(size_t) rd];
        delayPos = (delayPos + 1) % delaySize;

        L[i] = yL * outG;
        R[i] = yR * outG;
    }
}

// ===========================================================================
// Graphic EQ 7 bands (fixed RBJ bells, Q = 1.0, + output level in dB)
// ===========================================================================
void GraphicEQ7::prepare (double sampleRate)
{
    sr = sampleRate;
    updateCoeffs();
    levelGain = DB_TO_LIN (levelDb);
}

void GraphicEQ7::reset() noexcept
{
    for (auto& b : bandsL) b.reset();
    for (auto& b : bandsR) b.reset();
}

void GraphicEQ7::updateCoeffs()
{
    for (int i = 0; i < 7; ++i)
    {
        bandsL[(size_t) i].setParams (Biquad::Type::Bell, freqs[(size_t) i], 1.0f, gains[(size_t) i], (float) sr);
        bandsR[(size_t) i].setParams (Biquad::Type::Bell, freqs[(size_t) i], 1.0f, gains[(size_t) i], (float) sr);
    }
    levelGain = DB_TO_LIN (levelDb);
}

void GraphicEQ7::processBlock (float* L, float* R, int n) noexcept
{
    for (int i = 0; i < n; ++i)
    {
        float xL = L[i], xR = R[i];
        for (size_t b = 0; b < bandsL.size(); ++b)
        {
            xL = bandsL[b].processSample (xL);
            xR = bandsR[b].processSample (xR);
        }
        L[i] = xL * levelGain;
        R[i] = xR * levelGain;
    }
}

// ===========================================================================
// DeEsser (whole-band GR; bandpass sidechain at `freqHz`)
// ===========================================================================
void DeEsser::prepare (double sampleRate)
{
    sr = sampleRate;
    bpL.setParams (Biquad::Type::Bandpass, freqHz, 2.0f, 0.0f, (float) sr);
    bpR.setParams (Biquad::Type::Bandpass, freqHz, 2.0f, 0.0f, (float) sr);
    env.setSampleRate ((float) sr);
    env.setAttackRelease (1.5f, 120.0f);
    atkCoef = std::exp (-1.0f / (0.0015f * (float) sr));
    relCoef = std::exp (-1.0f / (0.120f  * (float) sr));
    grDb = 0.0f;
    applyGrL = applyGrR = 1.0f;
    smooth = 0.0f;
}

void DeEsser::reset() noexcept
{
    bpL.reset(); bpR.reset(); env.reset(); grDb = 0.0f; applyGrL = applyGrR = 1.0f;
}

void DeEsser::processBlock (float* L, float* R, int n) noexcept
{
    const float ratio = 4.0f;
    for (int i = 0; i < n; ++i)
    {
        const float s = 0.5f * (bpL.processSample (L[i]) + bpR.processSample (R[i]));
        env.process (std::fabs (s));
        const float envDb = LIN_TO_DB (env.getLevel() + 1e-9f);
        const float over = envDb - thresholdDb;
        const float targetGr = over > 0.0f ? over * (ratio - 1.0f) / ratio : 0.0f;
        const float capped = clampf (targetGr, 0.0f, 15.0f);
        // Smooth GR in dB domain: fast attack, slower release.
        const float coef = capped > grDb ? atkCoef : relCoef;
        grDb = coef * grDb + (1.0f - coef) * capped;
        const float g = DB_TO_LIN (-grDb);
        L[i] *= g;
        R[i] *= g;
    }
}

// ===========================================================================
// Compressor (shared FF/FB). Soft knee, stereo-linked detector.
// ===========================================================================
void Compressor::prepare (double sampleRate)
{
    sr = sampleRate;
    env.setSampleRate ((float) sr);
    env.setAttackRelease (attackMs, releaseMs);
    gainSmooth = 1.0f;
}

void Compressor::reset() noexcept { env.reset(); gainSmooth = 1.0f; prevOutL = prevOutR = 0.0f; }

void Compressor::processBlock (float* L, float* R, int n) noexcept
{
    const float knee = std::max (0.01f, kneeDb);
    const float ratioMinus1 = (ratio - 1.0f) / std::max (ratio, 1.0f);
    const float makeup = DB_TO_LIN (makeupDb);
    const bool feedback = mode >= 0.5f;   // 1 = feedback (FBK)

    for (int i = 0; i < n; ++i)
    {
        // Detector: FF reads the input; FB reads the previous OUTPUT sample.
        const float detector = feedback
            ? 0.5f * (std::fabs (prevOutL) + std::fabs (prevOutR))
            : std::max (std::fabs (L[i]), std::fabs (R[i]));

        env.process (detector);
        const float envDb = LIN_TO_DB (env.getLevel() + 1e-9f);
        const float x = envDb - thresholdDb;

        float gr = 0.0f;
        if (x <= -knee * 0.5f)      gr = 0.0f;
        else if (x >=  knee * 0.5f) gr = x * ratioMinus1;
        else
        {
            const float t = x + knee * 0.5f;
            gr = (t * t) / (2.0f * knee) * ratioMinus1;
        }

        const float targetGain = DB_TO_LIN (-gr);
        // One-pole toward target, rate selected by direction.
        const float atkSec = std::max (0.0005f, attackMs * 0.001f);
        const float relSec = std::max (0.001f, releaseMs * 0.001f);
        const float coef = targetGain < gainSmooth
            ? std::exp (-1.0f / (atkSec * (float) sr))
            : std::exp (-1.0f / (relSec * (float) sr));
        gainSmooth = coef * gainSmooth + (1.0f - coef) * targetGain;

        prevOutL = L[i] * gainSmooth * makeup;
        prevOutR = R[i] * gainSmooth * makeup;
        L[i] = prevOutL;
        R[i] = prevOutR;
    }
}

// ===========================================================================
// Reverb (Freeverb-style: 4 combs + 2 allpass per channel, stereo spread 23)
// ===========================================================================
void Reverb::prepare (double sampleRate)
{
    sr = sampleRate;
    const int spread = 23;
    const float sizeScale = 0.5f + (sizePct / 100.0f) * 1.5f;   // 0.5..2.0x
    static const int combBase[kCombs] = { 1116, 1188, 1277, 1356 };
    static const int apBase[kAllpass] = { 556, 441 };

    for (size_t i = 0; i < (size_t) kCombs; ++i)
    {
        combL[i].assign ((size_t) std::ceil (combBase[i] * sizeScale) + 16, 0.0f);
        combR[i].assign ((size_t) std::ceil ((combBase[i] + spread) * sizeScale) + 16, 0.0f);
        combPosL[i] = combPosR[i] = 0;
        combLpL[i] = combLpR[i] = 0.0f;
    }
    for (size_t i = 0; i < (size_t) kAllpass; ++i)
    {
        apL[i].assign ((size_t) std::ceil (apBase[i] * sizeScale) + 8, 0.0f);
        apR[i].assign ((size_t) std::ceil ((apBase[i] + spread) * sizeScale) + 8, 0.0f);
        apPosL[i] = apPosR[i] = 0;
    }
    updateParams();
}

void Reverb::reset() noexcept
{
    for (size_t i = 0; i < (size_t) kCombs; ++i) { std::fill (combL[i].begin(), combL[i].end(), 0.0f); std::fill (combR[i].begin(), combR[i].end(), 0.0f); combPosL[i] = combPosR[i] = 0; combLpL[i] = combLpR[i] = 0.0f; }
    for (size_t i = 0; i < (size_t) kAllpass; ++i) { std::fill (apL[i].begin(), apL[i].end(), 0.0f); std::fill (apR[i].begin(), apR[i].end(), 0.0f); apPosL[i] = apPosR[i] = 0; }
}

void Reverb::updateParams()
{
    const float fb   = 0.72f + (sizePct / 100.0f) * 0.22f;    // 0.72..0.94
    const float damp = 1.0f - (tonePct / 100.0f) * 0.7f;      // 1.0 dark .. 0.3 bright

    for (size_t i = 0; i < (size_t) kCombs; ++i)
    {
        combFbL[i] = combFbR[i] = fb;
        combDampL[i] = combDampR[i] = damp;
    }
    wetGain = mixPct / 100.0f;
    dryGain = 1.0f - wetGain * 0.5f;
}

void Reverb::processBlock (float* L, float* R, int n) noexcept
{
    for (int i = 0; i < n; ++i)
    {
        const float in = 0.5f * (L[i] + R[i]);
        float wetL = 0.0f, wetR = 0.0f;

        for (size_t c = 0; c < (size_t) kCombs; ++c)
        {
            // L comb
            {
                float out = combL[c][(size_t) combPosL[c]];
                combLpL[c] = out * (1.0f - combDampL[c]) + combLpL[c] * combDampL[c];
                combL[c][(size_t) combPosL[c]] = in + combLpL[c] * combFbL[c];
                combPosL[c] = (combPosL[c] + 1) % (int) combL[c].size();
                wetL += out;
            }
            // R comb (offset length)
            {
                float out = combR[c][(size_t) combPosR[c]];
                combLpR[c] = out * (1.0f - combDampR[c]) + combLpR[c] * combDampR[c];
                combR[c][(size_t) combPosR[c]] = in + combLpR[c] * combFbR[c];
                combPosR[c] = (combPosR[c] + 1) % (int) combR[c].size();
                wetR += out;
            }
        }

        // Allpass, Schroeder form: y[n] = -g*x[n] + x[n-N] + g*y[n-N]
        auto apProcess = [](std::vector<float>& buf, int& pos, float x, float g) {
            const float bufout = buf[(size_t) pos];
            buf[(size_t) pos] = x + g * bufout;
            pos = (pos + 1) % (int) buf.size();
            return bufout - g * x;
        };
        wetL = apProcess (apL[0], apPosL[0], wetL, 0.5f);
        wetR = apProcess (apR[0], apPosR[0], wetR, 0.5f);
        wetL = apProcess (apL[1], apPosL[1], wetL, 0.5f);
        wetR = apProcess (apR[1], apPosR[1], wetR, 0.5f);

        // Freeverb-style output scale
        wetL *= 0.25f; wetR *= 0.25f;

        L[i] = L[i] * dryGain + wetL * wetGain;
        R[i] = R[i] * dryGain + wetR * wetGain;
    }
}

// ===========================================================================
// Stereo delay with feedback HP/LP
// ===========================================================================
void StereoDelay::prepare (double sampleRate)
{
    sr = sampleRate;
    bufSize = (int) std::ceil (4.0 * sr);
    bufL.assign ((size_t) bufSize, 0.0f);
    bufR.assign ((size_t) bufSize, 0.0f);
    pos = 0;
    hpCoef = std::exp (-2.0f * 3.14159265f * hpHz / (float) sr);
    lpCoef = std::exp (-2.0f * 3.14159265f * lpHz / (float) sr);
    dryG = clampf (dryPct / 100.0f, 0.0f, 2.0f);
    wetG = clampf (wetPct / 100.0f, 0.0f, 2.0f);
    outG = DB_TO_LIN (outLevelDb);
    fbHpL = fbHpR = fbLpL = fbLpR = 0.0f;
}

void StereoDelay::reset() noexcept
{
    std::fill (bufL.begin(), bufL.end(), 0.0f);
    std::fill (bufR.begin(), bufR.end(), 0.0f);
    pos = 0; fbHpL = fbHpR = fbLpL = fbLpR = xPrevL = xPrevR = 0.0f;
}

void StereoDelay::processBlock (float* L, float* R, int n) noexcept
{
    const int delaySamples = (int) std::lround (timeSec * (float) sr);
    const float fb = clampf (feedbackPct / 100.0f, 0.0f, 0.9f);

    for (int i = 0; i < n; ++i)
    {
        const int rd = (pos - (delaySamples % bufSize) + bufSize) % bufSize;
        const float dL = bufL[(size_t) rd];
        const float dR = bufR[(size_t) rd];

        // Feedback path: one-pole HP then LP. State kept in members.
        float fL = dL, fR = dR;
        fbHpL = hpCoef * (fbHpL + fL - xPrevL); xPrevL = fL; fL = fbHpL;
        fbHpR = hpCoef * (fbHpR + fR - xPrevR); xPrevR = fR; fR = fbHpR;
        // Lowpass one-pole
        fbLpL = (1.0f - lpCoef) * fL + lpCoef * fbLpL; fL = fbLpL;
        fbLpR = (1.0f - lpCoef) * fR + lpCoef * fbLpR; fR = fbLpR;

        bufL[(size_t) pos] = L[i] + fL * fb;
        bufR[(size_t) pos] = R[i] + fR * fb;

        L[i] = (L[i] * dryG + dL * wetG) * outG;
        R[i] = (R[i] * dryG + dR * wetG) * outG;
        pos = (pos + 1) % bufSize;
    }
}

// ===========================================================================
// Tremolo
// ===========================================================================
void Tremolo::prepare (double sampleRate) { sr = sampleRate; phase = 0.0; }

void Tremolo::processBlock (float* L, float* R, int n) noexcept
{
    const double inc = 2.0 * 3.14159265 * rateHz / sr;
    const double spreadRad = spreadDeg * 3.14159265 / 180.0;
    for (int i = 0; i < n; ++i)
    {
        const float modL = 0.5f * (1.0f - std::sin ((float) phase));             // 0..1
        const float modR = 0.5f * (1.0f - std::sin ((float) (phase + spreadRad)));
        const float gL = 1.0f - depth * modL;
        const float gR = 1.0f - depth * modR;
        L[i] *= gL; R[i] *= gR;
        phase += inc;
        if (phase > 2.0 * 3.14159265) phase -= 2.0 * 3.14159265;
    }
}

// ===========================================================================
// "Dimension B" BBD chorus — presets 02/03 (assumptions documented)
//   02: 0.55 Hz, depth 1.6 ms, wet 0.35
//   03: 0.80 Hz, depth 2.5 ms, wet 0.40
// ===========================================================================
void ChorusBBD::applyPreset()
{
    if (preset == 2) { rateHz = 0.55f; depthMs = 1.6f; wetMix = 0.35f; }
    else              { rateHz = 0.80f; depthMs = 2.5f; wetMix = 0.40f; }
}

void ChorusBBD::prepare (double sampleRate)
{
    sr = sampleRate;
    applyPreset();
    bufSize = (int) std::ceil (0.02 * sr) + 4;   // 20 ms line is plenty
    bufL.assign ((size_t) bufSize, 0.0f);
    bufR.assign ((size_t) bufSize, 0.0f);
    pos = 0; lfoPhase = 0.0; lpL = lpR = 0.0f;
}

void ChorusBBD::reset() noexcept
{
    std::fill (bufL.begin(), bufL.end(), 0.0f);
    std::fill (bufR.begin(), bufR.end(), 0.0f);
    pos = 0; lfoPhase = 0.0; lpL = lpR = 0.0f;
}

void ChorusBBD::processBlock (float* L, float* R, int n) noexcept
{
    const double inc = 2.0 * 3.14159265 * rateHz / sr;
    const float baseMs = 2.6f;
    const float baseSamp = baseMs * 0.001f * (float) sr;
    const float depthSamp = depthMs * 0.001f * (float) sr;
    const float lpCoef = std::exp (-2.0f * 3.14159265f * 8000.0f / (float) sr);

    for (int i = 0; i < n; ++i)
    {
        const float lfoL = std::sin ((float) lfoPhase);
        const float lfoR = std::sin ((float) (lfoPhase + 3.14159265 / 2.0)); // 90° offset
        lfoPhase += inc;

        bufL[(size_t) pos] = L[i];
        bufR[(size_t) pos] = R[i];

        const float delL = baseSamp + depthSamp * lfoL;
        const float delR = baseSamp + depthSamp * lfoR;

        auto readAt = [&](const std::vector<float>& buf, float delaySamples) {
            const float idx = (float) pos - delaySamples;
            int i0 = (int) std::floor (idx);
            const float frac = idx - (float) i0;
            while (i0 < 0) i0 += bufSize;
            const int i1 = (i0 + 1) % bufSize;
            return buf[(size_t) i0] * (1.0f - frac) + buf[(size_t) i1] * frac;
        };

        float wetL = readAt (bufL, delL);
        float wetR = readAt (bufR, delR);
        // BBD-ish gentle LP on the wet path
        lpL = (1.0f - lpCoef) * wetL + lpCoef * lpL; wetL = lpL;
        lpR = (1.0f - lpCoef) * wetR + lpCoef * lpR; wetR = lpR;

        L[i] = L[i] * (1.0f - wetMix) + wetL * wetMix;
        R[i] = R[i] * (1.0f - wetMix) + wetR * wetMix;
        pos = (pos + 1) % bufSize;
    }
}

// ===========================================================================
// "VibeToo" vibrato — 100% wet delay-time modulation
// ===========================================================================
void Vibrato::prepare (double sampleRate)
{
    sr = sampleRate;
    bufSize = (int) std::ceil (0.02 * sr) + 4;
    bufL.assign ((size_t) bufSize, 0.0f);
    bufR.assign ((size_t) bufSize, 0.0f);
    pos = 0; phase = 0.0;
}

void Vibrato::reset() noexcept
{
    std::fill (bufL.begin(), bufL.end(), 0.0f);
    std::fill (bufR.begin(), bufR.end(), 0.0f);
    pos = 0; phase = 0.0;
}

void Vibrato::processBlock (float* L, float* R, int n) noexcept
{
    const double inc = 2.0 * 3.14159265 * speedHz / sr;
    const float baseMs = 2.2f;
    const float depthMs = 1.0f + (amount / 100.0f) * 2.5f;
    const float baseSamp = baseMs * 0.001f * (float) sr;
    const float depthSamp = depthMs * 0.001f * (float) sr;

    for (int i = 0; i < n; ++i)
    {
        const float lfo = std::sin ((float) phase);
        phase += inc;

        bufL[(size_t) pos] = L[i];
        bufR[(size_t) pos] = R[i];

        const float delaySamples = baseSamp + depthSamp * lfo;
        auto readAt = [&](const std::vector<float>& buf) {
            const float idx = (float) pos - delaySamples;
            int i0 = (int) std::floor (idx);
            const float frac = idx - (float) i0;
            while (i0 < 0) i0 += bufSize;
            const int i1 = (i0 + 1) % bufSize;
            return buf[(size_t) i0] * (1.0f - frac) + buf[(size_t) i1] * frac;
        };
        L[i] = readAt (bufL);
        R[i] = readAt (bufR);
        pos = (pos + 1) % bufSize;
    }
}

// ===========================================================================
// "Small Stone" phaser — 4 × 1st-order allpass, LFO sweep, feedback switch
//   resonance 0 (low) -> 0.35 feedback; 1 (high) -> 0.60
// ===========================================================================
void Phaser::prepare (double sampleRate)
{
    sr = sampleRate;
    lfoPhase = 0.0;
    fb = (resonance == 0) ? 0.35f : 0.60f;
    x1L.fill (0.0f); y1L.fill (0.0f); x1R.fill (0.0f); y1R.fill (0.0f);
    fbL = fbR = 0.0f;
}

void Phaser::reset() noexcept
{
    lfoPhase = 0.0;
    x1L.fill (0.0f); y1L.fill (0.0f); x1R.fill (0.0f); y1R.fill (0.0f);
    fbL = fbR = 0.0f;
}

void Phaser::processBlock (float* L, float* R, int n) noexcept
{
    const double inc = 2.0 * 3.14159265 * speedHz / sr;
    // Sweep centre frequency 220 .. 1800 Hz
    const float fcMin = 220.0f, fcMax = 1800.0f;

    for (int i = 0; i < n; ++i)
    {
        const float lfo = 0.5f * (1.0f + std::sin ((float) lfoPhase));   // 0..1
        lfoPhase += inc;
        const float fc = fcMin + (fcMax - fcMin) * lfo;
        const float w = std::tan (3.14159265f * fc / (float) sr);
        const float apCoef = (w - 1.0f) / (w + 1.0f);                    // 1st-order allpass g

        // 1st-order allpass (DAFX):  y = -g*x + x1 + g*y1
        auto ap1 = [&](float x, float& x1, float& y1) {
            const float y = -apCoef * x + x1 + apCoef * y1;
            x1 = x; y1 = y;
            return y;
        };
        float aL = L[i] + fbL, aR = R[i] + fbR;
        for (int s = 0; s < 4; ++s) aL = ap1 (aL, x1L[(size_t) s], y1L[(size_t) s]);
        for (int s = 0; s < 4; ++s) aR = ap1 (aR, x1R[(size_t) s], y1R[(size_t) s]);
        fbL = aL * fb; fbR = aR * fb;
        L[i] = 0.5f * (L[i] + aL);
        R[i] = 0.5f * (R[i] + aR);
    }
}

// ===========================================================================
// "Stereowise" widener — M/S intensities below/above crossover
// ===========================================================================
void Widener::prepare (double sampleRate)
{
    sr = sampleRate;

    // Butterworth 2nd order pairs at Q=0.7071 (two cascaded => ~LR4 slope)
    const float q = 0.7071f;
    lo1L.setParams (Biquad::Type::LowPass, upperHz, q, 0.0f, (float) sr);
    lo1R.setParams (Biquad::Type::LowPass, upperHz, q, 0.0f, (float) sr);
    lo2L.setParams (Biquad::Type::LowPass, upperHz, q, 0.0f, (float) sr);
    lo2R.setParams (Biquad::Type::LowPass, upperHz, q, 0.0f, (float) sr);
    hi1L.setParams (Biquad::Type::HighPass, upperHz, q, 0.0f, (float) sr);
    hi1R.setParams (Biquad::Type::HighPass, upperHz, q, 0.0f, (float) sr);
    hi2L.setParams (Biquad::Type::HighPass, upperHz, q, 0.0f, (float) sr);
    hi2R.setParams (Biquad::Type::HighPass, upperHz, q, 0.0f, (float) sr);
}

void Widener::reset() noexcept
{
    lo1L.reset(); lo1R.reset(); lo2L.reset(); lo2R.reset();
    hi1L.reset(); hi1R.reset(); hi2L.reset(); hi2R.reset();
}

void Widener::processBlock (float* L, float* R, int n) noexcept
{
    const float lowG  = 1.0f + lowerInt;
    const float highG = 1.0f + upperInt;
    for (int i = 0; i < n; ++i)
    {
        const float m = 0.5f * (L[i] + R[i]);
        float s = 0.5f * (L[i] - R[i]);

        // Parallel Butterworth crossover on the side band (~LR4 slope):
        // lo cascade + hi cascade read the same s; bands scaled, then re-summed.
        // (Butterworth Q = 1/sqrt(2) makes the pair near power-complementary.)
        const float sLo = lo2L.processSample (lo1L.processSample (s));
        const float sHi = hi2L.processSample (hi1L.processSample (s));
        s = sLo * lowG + sHi * highG;

        L[i] = m + s;
        R[i] = m - s;
    }
}

// ===========================================================================
// configure() overrides — stash spec values; coefficients are recomputed in
// prepare(), so configuration order never corrupts derived state.
// ===========================================================================
void NoiseGate::configure (const ModuleParams& p) noexcept { thresholdDb = p.a; attackMs = p.b; releaseMs = p.c; }
void HighPass::configure  (const ModuleParams& p) noexcept { freqHz = p.a; q = p.b; }
void TapeSim::configure   (const ModuleParams& p) noexcept { drive = p.a; bias = p.b; outLevelDb = p.c; speed = p.d; wowDepth = p.e; hfCutHz = p.f > 0.0f ? p.f : 16000.0f; }
void GraphicEQ7::configure (const ModuleParams& p) noexcept
{
    gains = { p.a, p.b, p.c, p.d, p.e, p.f, p.g };
    levelDb = p.h;
}
void DeEsser::configure   (const ModuleParams& p) noexcept { freqHz = p.a; thresholdDb = p.b; }
void Compressor::configure (const ModuleParams& p) noexcept
{
    attackMs = p.a; kneeDb = p.b; ratio = p.c; releaseMs = p.d;
    thresholdDb = p.e; makeupDb = p.f; mode = p.g;
}
void Reverb::configure    (const ModuleParams& p) noexcept { mixPct = p.a; sizePct = p.b; tonePct = p.c; }
void StereoDelay::configure (const ModuleParams& p) noexcept
{
    dryPct = p.a; wetPct = p.b; hpHz = p.c; lpHz = p.d;
    feedbackPct = p.e; timeSec = p.f; outLevelDb = p.g;
}
void Tremolo::configure   (const ModuleParams& p) noexcept { depth = p.a; rateHz = p.b; spreadDeg = p.c; }
void ChorusBBD::configure (const ModuleParams& p) noexcept { preset = (int) p.a; applyPreset(); }
void Vibrato::configure   (const ModuleParams& p) noexcept { amount = p.a; speedHz = p.b; }
void Phaser::configure    (const ModuleParams& p) noexcept { speedHz = p.a; resonance = (int) p.b; }
void Widener::configure   (const ModuleParams& p) noexcept { lowerInt = p.a; upperHz = p.b; upperInt = p.c; }

// ===========================================================================
// Exciter: HP(tune) -> threshold-gated tanh harmonics -> blend at mixDb
// Assumption (flagged): `harmonics` scales the tanh drive (10 -> x3.5),
// threshold gate transition is 12 dB. Verify by ear and document.
// ===========================================================================
void Exciter::prepare (double sampleRate)
{
    sr = sampleRate;
    hpL.setParams (Biquad::Type::HighPass, tuneHz, 0.7071f, 0.0f, (float) sr);
    hpR.setParams (Biquad::Type::HighPass, tuneHz, 0.7071f, 0.0f, (float) sr);
    env.setSampleRate ((float) sr);
    env.setAttackRelease (5.0f, 100.0f);
    mixG  = DB_TO_LIN (mixDb);
    drive = 1.0f + harmonics * 0.25f;
}

void Exciter::reset() noexcept { hpL.reset(); hpR.reset(); env.reset(); }

void Exciter::processBlock (float* L, float* R, int n) noexcept
{
    for (int i = 0; i < n; ++i)
    {
        float sL = hpL.processSample (L[i]);
        float sR = hpR.processSample (R[i]);
        env.process (std::max (std::fabs (sL), std::fabs (sR)));
        const float envDb = LIN_TO_DB (env.getLevel() + 1e-9f);
        // 12 dB smoothstep gate above threshold
        const float t = clampf ((envDb - thresholdDb + 12.0f) / 12.0f, 0.0f, 1.0f);
        const float gate = t * t * (3.0f - 2.0f * t);

        const float hL = std::tanh (sL * drive) * gate;
        const float hR = std::tanh (sR * drive) * gate;
        L[i] += hL * mixG;
        R[i] += hR * mixG;
    }
}

void Exciter::configure (const ModuleParams& p) noexcept
{
    tuneHz = p.a; harmonics = p.b; thresholdDb = p.c; mixDb = p.d;
}

// ===========================================================================
// Vintage Limiter: internal presence shelf -> peak limiter, soft knee,
// fixed threshold at -1 dBFS (documented; no LUFS auto-gain anywhere).
//   Enhance = shelf gain at 3.5 kHz, Soft Knee = knee width,
//   Master Level = output gain added after limiting.
// ===========================================================================
void Limiter::prepare (double sampleRate)
{
    sr = sampleRate;
    shelfL.setParams (Biquad::Type::HighShelf, 3500.0f, 0.75f, enhanceDb, (float) sr);
    shelfR.setParams (Biquad::Type::HighShelf, 3500.0f, 0.75f, enhanceDb, (float) sr);
    env.setSampleRate ((float) sr);
    env.setAttackRelease (std::max (0.05f, attackMs), releaseMs);
    outG = DB_TO_LIN (masterLevelDb);
    gainSmooth = 1.0f;
}

void Limiter::reset() noexcept { shelfL.reset(); shelfR.reset(); env.reset(); gainSmooth = 1.0f; }

void Limiter::processBlock (float* L, float* R, int n) noexcept
{
    const float knee = std::max (0.01f, softKneeDb);
    for (int i = 0; i < n; ++i)
    {
        float xL = shelfL.processSample (L[i]);
        float xR = shelfR.processSample (R[i]);

        env.process (std::max (std::fabs (xL), std::fabs (xR)));
        const float envDb = LIN_TO_DB (env.getLevel() + 1e-9f);
        const float x = envDb - thresholdDb;

        float gr = 0.0f;
        if (x <= -knee * 0.5f)      gr = 0.0f;
        else if (x >=  knee * 0.5f) gr = x;
        else
        {
            const float t = x + knee * 0.5f;
            gr = (t * t) / (2.0f * knee);
        }
        // Attack smooths the gain reduction so fast transients don't click.
        const float atkSec = std::max (0.0002f, attackMs * 0.001f);
        const float coef = std::exp (-1.0f / (atkSec * (float) sr));
        gainSmooth = coef * gainSmooth + (1.0f - coef) * DB_TO_LIN (-gr);

        xL *= gainSmooth; xR *= gainSmooth;
        xL = clampf (xL, -1.0f, 1.0f);
        xR = clampf (xR, -1.0f, 1.0f);
        L[i] = xL * outG;
        R[i] = xR * outG;
    }
}

void Limiter::configure (const ModuleParams& p) noexcept
{
    attackMs = p.a; releaseMs = p.b; softKneeDb = p.c;
    enhanceDb = p.d; masterLevelDb = p.e; thresholdDb = p.f;
}

// ===========================================================================
// GainStage
// ===========================================================================
void GainStage::processBlock (float* L, float* R, int n) noexcept
{
    for (int i = 0; i < n; ++i) { L[i] *= g; R[i] *= g; }
}

void GainStage::configure (const ModuleParams& p) noexcept { gainDb = p.a; g = DB_TO_LIN (gainDb); }

// ===========================================================================
// Factory
// ===========================================================================
Module* createModule (const ModuleSpec& spec, double sampleRate) noexcept
{
    Module* m = nullptr;
    switch (spec.type)
    {
        case ModuleType::Gate:         m = new NoiseGate();   break;
        case ModuleType::HighPass:     m = new HighPass();    break;
        case ModuleType::TapeSim:      m = new TapeSim();     break;
        case ModuleType::GraphicEQ7:   m = new GraphicEQ7();  break;
        case ModuleType::DeEsser:      m = new DeEsser();     break;
        case ModuleType::CompressorFF:
        case ModuleType::CompressorFB: m = new Compressor();  break;
        case ModuleType::Reverb:       m = new Reverb();      break;
        case ModuleType::StereoDelay:  m = new StereoDelay(); break;
        case ModuleType::Tremolo:      m = new Tremolo();     break;
        case ModuleType::ChorusBBD:    m = new ChorusBBD();   break;
        case ModuleType::Vibrato:      m = new Vibrato();     break;
        case ModuleType::Phaser:       m = new Phaser();      break;
        case ModuleType::Widener:      m = new Widener();     break;
        case ModuleType::Exciter:      m = new Exciter();     break;
        case ModuleType::Limiter:      m = new Limiter();     break;
        case ModuleType::GainStage:    m = new GainStage();   break;
    }
    if (m == nullptr) return nullptr;
    m->configure (spec.p);
    m->type = spec.type;
    m->name = spec.name;
    m->prepare (sampleRate);
    return m;
}

void destroyModule (Module* m) noexcept { delete m; }
