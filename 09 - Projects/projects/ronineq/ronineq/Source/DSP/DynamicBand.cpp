/*
    RoninEQ — DynamicBand DSP core
    DynamicBand.cpp : one parametric band with optional downward dynamics.

    Pure C++17, no JUCE dependency. Clean-room implementation.
*/
#include "DynamicBand.h"
#include <cmath>

namespace RoninEQ
{

void DynamicBand::setSampleRate (float sampleRate)
{
    sr = sampleRate > 0.0f ? sampleRate : 48000.0f;
    envL.setSampleRate (sr);
    envR.setSampleRate (sr);
    // ~2 ms one-pole smoothing on the applied gain: fast enough to track
    // the envelope follower, slow enough to kill zipper noise on drags.
    gainSmoothCoef = 1.0f - std::exp (-1.0f / (0.002f * sr));
    configureDetector();
    structDirty = true;
}

void DynamicBand::setParams (const BandParams& p)
{
    const bool typeChanged = (p.type != params.type);
    const bool freqChanged = (p.freq != params.freq);
    const bool qChanged    = (p.q    != params.q);

    params = p;

    if (typeChanged || freqChanged || qChanged)
    {
        configureDetector();
        structDirty = true;
    }
    if (p.attackMs != envAttackMs || p.releaseMs != envReleaseMs)
    {
        envL.setAttackRelease (p.attackMs, p.releaseMs);
        envR.setAttackRelease (p.attackMs, p.releaseMs);
        envAttackMs   = p.attackMs;
        envReleaseMs  = p.releaseMs;
    }
}

void DynamicBand::reset()
{
    audioL.reset(); audioR.reset();
    detL.reset();   detR.reset();
    envL.reset();   envR.reset();
    appliedGainDb = params.gainDb;
    coeffGainDb   = params.gainDb - 1.0f; // force coefficient refresh
    dynMeterDb    = 0.0f;
    structDirty   = true;
}

void DynamicBand::processSample (float& l, float& r, float keyL, float keyR) noexcept
{
    if (! params.enabled)
        return;

    const bool useKey = (params.dynSource == DetectorSource::External);
    float targetDynGainDb = 0.0f;

    if (dynamicsActive())
    {
        // Detector input follows the placed channel(s), unless an external
        // sidechain key overrides it.
        float detInL = l, detInR = r;
        switch (params.placement)
        {
            case Placement::Left:
                detInL = detInR = useKey ? keyL : l;
                break;
            case Placement::Right:
                detInL = detInR = useKey ? keyR : r;
                break;
            case Placement::Mid:
                detInL = detInR = useKey ? (keyL + keyR) * 0.5f : (l + r) * 0.5f;
                break;
            case Placement::Side:
                detInL = detInR = useKey ? (keyL - keyR) * 0.5f : (l - r) * 0.5f;
                break;
            default:
                if (useKey) { detInL = keyL; detInR = keyR; }
                break;
        }

        const float dl = detL.processSample (detInL);
        const float dr = detR.processSample (detInR);
        const float eL = envL.process (dl);
        const float eR = envR.process (dr);
        // Stereo-linked detector: the hotter channel drives the gain law.
        const float envDb = 20.0f * std::log10 ((eL > eR ? eL : eR) + 1.0e-6f);

        const float overDb = envDb - params.thresholdDb;
        if (overDb > 0.0f && params.ratio > 1.0f)
            targetDynGainDb = -overDb * (1.0f - 1.0f / params.ratio);
    }

    // Smooth the total applied gain, then refresh coefficients only when the
    // smoothed gain has actually moved (avoids per-sample sin/cos).
    const float targetGainDb = params.gainDb + targetDynGainDb;
    appliedGainDb += gainSmoothCoef * (targetGainDb - appliedGainDb);

    if (structDirty || std::fabs (appliedGainDb - coeffGainDb) > 1.0e-4f)
    {
        audioL.setParams (params.type, params.freq, params.q, appliedGainDb, sr);
        audioR.setParams (params.type, params.freq, params.q, appliedGainDb, sr);
        coeffGainDb = appliedGainDb;
        structDirty = false;
    }

    // Meter follows the dynamic portion only (0 when static).
    dynMeterDb += gainSmoothCoef * (targetDynGainDb - dynMeterDb);

    // Audio routing per placement. Mid/side use the exact-reconstruction
    // pair m = (l+r)/2, s = (l-r)/2 so untouched channels pass bit-clean.
    switch (params.placement)
    {
        case Placement::Left:
            l = audioL.processSample (l);
            break;
        case Placement::Right:
            r = audioR.processSample (r);
            break;
        case Placement::Mid:
        {
            float m = (l + r) * 0.5f;
            const float s = (l - r) * 0.5f;
            m = audioL.processSample (m);
            l = m + s;
            r = m - s;
            break;
        }
        case Placement::Side:
        {
            const float m = (l + r) * 0.5f;
            float s = (l - r) * 0.5f;
            s = audioL.processSample (s);
            l = m + s;
            r = m - s;
            break;
        }
        default:
            l = audioL.processSample (l);
            r = audioR.processSample (r);
            break;
    }
}

float DynamicBand::magnitudeResponseDb (float freqHz) const
{
    if (! params.enabled)
        return 0.0f;
    Biquad tmp;
    tmp.setParams (params.type, params.freq, params.q, params.gainDb, sr);
    return tmp.magnitudeResponseDb (freqHz, sr);
}

void DynamicBand::configureDetector()
{
    // Band-appropriate detection filter (see DynamicBand.h for rationale).
    switch (params.type)
    {
        case Biquad::Type::Bell:
        case Biquad::Type::Notch:
            detL.setParams (Biquad::Type::Bandpass, params.freq, params.q, 0.0f, sr);
            detR.setParams (Biquad::Type::Bandpass, params.freq, params.q, 0.0f, sr);
            break;
        case Biquad::Type::LowShelf:
            detL.setParams (Biquad::Type::LowPass, params.freq, 0.71f, 0.0f, sr);
            detR.setParams (Biquad::Type::LowPass, params.freq, 0.71f, 0.0f, sr);
            break;
        case Biquad::Type::HighShelf:
            detL.setParams (Biquad::Type::HighPass, params.freq, 0.71f, 0.0f, sr);
            detR.setParams (Biquad::Type::HighPass, params.freq, 0.71f, 0.0f, sr);
            break;
        default:
            // HighPass/LowPass: no detection needed (dynamics N/A in v1).
            break;
    }
}

} // namespace RoninEQ
