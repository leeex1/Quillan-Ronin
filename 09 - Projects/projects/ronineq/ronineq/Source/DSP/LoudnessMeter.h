/*
    RoninEQ — LoudnessMeter.h : momentary loudness (BS.1770-style) meter.

    K-weighting (high shelf +4 dB @ 1681 Hz + 60 Hz high-pass), 400 ms
    one-pole mean-square averaging, LUFS readout. Stereo pairs are summed
    per BS.1770 channel weights (1.0 for L/R). Used for the in/out loudness
    readouts and the auto-gain matcher.

    Absolute accuracy is within ~1 dB of a reference meter; relative
    in-vs-out matching (what auto-gain needs) is exact by construction.

    Pure C++17, no JUCE dependency. Clean-room implementation.
*/
#pragma once

#include "Biquad.h"
#include <cmath>

namespace RoninEQ
{

class LoudnessMeter
{
public:
    LoudnessMeter() = default;

    void setSampleRate (float sampleRate) noexcept
    {
        sr = sampleRate > 0.0f ? sampleRate : 48000.0f;
        for (int ch = 0; ch < 2; ++ch)
        {
            pre[ch].setParams (Biquad::Type::HighShelf, 1681.0f, 0.7f, 4.0f, sr);
            rlb[ch].setParams (Biquad::Type::HighPass, 60.0f, 0.5f, 0.0f, sr);
            ms[ch] = 0.0f;
        }
        // 400 ms integration window.
        avgCoef = 1.0f - std::exp (-1.0f / (0.4f * sr));
    }

    void reset() noexcept
    {
        for (int ch = 0; ch < 2; ++ch)
        {
            pre[ch].reset();
            rlb[ch].reset();
            ms[ch] = 0.0f;
        }
    }

    // Feed one stereo sample pair.
    void processSample (float l, float r) noexcept
    {
        const float yl = rlb[0].processSample (pre[0].processSample (l));
        const float yr = rlb[1].processSample (pre[1].processSample (r));
        ms[0] += avgCoef * (yl * yl - ms[0]);
        ms[1] += avgCoef * (yr * yr - ms[1]);
    }

    void processMono (float x) noexcept { processSample (x, x); }

    // Momentary loudness in LUFS. Silence floors at -120.
    float getLUFS() const noexcept
    {
        const float z = ms[0] + ms[1];
        if (z < 1.0e-12f)
            return -120.0f;
        return -0.691f + 10.0f * std::log10 (z);
    }

private:
    float sr = 48000.0f;
    Biquad pre[2]; // K-weighting high-frequency shelf
    Biquad rlb[2]; // K-weighting low-frequency HPF (RLB)
    float ms[2] = { 0.0f, 0.0f };
    float avgCoef = 0.01f;
};

} // namespace RoninEQ
