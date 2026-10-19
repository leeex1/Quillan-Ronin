/*
    RoninEQ — LinearPhaseFIR.h : fixed-size linear-phase FIR convolver.

    One instance per channel. Coefficients are written from the audio thread
    (memcpy into a fixed array — no allocation after construction), so IR
    updates are realtime-safe. Group delay is exactly kTaps/2 samples and is
    reported to the host for compensation.

    Pure C++17, no JUCE dependency. Clean-room implementation.
*/
#pragma once

#include <algorithm>
#include <cstring>

namespace RoninEQ
{

class LinearPhaseFIR
{
public:
    static constexpr int kTaps = 1024;
    static constexpr int kLatencySamples = kTaps / 2; // exact group delay

    LinearPhaseFIR() { reset(); }

    void reset() noexcept
    {
        std::fill (std::begin (coeffs), std::end (coeffs), 0.0f);
        coeffs[kLatencySamples] = 1.0f; // unity until a real IR is loaded
        std::fill (std::begin (state), std::end (state), 0.0f);
        pos = 0;
    }

    // Realtime-safe: fixed-size copy, no allocation.
    void setCoefficients (const float* c) noexcept
    {
        std::memcpy (coeffs, c, sizeof (coeffs));
    }

    float processSample (float x) noexcept
    {
        state[pos] = x;
        float y = 0.0f;
        // Unrolled-by-4 direct form over the ring buffer.
        int idx = pos;
        for (int t = 0; t < kTaps; t += 4)
        {
            y += coeffs[t]     * state[idx];
            idx = (idx == 0) ? kTaps - 1 : idx - 1;
            y += coeffs[t + 1] * state[idx];
            idx = (idx == 0) ? kTaps - 1 : idx - 1;
            y += coeffs[t + 2] * state[idx];
            idx = (idx == 0) ? kTaps - 1 : idx - 1;
            y += coeffs[t + 3] * state[idx];
            idx = (idx == 0) ? kTaps - 1 : idx - 1;
        }
        pos = (pos + 1 < kTaps) ? pos + 1 : 0;
        // Cheap denormal guard on the accumulator output.
        return (y > -1.0e-18f && y < 1.0e-18f) ? 0.0f : y;
    }

private:
    float coeffs[kTaps];
    float state[kTaps];
    int pos = 0;
};

} // namespace RoninEQ
