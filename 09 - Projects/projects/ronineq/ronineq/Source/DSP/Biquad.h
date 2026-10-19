/*
    RoninEQ — DynamicBand DSP core
    Biquad.h : RBJ Audio-EQ-Cookbook biquad filter.

    Pure C++17, no JUCE dependency. Clean-room implementation written for
    the RoninEQ project (AGPL-3.0). Coefficient formulas follow Robert
    Bristow-Johnson's "Audio EQ Cookbook" (public domain recipes).
*/
#pragma once

#include <cmath>

namespace RoninEQ
{

class Biquad
{
public:
    enum class Type
    {
        Bell = 0,     // peaking EQ
        LowShelf,     // low shelving
        HighShelf,   // high shelving
        HighPass,    // high-pass
        LowPass,     // low-pass
        Notch,       // notch (band reject)
        // Detection-only: constant 0 dB peak-gain bandpass used for the
        // dynamics sidechain. Never exposed in the plugin UI type list.
        Bandpass
    };

    Biquad() = default;

    // Recompute coefficients. freq in Hz (clamped to (0, nyquist)), Q > 0,
    // gainDb only used by Bell/LowShelf/HighShelf.
    void setParams (Type type, float freqHz, float q, float gainDb, float sampleRate);

    // Process one sample (Direct Form I, normalized so a0 == 1).
    float processSample (float x) noexcept
    {
        const float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = flushDenormal (x1); x1 = flushDenormal (x);
        y2 = flushDenormal (y1); y1 = flushDenormal (y);
        return y;
    }

    void reset() noexcept { x1 = x2 = y1 = y2 = 0.0f; }

    // Unity (passthrough): 0 dB at all frequencies.
    void setUnity() noexcept
    {
        type = Type::Bell;
        b0 = 1.0f; b1 = 0.0f; b2 = 0.0f; a1 = 0.0f; a2 = 0.0f;
        reset();
    }

    // Magnitude response in dB at freqHz (for UI curve drawing).
    // Uses the raw (unnormalized-then-normalized) transfer function H(z).
    float magnitudeResponseDb (float freqHz, float sampleRate) const noexcept;

    Type getType() const noexcept { return type; }

private:
    Type  type      = Type::Bell;
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;          // a0 is normalized to 1
    float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;

    static float flushDenormal (float v) noexcept
    {
        // Cheap denormal guard: values this small are inaudible and can
        // otherwise burn CPU on some chips when they go denormal.
        return (std::fabs (v) < 1.0e-18f) ? 0.0f : v;
    }
};

} // namespace RoninEQ
