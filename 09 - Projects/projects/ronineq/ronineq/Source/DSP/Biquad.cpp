/*
    RoninEQ — DynamicBand DSP core
    Biquad.cpp : RBJ cookbook coefficient calculation.

    Pure C++17, no JUCE dependency. Clean-room implementation.
    Formulas: Robert Bristow-Johnson, "Audio EQ Cookbook".
*/
#include "Biquad.h"

namespace RoninEQ
{

void Biquad::setParams (Type newType, float freqHz, float q, float gainDb, float sampleRate)
{
    type = newType;

    const float nyquist = sampleRate * 0.5f;
    // Keep the frequency strictly inside (0, nyquist) so the cookbook
    // math never divides by zero or explodes at the rails.
    const float f0 = freqHz < 10.0f ? 10.0f : (freqHz > nyquist * 0.999f ? nyquist * 0.999f : freqHz);
    const float Q  = q < 0.05f ? 0.05f : (q > 24.0f ? 24.0f : q);

    const float w0    = 2.0f * 3.14159265358979323846f * f0 / sampleRate;
    const float cosW0 = std::cos (w0);
    const float sinW0 = std::sin (w0);
    const float alpha = sinW0 / (2.0f * Q);

    // Shelf filters in the cookbook use a shelf-slope parameter S; we use the
    // common simplification alpha = sin(w0)/(2Q) so one Q knob drives everything.
    float nb0, nb1, nb2, na0, na1, na2;

    switch (type)
    {
        case Type::Bell:
        {
            const float A = std::pow (10.0f, gainDb / 40.0f);
            nb0 = 1.0f + alpha * A;
            nb1 = -2.0f * cosW0;
            nb2 = 1.0f - alpha * A;
            na0 = 1.0f + alpha / A;
            na1 = -2.0f * cosW0;
            na2 = 1.0f - alpha / A;
            break;
        }
        case Type::LowShelf:
        {
            const float A = std::pow (10.0f, gainDb / 40.0f);
            const float sqrtA = std::sqrt (A);
            const float twoSqrtAAlpha = 2.0f * sqrtA * alpha;
            nb0 =      A * ((A + 1.0f) - (A - 1.0f) * cosW0 + twoSqrtAAlpha);
            nb1 =  2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosW0);
            nb2 =      A * ((A + 1.0f) - (A - 1.0f) * cosW0 - twoSqrtAAlpha);
            na0 =           (A + 1.0f) + (A - 1.0f) * cosW0 + twoSqrtAAlpha;
            na1 = -2.0f *          ((A - 1.0f) + (A + 1.0f) * cosW0);
            na2 =           (A + 1.0f) + (A - 1.0f) * cosW0 - twoSqrtAAlpha;
            break;
        }
        case Type::HighShelf:
        {
            const float A = std::pow (10.0f, gainDb / 40.0f);
            const float sqrtA = std::sqrt (A);
            const float twoSqrtAAlpha = 2.0f * sqrtA * alpha;
            nb0 =      A * ((A + 1.0f) + (A - 1.0f) * cosW0 + twoSqrtAAlpha);
            nb1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosW0);
            nb2 =      A * ((A + 1.0f) + (A - 1.0f) * cosW0 - twoSqrtAAlpha);
            na0 =           (A + 1.0f) - (A - 1.0f) * cosW0 + twoSqrtAAlpha;
            na1 =  2.0f *          ((A - 1.0f) - (A + 1.0f) * cosW0);
            na2 =           (A + 1.0f) - (A - 1.0f) * cosW0 - twoSqrtAAlpha;
            break;
        }
        case Type::HighPass:
        {
            const float c = (1.0f + cosW0) * 0.5f;
            nb0 = c; nb1 = -2.0f * c; nb2 = c;
            na0 = 1.0f + alpha; na1 = -2.0f * cosW0; na2 = 1.0f - alpha;
            break;
        }
        case Type::LowPass:
        {
            const float c = (1.0f - cosW0) * 0.5f;
            nb0 = c; nb1 = 2.0f * c; nb2 = c;
            na0 = 1.0f + alpha; na1 = -2.0f * cosW0; na2 = 1.0f - alpha;
            break;
        }
        case Type::Notch:
        {
            nb0 = 1.0f; nb1 = -2.0f * cosW0; nb2 = 1.0f;
            na0 = 1.0f + alpha; na1 = -2.0f * cosW0; na2 = 1.0f - alpha;
            break;
        }
        case Type::Bandpass:
        default:
        {
            // RBJ "bandpass, constant 0 dB peak gain": unity at center freq.
            nb0 = alpha; nb1 = 0.0f; nb2 = -alpha;
            na0 = 1.0f + alpha; na1 = -2.0f * cosW0; na2 = 1.0f - alpha;
            break;
        }
    }

    // Normalize so a0 == 1 (Direct Form I below assumes this).
    const float invA0 = 1.0f / na0;
    b0 = nb0 * invA0;
    b1 = nb1 * invA0;
    b2 = nb2 * invA0;
    a1 = na1 * invA0;
    a2 = na2 * invA0;
}

float Biquad::magnitudeResponseDb (float freqHz, float sampleRate) const noexcept
{
    const float w = 2.0f * 3.14159265358979323846f * freqHz / sampleRate;
    const float cw  = std::cos (w);
    const float cw2 = std::cos (2.0f * w);
    const float sw  = std::sin (w);
    const float sw2 = std::sin (2.0f * w);

    // H(e^jw) = (b0 + b1 e^-jw + b2 e^-j2w) / (1 + a1 e^-jw + a2 e^-j2w)
    const float numRe = b0 + b1 * cw  + b2 * cw2;
    const float numIm =     - b1 * sw  - b2 * sw2;
    const float denRe = 1.0f + a1 * cw + a2 * cw2;
    const float denIm =       - a1 * sw - a2 * sw2;

    const float numMag2 = numRe * numRe + numIm * numIm;
    const float denMag2 = denRe * denRe + denIm * denIm;
    if (denMag2 < 1.0e-24f)
        return -120.0f;

    return 10.0f * std::log10 (numMag2 / denMag2 + 1.0e-24f);
}

} // namespace RoninEQ
