/*
    RoninEQ — DynamicBand DSP core
    DynamicBand.h : one parametric band with optional downward dynamics.

    Signal flow per sample (stereo):
      detection:  in -> detection biquad (band-appropriate) -> peak envelope
                  followers (L/R) -> stereo link = max(L, R)
      gain law:   over = envDb - thresholdDb
                  dynGainDb = over > 0 ? -over * (1 - 1/ratio) : 0   (downward only)
      audio:      in -> audio biquad at (staticGainDb + dynGainDb), smoothed

    The detector runs on a band-appropriate filter, NOT on the audio band
    filter at 0 dB (which would be unity and useless for detection):
      Bell     -> bandpass at freq, Q
      Notch    -> bandpass at freq, Q
      LowShelf -> low-pass at freq (shelf region energy)
      HighShelf-> high-pass at freq (shelf region energy)
    High-pass / low-pass bands carry no gain parameter, so dynamics do not
    apply to them in v1 (dynEnabled is ignored for those types).

    Pure C++17, no JUCE dependency. Clean-room implementation.
*/
#pragma once

#include "Biquad.h"
#include "EnvelopeFollower.h"

namespace RoninEQ
{

enum class Placement
{
    Stereo = 0, // linked stereo (both channels through the band)
    Left,       // left channel only, right passes untouched
    Right,      // right channel only
    Mid,        // mid (L+R)/2 only, side passes untouched
    Side        // side (L-R)/2 only, mid passes untouched
};

enum class DetectorSource
{
    Internal = 0, // detector follows the (placed) program signal
    External      // detector follows the sidechain key input
};

struct BandParams
{
    bool         enabled     = true;
    Biquad::Type type        = Biquad::Type::Bell;
    float        freq        = 1000.0f;   // Hz
    float        gainDb      = 0.0f;      // static gain, dB
    float        q           = 1.0f;
    Placement    placement   = Placement::Stereo;
    bool         dynEnabled  = false;
    DetectorSource dynSource = DetectorSource::Internal;
    float        thresholdDb = -24.0f;
    float        ratio       = 4.0f;      // 1 = no reduction
    float        attackMs    = 10.0f;
    float        releaseMs   = 100.0f;
};

class DynamicBand
{
public:
    DynamicBand() = default;

    void setSampleRate (float sampleRate);
    void setParams (const BandParams& p);
    const BandParams& getParams() const noexcept { return params; }
    void reset();

    // In-place stereo sample processing. keyL/keyR feed the detector only
    // when params.dynSource == DetectorSource::External, otherwise ignored.
    void processSample (float& l, float& r, float keyL, float keyR) noexcept;

    // Backwards-compatible overload: internal detection.
    void processSample (float& l, float& r) noexcept
    {
        processSample (l, r, l, r);
    }

    // Static magnitude response in dB (for UI curve drawing). Builds a
    // throwaway filter from the stored params, so it never touches the
    // live audio-thread filter state.
    float magnitudeResponseDb (float freqHz) const;

    // Current (smoothed) dynamic gain in dB, for UI metering. <= 0.
    float getDynamicGainDb() const noexcept { return dynMeterDb; }

    bool dynamicsActive() const noexcept
    {
        return params.dynEnabled && isGainType (params.type);
    }

    static bool isGainType (Biquad::Type t) noexcept
    {
        return t == Biquad::Type::Bell || t == Biquad::Type::LowShelf
            || t == Biquad::Type::HighShelf || t == Biquad::Type::Notch;
    }

private:
    void configureDetector();

    float sr = 48000.0f;
    BandParams params;

    Biquad audioL, audioR;   // the actual EQ band, stereo pair
    Biquad detL, detR;       // sidechain detection filters
    EnvelopeFollower envL, envR;

    float envAttackMs = 10.0f, envReleaseMs = 100.0f;
    float gainSmoothCoef = 0.01f;
    float appliedGainDb = 0.0f;  // smoothed total gain driving the audio filter
    float coeffGainDb   = -1.0f; // gain the current coefficients were built with
    float dynMeterDb    = 0.0f;  // smoothed dynamic portion, for metering
    bool  structDirty   = true;
};

} // namespace RoninEQ
