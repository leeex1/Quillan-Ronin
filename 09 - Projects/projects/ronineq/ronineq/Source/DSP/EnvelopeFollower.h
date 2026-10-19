/*
    RoninEQ — DynamicBand DSP core
    EnvelopeFollower.h : peak detector with attack/release ballistics.

    Classic one-pole peak envelope follower:
      - rises toward |x| with the attack time constant
      - falls toward |x| with the release time constant
    Level is kept linear internally; use getLevelDb() for the dB value the
    dynamics gain computer works with.

    Pure C++17, no JUCE dependency. Clean-room implementation.
*/
#pragma once

namespace RoninEQ
{

class EnvelopeFollower
{
public:
    EnvelopeFollower() = default;

    void setSampleRate (float sampleRate) noexcept;
    void setAttackRelease (float attackMs, float releaseMs) noexcept;
    void reset() noexcept;

    // Feed one sample; returns the current linear envelope level.
    float process (float x) noexcept;

    float getLevel() const noexcept { return level; }
    float getLevelDb() const noexcept;

private:
    void updateCoeffs() noexcept;

    float sr = 48000.0f;
    float atkMs = 10.0f, relMs = 100.0f;
    float attackCoef = 0.0f, releaseCoef = 0.0f;
    float level = 0.0f;
};

} // namespace RoninEQ
