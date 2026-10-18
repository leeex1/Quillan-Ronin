/*
    RoninEQ — DynamicBand DSP core
    EnvelopeFollower.cpp : peak detector with attack/release ballistics.

    Pure C++17, no JUCE dependency. Clean-room implementation.
*/
#include "EnvelopeFollower.h"
#include <cmath>

namespace RoninEQ
{

void EnvelopeFollower::setSampleRate (float sampleRate) noexcept
{
    sr = sampleRate > 0.0f ? sampleRate : 48000.0f;
    updateCoeffs();
}

void EnvelopeFollower::setAttackRelease (float attackMs, float releaseMs) noexcept
{
    atkMs = attackMs  < 0.01f ? 0.01f : attackMs;
    relMs = releaseMs < 1.0f  ? 1.0f  : releaseMs;
    updateCoeffs();
}

void EnvelopeFollower::reset() noexcept
{
    level = 0.0f;
}

float EnvelopeFollower::process (float x) noexcept
{
    const float target = std::fabs (x);
    const float coef = (target > level) ? attackCoef : releaseCoef;
    level += coef * (target - level);
    return level;
}

float EnvelopeFollower::getLevelDb() const noexcept
{
    // Floor at -120 dB so silence never produces -inf.
    return 20.0f * std::log10 (level + 1.0e-6f);
}

void EnvelopeFollower::updateCoeffs() noexcept
{
    // One-pole smoothing coefficient for a time constant of t seconds:
    //   coef = 1 - exp(-1 / (t * sr))
    // Attack/release times are the time to ~63% of the way there, the
    // standard convention for envelope followers.
    attackCoef  = 1.0f - std::exp (-1.0f / ((atkMs * 0.001f) * sr));
    releaseCoef = 1.0f - std::exp (-1.0f / ((relMs * 0.001f) * sr));
}

} // namespace RoninEQ
