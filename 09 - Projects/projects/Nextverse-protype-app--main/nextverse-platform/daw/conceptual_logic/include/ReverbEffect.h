#pragma once
#include "AudioEffect.h"

/**
 * @class ReverbEffect
 * @brief An audio effect that simulates reverberation.
 *
 * This class implements the AudioEffect interface to provide a reverb effect.
 * It includes parameters to control the characteristics of the reverb.
 */
class ReverbEffect : public AudioEffect {
public:
    /**
     * @brief Constructor for ReverbEffect.
     * Initializes with a default name "Reverb Effect".
     */
    ReverbEffect();
    /**
     * @brief Destructor for ReverbEffect.
     */
    ~ReverbEffect() override;

    /**
     * @brief Processes audio data to apply a reverb effect.
     * Overrides the pure virtual method from AudioEffect.
     * @param inputBuffer Pointer to the input audio buffer.
     * @param outputBuffer Pointer to the output audio buffer.
     * @param numSamples Number of samples to process.
     * @return True if processing was successful, false on error.
     */
    bool process(float* inputBuffer, float* outputBuffer, int numSamples) override;

    // Reverb-specific parameters
    float roomSize = 0.5f; ///< Controls the perceived size of the simulated room (0.0 to 1.0).
    float damping = 0.5f;  ///< Controls the high-frequency damping of the reverb (0.0 to 1.0).
    float wetLevel = 0.33f;///< Controls the amount of processed (reverb) signal mixed into the output (0.0 to 1.0).
    float dryLevel = 0.67f;///< Controls the amount of original (dry) signal mixed into the output (0.0 to 1.0).
};
