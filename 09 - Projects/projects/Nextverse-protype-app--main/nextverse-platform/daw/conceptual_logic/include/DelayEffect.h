#pragma once
#include "AudioEffect.h"
#include <vector> // For internal delay buffer

/**
 * @class DelayEffect
 * @brief An audio effect that creates echoes or delays.
 *
 * This class implements the AudioEffect interface to provide a delay effect.
 * It includes parameters for delay time, feedback, and mix level.
 */
class DelayEffect : public AudioEffect {
public:
    /**
     * @brief Constructor for DelayEffect.
     * Initializes with a default name "Delay Effect".
     * @param sampleRate The sample rate of the audio to be processed. Needed for delay time calculations.
     */
    DelayEffect(double sampleRate = 44100.0); // Default sample rate
    /**
     * @brief Destructor for DelayEffect.
     */
    ~DelayEffect() override;

    /**
     * @brief Processes audio data to apply a delay effect.
     * Overrides the pure virtual method from AudioEffect.
     * @param inputBuffer Pointer to the input audio buffer.
     * @param outputBuffer Pointer to the output audio buffer.
     * @param numSamples Number of samples to process.
     * @return True if processing was successful, false on error.
     */
    bool process(float* inputBuffer, float* outputBuffer, int numSamples) override;

    // Delay-specific parameters
    float delayTimeSeconds = 0.5f; ///< Delay time in seconds.
    float feedback = 0.4f;         ///< Feedback amount for repeated echoes (0.0 to <1.0 typically).
    float mixLevel = 0.5f;         ///< Mix between dry (original) and wet (delayed) signal (0.0 to 1.0).

    /**
     * @brief Sets the sample rate for the delay effect.
     * This is important for correctly calculating delay buffer sizes based on time.
     * @param sampleRate The new sample rate.
     */
    void setSampleRate(double sampleRate);

private:
    std::vector<float> delay_buffer_; ///< Internal buffer to store delayed samples.
    int delay_buffer_size_ = 0;       ///< Size of the delay buffer.
    int write_position_ = 0;          ///< Current position to write into the delay buffer.
    double current_sample_rate_;      ///< Current sample rate.

    /**
     * @brief Resizes the internal delay buffer based on current sample rate and delay time.
     */
    void updateDelayBufferSize();
};
