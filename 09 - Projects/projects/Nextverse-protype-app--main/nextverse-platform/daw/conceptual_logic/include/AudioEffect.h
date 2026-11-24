#pragma once
#include <vector>
#include <string>
#include <iostream> // For logging

// A very simple representation of an audio buffer for placeholder effects
// In a real system, this would be much more complex (e.g., juce::AudioBuffer)
// This using statement might become less relevant if process method uses raw pointers.
using AudioBuffer = std::vector<float>;

/**
 * @class AudioEffect
 * @brief Base class for audio effects.
 *
 * Defines the interface for audio effects that can be applied to audio data.
 * Derived classes must implement the `process` method to apply their specific effect.
 */
class AudioEffect {
public:
    /**
     * @brief Constructor for AudioEffect.
     * @param name The name of the effect.
     */
    AudioEffect(const std::string& name);
    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~AudioEffect();

    /**
     * @brief Processes a block of audio data.
     * Pure virtual method to be implemented by derived effect classes.
     * This method applies the audio effect to the input buffer and writes the result to the output buffer.
     * @param inputBuffer Pointer to the input audio buffer. Must not be null.
     * @param outputBuffer Pointer to the output audio buffer. Must not be null.
     * @param numSamples Number of samples to process. Must be greater than 0.
     * @return True if processing was successful, false on error (e.g., invalid parameters).
     */
    virtual bool process(float* inputBuffer, float* outputBuffer, int numSamples) = 0;

    /**
     * @brief Gets the name of the effect.
     * @return The name of the effect.
     */
    std::string getName() const;
    /**
     * @brief Enables or disables the effect.
     * @param enabled True to enable the effect, false to disable.
     */
    void setEnabled(bool enabled);
    /**
     * @brief Checks if the effect is currently enabled.
     * @return True if the effect is enabled, false otherwise.
     */
    bool isEnabled() const;

protected:
    std::string effectName; ///< The name of the effect.
    bool enabled;           ///< Current enabled state of the effect.

    /**
     * @brief Logs a message specific to this effect instance.
     * @param message The message string to log.
     */
    void log(const std::string& message) const {
        std::cout << "[" << effectName << " LOG]: " << message << std::endl;
    }

    /**
     * @brief Logs an error message specific to this effect instance.
     * @param message The error message string to log.
     */
    void logError(const std::string& message) const {
        std::cerr << "[" << effectName << " ERROR]: " << message << std::endl;
    }
};
