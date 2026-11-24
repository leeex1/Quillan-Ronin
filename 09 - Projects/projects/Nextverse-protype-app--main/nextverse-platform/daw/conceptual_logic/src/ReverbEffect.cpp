#include "ReverbEffect.h"
#include <cstring> // For std::memcpy if used for dry signal pass-through

ReverbEffect::ReverbEffect() : AudioEffect("Reverb Effect") {
    log("Initialized. Room Size: " + std::to_string(roomSize) + ", Wet: " + std::to_string(wetLevel));
}

ReverbEffect::~ReverbEffect() {
    // Log message will be printed by base class destructor after this.
}

bool ReverbEffect::process(float* inputBuffer, float* outputBuffer, int numSamples) {
    if (!enabled) {
        // If disabled, it's often best to pass through the dry signal
        // or fill outputBuffer with zeros if it's an insert effect on an empty track.
        // For simplicity here, let's assume pass-through if input and output are different.
        if (inputBuffer != outputBuffer) {
            std::memcpy(outputBuffer, inputBuffer, numSamples * sizeof(float));
        }
        // No log message here, or a very verbose one, as this happens per buffer.
        return true;
    }

    if (!inputBuffer || !outputBuffer) {
        logError("Process called with null buffer(s).");
        return false;
    }
    if (numSamples <= 0) {
        logError("Process called with zero or negative samples.");
        return false;
    }

    // Simulate reverb processing:
    // A real reverb would involve complex algorithms (e.g., Schroeder reverberators, convolution).
    // For this placeholder, we'll just mix dry and a "simulated" wet signal.
    for (int i = 0; i < numSamples; ++i) {
        float drySignal = inputBuffer[i];
        float wetSignal = drySignal * 0.5f; // Extremely simplified "wet" signal - replace with actual reverb algorithm

        // Apply wet/dry mix
        outputBuffer[i] = (drySignal * dryLevel) + (wetSignal * wetLevel);

        // Simulate some effect based on roomSize and damping (conceptual)
        // This is not sonically accurate, just a placeholder for parameter usage.
        outputBuffer[i] *= (1.0f + roomSize * 0.1f);
        if (damping > 0.5f) {
            outputBuffer[i] *= (1.0f - (damping - 0.5f) * 0.1f); // Reduce high-freq conceptually
        }
    }

    // log("Processed " + std::to_string(numSamples) + " samples with Room Size: " + std::to_string(roomSize));
    // Logging every buffer can be too verbose for real-time effects. Log on parameter change or error.
    return true;
}
