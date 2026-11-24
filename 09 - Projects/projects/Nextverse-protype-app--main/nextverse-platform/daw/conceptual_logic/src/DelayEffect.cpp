#include "DelayEffect.h"
#include <algorithm> // For std::max
#include <cstring>   // For std::memcpy for dry signal pass-through

DelayEffect::DelayEffect(double sampleRate)
    : AudioEffect("Delay Effect"), current_sample_rate_(sampleRate) {
    updateDelayBufferSize(); // Initialize buffer size based on default delayTimeSeconds
    log("Initialized. Delay Time: " + std::to_string(delayTimeSeconds) + "s, Feedback: " + std::to_string(feedback) + ", Sample Rate: " + std::to_string(current_sample_rate_));
}

DelayEffect::~DelayEffect() {
    // Log message will be printed by base class destructor
}

void DelayEffect::setSampleRate(double sampleRate) {
    if (sampleRate <= 0) {
        logError("Invalid sample rate: " + std::to_string(sampleRate));
        return;
    }
    current_sample_rate_ = sampleRate;
    updateDelayBufferSize();
    log("Sample rate set to: " + std::to_string(current_sample_rate_));
}

void DelayEffect::updateDelayBufferSize() {
    // Calculate delay buffer size based on current_sample_rate_ and delayTimeSeconds
    // Ensure a minimum size to avoid issues, or handle zero delay time appropriately.
    int newSize = static_cast<int>(std::max(1.0, current_sample_rate_ * delayTimeSeconds));
    if (delay_buffer_size_ != newSize) {
        delay_buffer_size_ = newSize;
        delay_buffer_.assign(delay_buffer_size_, 0.0f); // Resize and fill with zeros
        write_position_ = 0; // Reset write position
        log("Delay buffer resized to: " + std::to_string(delay_buffer_size_) + " samples for " + std::to_string(delayTimeSeconds) + "s.");
    }
}


bool DelayEffect::process(float* inputBuffer, float* outputBuffer, int numSamples) {
    if (!enabled) {
        if (inputBuffer != outputBuffer) {
            std::memcpy(outputBuffer, inputBuffer, numSamples * sizeof(float));
        }
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
    if (delay_buffer_size_ == 0) { // Can happen if delayTimeSeconds is 0 or sample rate is not set
        logError("Delay buffer is not initialized. Call setSampleRate() and ensure delayTimeSeconds > 0.");
         if (inputBuffer != outputBuffer) { // Pass through if buffer not ready
            std::memcpy(outputBuffer, inputBuffer, numSamples * sizeof(float));
        }
        return false;
    }

    // Re-calculate buffer size if delay time has changed since last update.
    // This is a simplified way; ideally, parameter changes are handled more explicitly.
    if (static_cast<int>(current_sample_rate_ * delayTimeSeconds) != delay_buffer_size_ ) {
        // RT-safe: do not allocate on the audio thread. Buffer size is latched
        // in setSampleRate(); live delay-time changes use a clamped read offset.
         if (delay_buffer_size_ == 0) { // check again after update
            logError("Delay buffer is still not initialized after update.");
            if (inputBuffer != outputBuffer) {
                std::memcpy(outputBuffer, inputBuffer, numSamples * sizeof(float));
            }
            return false;
        }
    }


    const int buffer_size = static_cast<int>(delay_buffer_.size());
    int delay_samples = static_cast<int>(current_sample_rate_ * delayTimeSeconds);
    if (delay_samples < 1) { delay_samples = 1; }
    if (delay_samples > buffer_size) { delay_samples = buffer_size; }

    for (int i = 0; i < numSamples; ++i) {
        // Read offset is clamped to the pre-allocated buffer: no resize, no allocation.
        int read_position = (write_position_ - delay_samples + buffer_size) % buffer_size;

        float delayed_sample = delay_buffer_[read_position];

        // Store current input sample mixed with feedback from the delayed sample into the buffer
        float current_input_with_feedback = inputBuffer[i] + delayed_sample * feedback;
        delay_buffer_[write_position_] = current_input_with_feedback;

        // Mix dry input with wet (delayed) signal for the output
        outputBuffer[i] = (inputBuffer[i] * (1.0f - mixLevel)) + (delayed_sample * mixLevel);

        // Advance write position
        write_position_ = (write_position_ + 1) % delay_buffer_.size();
    }

    // Logging per buffer is too verbose for real-time. Log on significant event/error.
    return true;
}
