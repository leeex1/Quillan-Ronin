#include "AudioEffect.h"

AudioEffect::AudioEffect(const std::string& name) : effectName(name), enabled(true) {
    // Using the new log method, though it's called from constructor,
    // effectName is already initialized.
    std::cout << "[" << effectName << " LOG]: " << "Effect created." << std::endl;
}

AudioEffect::~AudioEffect() {
    // Similar logging adjustment for consistency
    std::cout << "[" << effectName << " LOG]: " << "Effect destroyed." << std::endl;
}

// The pure virtual 'process' method has no implementation in the base class.
// void AudioEffect::process(float* inputBuffer, float* outputBuffer, int numSamples)
// {
//     // This is a pure virtual function, so no base implementation.
//     // Derived classes must implement this.
// }

std::string AudioEffect::getName() const {
    return effectName;
}

void AudioEffect::setEnabled(bool en) {
    enabled = en;
    if (enabled) {
        log("Enabled.");
    } else {
        log("Disabled.");
    }
}

bool AudioEffect::isEnabled() const {
    return enabled;
}
