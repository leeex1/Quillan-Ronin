#include "AudioTrack.h"
#include "EventTypes.h" // Will be included when AudioEvent is fully defined in EventTypes.h
#include "AudioEffect.h" // Include AudioEffect header
#include <algorithm> // For std::find_if, std::remove_if
#include <stdexcept> // For std::out_of_range
#include <utility>   // For std::move

AudioTrack::AudioTrack(const std::string& name) : trackName(name) {
    log("AudioTrack '" + name + "' created.");
}

AudioTrack::~AudioTrack() {
    log("AudioTrack '" + trackName + "' destroying effects...");
    for (AudioEffect* effect : effects_) {
        delete effect; // Clean up dynamically allocated effects
    }
    effects_.clear();
    log("AudioTrack '" + trackName + "' destroyed.");
}

void AudioTrack::setName(const std::string& name) {
    log("Name changed from '" + trackName + "' to '" + name + "'");
    trackName = name;
}

std::string AudioTrack::getName() const {
    return trackName;
}

// Add actual AudioEvent struct definition before uncommenting push_back
void AudioTrack::addEvent(const AudioEvent& event) {
    // Assuming AudioEvent has a 'type' member or similar for logging
    // log("Adding event of type: " + event.type);
    audioEvents.push_back(event);
}

const std::vector<AudioEvent>& AudioTrack::getEvents() const {
    return audioEvents;
}

// --- AudioClip CRUD Operations ---

int AudioTrack::addClip(const std::string& clipName, double startTime, double duration, const std::string& filePath) {
    if (clipName.empty()) {
        logError("Cannot add clip with empty name.");
        return -1;
    }
    if (duration <= 0) {
        logError("Cannot add clip with non-positive duration.");
        return -1;
    }
    // FilePath can be empty if it's, for instance, a "silence" clip or generated content.
    // Add specific validation for filePath if needed (e.g., check extension, existence)

    AudioClip newClip(nextClipId_++, clipName, startTime, duration, filePath);
    clips_.push_back(newClip);
    log("Added clip '" + clipName + "' (ID: " + std::to_string(newClip.id) +
        ", Path: " + filePath + ") at " + std::to_string(startTime) +
        "s for " + std::to_string(duration) + "s.");
    return newClip.id;
}

AudioClip AudioTrack::getClip(int clipId) const {
    auto it = std::find_if(clips_.begin(), clips_.end(),
                           [clipId](const AudioClip& clip) { return clip.id == clipId; });
    if (it != clips_.end()) {
        return *it;
    }
    logError("Clip with ID " + std::to_string(clipId) + " not found.");
    throw std::out_of_range("Clip not found with ID: " + std::to_string(clipId));
}

const std::vector<AudioClip>& AudioTrack::getAllClips() const {
    return clips_;
}

bool AudioTrack::updateClip(int clipId, const std::string& newName, double newStartTime, double newDuration, const std::string& newFilePath) {
    auto it = std::find_if(clips_.begin(), clips_.end(),
                           [clipId](AudioClip& clip) { return clip.id == clipId; }); // Non-const iterator needed
    if (it != clips_.end()) {
        if (newName.empty()) {
            logError("Cannot update clip ID " + std::to_string(clipId) + " with empty name.");
            return false;
        }
        if (newDuration <= 0) {
            logError("Cannot update clip ID " + std::to_string(clipId) + " with non-positive duration.");
            return false;
        }
        it->name = newName;
        it->startTime = newStartTime;
        it->duration = newDuration;
        it->filePath = newFilePath;
        log("Updated clip ID " + std::to_string(clipId) + " to Name: '" + newName +
            "', Path: " + newFilePath + ", Start: " + std::to_string(newStartTime) +
            "s, Duration: " + std::to_string(newDuration) + "s.");
        return true;
    }
    logError("Failed to update: Clip with ID " + std::to_string(clipId) + " not found.");
    return false;
}

bool AudioTrack::removeClip(int clipId) {
    auto it = std::remove_if(clips_.begin(), clips_.end(),
                             [clipId](const AudioClip& clip) { return clip.id == clipId; });
    if (it != clips_.end()) {
        clips_.erase(it, clips_.end());
        log("Removed clip ID " + std::to_string(clipId) + ".");
        return true;
    }
    logError("Failed to remove: Clip with ID " + std::to_string(clipId) + " not found.");
    return false;
}

// --- Audio Effects ---
void AudioTrack::addEffect(AudioEffect* effect) {
    if (effect) {
        effects_.push_back(effect);
        log("Added effect '" + effect->getName() + "' to track '" + trackName + "'.");
    } else {
        logError("Attempted to add a null effect pointer.");
    }
}

const std::vector<AudioEffect*>& AudioTrack::getEffects() const {
    return effects_;
}
