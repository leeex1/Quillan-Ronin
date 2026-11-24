#include "MidiTrack.h"
#include "EventTypes.h"  // Will be included when MidiEvent is fully defined in EventTypes.h
#include "AudioEffect.h" // Include AudioEffect header

MidiTrack::MidiTrack(const std::string& name) : trackName(name) {
    log("MidiTrack '" + name + "' created.");
}

MidiTrack::~MidiTrack() {
    log("MidiTrack '" + trackName + "' destroying effects...");
    for (AudioEffect* effect : effects_) {
        delete effect; // Clean up dynamically allocated effects
    }
    effects_.clear();
    log("MidiTrack '" + trackName + "' destroyed.");
}

void MidiTrack::setName(const std::string& name) {
    log("Name changed from '" + trackName + "' to '" + name + "'.");
    trackName = name;
}

std::string MidiTrack::getName() const {
    return trackName;
}

// Add actual MidiEvent struct definition before uncommenting push_back
void MidiTrack::addEvent(const MidiEvent& event) {
    // log("Adding MIDI event, Note: " + std::to_string(event.noteNumber)); // Example if event has noteNumber
    midiEvents.push_back(event);
}

const std::vector<MidiEvent>& MidiTrack::getEvents() const {
    return midiEvents;
}

// --- Audio Effects ---
void MidiTrack::addEffect(AudioEffect* effect) {
    if (effect) {
        effects_.push_back(effect);
        log("Added effect '" + effect->getName() + "' to track '" + trackName + "'.");
    } else {
        logError("Attempted to add a null effect pointer.");
    }
}

const std::vector<AudioEffect*>& MidiTrack::getEffects() const {
    return effects_;
}
