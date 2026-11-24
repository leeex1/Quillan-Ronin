#include "EditSession.h"
#include <algorithm> // For std::max in getTotalDuration (conceptual)
#include <utility>   // For std::move (track ownership transfer)

EditSession::EditSession(const std::string& name) : sessionName(name) {
    log("Session '" + name + "' created.");
}

EditSession::~EditSession() {
    log("Session '" + sessionName + "' destroyed.");
}

void EditSession::setSessionName(const std::string& name) {
    log("Name changed from '" + sessionName + "' to '" + name + "'.");
    sessionName = name;
}

std::string EditSession::getSessionName() const {
    return sessionName;
}

void EditSession::addAudioTrack(AudioTrack&& track) {
    const std::string track_name = track.getName(); // capture before move
    audioTracks.push_back(std::move(track));
    log("Audio track '" + track_name + "' added to session '" + sessionName + "'.");
}

void EditSession::addMidiTrack(MidiTrack&& track) {
    const std::string track_name = track.getName(); // capture before move
    midiTracks.push_back(std::move(track));
    log("MIDI track '" + track_name + "' added to session '" + sessionName + "'.");
}

std::vector<AudioTrack>& EditSession::getAudioTracks() {
    return audioTracks;
}

const std::vector<AudioTrack>& EditSession::getAudioTracks() const {
    return audioTracks;
}

std::vector<MidiTrack>& EditSession::getMidiTracks() {
    return midiTracks;
}

const std::vector<MidiTrack>& EditSession::getMidiTracks() const {
    return midiTracks;
}

double EditSession::getTotalDuration() const {
    // Placeholder implementation: Could calculate based on the end of the last clip/event.
    // For now, let's return a fixed duration or a calculation based on number of tracks/clips.
    double maxDuration = 0.0;
    for (const auto& track : audioTracks) {
        for (const auto& clip : track.getAllClips()) { // Assuming getAllClips() exists
            if (clip.startTime + clip.duration > maxDuration) {
                maxDuration = clip.startTime + clip.duration;
            }
        }
    }
    for (const auto& track : midiTracks) {
        // Similar logic if MidiTrack has getEvents() and events have duration
        // For now, let's assume MIDI tracks don't contribute to duration in this placeholder
        // To silence unused variable warning, conceptually use track:
        (void)track; // Indicate track is intentionally unused in this placeholder.
    }
    if (maxDuration == 0.0) { // If no clips, default to a short duration for playback simulation
        return 10.0; // Default 10 seconds if empty
    }
    return maxDuration;
}
