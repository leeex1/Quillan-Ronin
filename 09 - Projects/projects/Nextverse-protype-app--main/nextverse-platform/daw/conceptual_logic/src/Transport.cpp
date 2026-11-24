#include "Transport.h"
#include "EditSession.h" // Required for EditSession context
#include "AudioTrack.h"  // For iterating audio tracks
#include "MidiTrack.h"   // For iterating midi tracks
#include "AudioEffect.h" // For accessing effect names, processing
#include <thread>        // For std::this_thread::sleep_for
#include <chrono>        // For std::chrono::milliseconds
#include <algorithm>     // For std::copy, std::max
#include <cmath>         // For std::sin, std::abs

Transport::Transport() : currentPosition_(0.0), isPlaying_(false), isRecording_(false) {
    log("Transport Initialized.");
}

Transport::~Transport() {
    log("Transport Destroyed.");
}

void Transport::processRealTimeAudio(EditSession& session, double currentTime) {
    // This is a conceptual placeholder for real-time processing
    log("Processing real-time audio for session '" + session.getSessionName() + "' at time " + std::to_string(currentTime));

    // Render one block. Active clips contribute a deterministic test tone so the
    // signal path carries real samples rather than permanent silence.
    const int dummyBufferSize = 512;
    constexpr double sampleRate = 44100.0;
    float dummyInputBuffer[dummyBufferSize] = {0.0f};
    float dummyOutputBuffer[dummyBufferSize] = {0.0f};

    for (const auto& track : session.getAudioTracks()) {
        int clip_index = 0;
        for (const auto& clip : track.getAllClips()) {
            if (currentTime >= clip.startTime && currentTime < (clip.startTime + clip.duration)) {
                const double frequency = 220.0 + 110.0 * clip_index;
                for (int i = 0; i < dummyBufferSize; ++i) {
                    const double t = currentTime + static_cast<double>(i) / sampleRate;
                    dummyInputBuffer[i] += static_cast<float>(0.25 * std::sin(2.0 * 3.141592653589793 * frequency * t));
                }
            }
            ++clip_index;
        }
    }
    std::copy(dummyInputBuffer, dummyInputBuffer + dummyBufferSize, dummyOutputBuffer);

    // Process Audio Tracks
    for (const auto& track : session.getAudioTracks()) { // Using const reference
        log("Processing audio track: " + track.getName());
        // Simulate fetching audio data for the current time from the track's clips
        for (const auto& clip : track.getAllClips()) {
            if (currentTime >= clip.startTime && currentTime < (clip.startTime + clip.duration)) {
                log("  - Clip '" + clip.name + "' is active.");
                // In a real system, actual audio data from clip would be processed here.
            }
        }
        // Simulate applying effects
        for (auto* effect : track.getEffects()) { // Changed to auto*
            if (effect && effect->isEnabled()) {
                log("  - Applying effect: " + effect->getName() + " on track " + track.getName());
                // Conceptual call to process - actual buffers and samples would be used
                effect->process(dummyInputBuffer, dummyOutputBuffer, dummyBufferSize);
            }
        }
    }

    // Process MIDI Tracks (conceptual: synthesize MIDI, then apply effects)
    for (const auto& track : session.getMidiTracks()) { // Using const reference
        log("Processing MIDI track: " + track.getName() + " (instrument: " + track.instrumentName + ")");
        // Simulate synthesizing MIDI to audio for the current time
        // ... (MIDI processing logic would go here) ...
        // Then, apply effects to the synthesized audio
        for (auto* effect : track.getEffects()) { // Changed to auto*
            if (effect && effect->isEnabled()) {
                log("  - Applying effect: " + effect->getName() + " on MIDI track " + track.getName() + "'s output.");
                effect->process(dummyInputBuffer, dummyOutputBuffer, dummyBufferSize); // Conceptual
            }
        }
    }
    lastPeak_ = 0.0;
    for (int i = 0; i < dummyBufferSize; ++i) {
        lastPeak_ = std::max(lastPeak_, static_cast<double>(std::abs(dummyOutputBuffer[i])));
    }
    log("Rendered block peak amplitude: " + std::to_string(lastPeak_));
    log("Finished conceptual real-time audio processing for current time slice.");
}

void Transport::play(EditSession& session) {
    if (!isPlaying_) {
        isPlaying_ = true;
        isRecording_ = false; // Cannot record and play simultaneously
        // currentPosition_ = 0.0; // Reset or resume depending on desired behavior, for now let's keep it simple
        log("Playback started for session: " + session.getSessionName() + " from " + std::to_string(currentPosition_) + "s.");

        // Simulate playback loop
        // In a real DAW, this loop would be driven by an audio callback from the sound card.
        double sessionDuration = session.getTotalDuration();
        while (isPlaying_ && currentPosition_ < sessionDuration) {
            processRealTimeAudio(session, currentPosition_);

            // Simulate time passing
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Placeholder for audio buffer duration
            currentPosition_ += 0.1; // Advance time by 0.1s for simulation

            // log("Current time " + std::to_string(currentPosition_)); // Can be too verbose

            // Simplified break for testing, remove for continuous playback up to sessionDuration
            // if (currentPosition_ > 2.0) { // Simulate short playback for testing
            //     log("Simulated playback reached 2.0s, stopping for test.");
            //     break;
            // }
        }
        if (isPlaying_) { // Stopped due to reaching end or other condition
           stop(session); // Call stop to finalize state
        }
    } else {
        log("Already playing.");
    }
}

void Transport::stop(EditSession& session) {
    if (isPlaying_ || isRecording_) {
        log("Stop pressed for session: " + session.getSessionName() + ". Was "
              + (isPlaying_ ? "playing" : "")
              + (isRecording_ ? "recording" : "")
              + ". Final position: " + std::to_string(currentPosition_) + "s.");
        isPlaying_ = false;
        isRecording_ = false;
        // currentPosition_ = 0.0; // Optionally reset position on stop
    } else {
        log("Already stopped.");
    }
}

void Transport::record(EditSession& session) {
    if (!isRecording_) {
        isRecording_ = true;
        isPlaying_ = false; // Stop playback if it was on
        log("Record armed for session: " + session.getSessionName() + ". Current position: " + std::to_string(currentPosition_) + "s.");
        // To start actual recording, play() would usually be called while isRecording_ is true.
    } else {
        log("Already recording or armed. Press stop to disarm/stop recording.");
    }
}

void Transport::seek(double timeInSeconds, EditSession& session) {
    bool wasPlaying = isPlaying_;
    if (isPlaying_) {
        // For simplicity, just update position. A real seek might involve more state changes.
        log("Seeking while playing is not fully handled in this conceptual model. Stopping first.");
        stop(session);
    }

    currentPosition_ = timeInSeconds;
    if (currentPosition_ < 0) currentPosition_ = 0;
    // Consider session.getTotalDuration() as upper bound if needed
    // if (currentPosition_ > session.getTotalDuration()) currentPosition_ = session.getTotalDuration();

    log("Seek to " + std::to_string(currentPosition_) + "s in session: " + session.getSessionName());

    if (wasPlaying) {
        // play(session); // Optionally resume playing, or require explicit play command
        log("Playback was active. Transport stopped for seek. Press play to resume.");
    }
}

double Transport::getCurrentPosition() const {
    return currentPosition_;
}

bool Transport::isPlaying() const {
    return isPlaying_;
}

bool Transport::isRecording() const {
    return isRecording_;
}
