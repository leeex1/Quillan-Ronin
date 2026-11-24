#pragma once
#include <string>
#include <vector> // Might be useful for polyphonic MIDI notes or audio event properties later

struct AudioEvent {
    double startTime = 0.0;       // In seconds from the beginning of the track/session
    double duration = 0.0;        // In seconds
    std::string audioFileReference; // Path or identifier for the audio data
    // Could add properties like:
    // double fadeInDuration = 0.0;
    // double fadeOutDuration = 0.0;
    // float gain = 1.0f;
};

struct MidiEvent {
    double startTime = 0.0;       // In seconds
    int noteNumber = 60;          // Middle C
    int velocity = 100;           // 0-127
    double duration = 1.0;        // In seconds (for note-off, could also be a separate NoteOffEvent)
    // Could add properties like:
    // int channel = 1;
    // std::vector<ControlChange> controlChanges; // For CC messages
};

// Placeholder for potential ControlChange definition if needed later
// struct ControlChange {
//     int controllerNumber;
//     int controllerValue;
//     double offsetFromStartTime; // relative to MidiEvent startTime
// };
