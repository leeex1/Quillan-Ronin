#include <iostream>
#include <utility> // std::move for track ownership transfer
#include <vector> // Required for std::vector
#include "EditSession.h"
#include "AudioTrack.h"
#include "MidiTrack.h"
#include "EventTypes.h"
#include "Transport.h"
#include "ReverbEffect.h"
#include "DelayEffect.h"
#include "AudioEffect.h" // Make sure AudioEffect is included for direct manipulation if any

// Define a sample rate to be used throughout main.cpp for effects
const double MAIN_SAMPLE_RATE = 44100.0;

int main() {
    std::cout << "--- Conceptual DAW Test Runner ---" << std::endl;

    // 1. Create an EditSession
    EditSession session("My Conceptual Song");
    std::cout << "Created session: " << session.getSessionName() << std::endl;

    // 2. Create and add tracks
    // These are local track objects. When added to session, session stores copies.
    AudioTrack guitarTrack("Guitar");
    AudioTrack vocalTrack("Vocals");
    MidiTrack pianoTrack("Piano");
    MidiTrack drumTrack("Drums");

    // Ownership transfers into the session; the locals must not be copied.
    session.addAudioTrack(std::move(guitarTrack));
    session.addAudioTrack(std::move(vocalTrack));
    session.addMidiTrack(std::move(pianoTrack));
    session.addMidiTrack(std::move(drumTrack));
    std::cout << "Added " << session.getAudioTracks().size() << " audio tracks and "
              << session.getMidiTracks().size() << " MIDI tracks." << std::endl;

    // 3. Add events to tracks stored in the session
    if (!session.getAudioTracks().empty()) {
        // Get a reference to the track inside the session to modify it
        AudioTrack& sessionGuitarTrack = session.getAudioTracks()[0];
        sessionGuitarTrack.addClip("Guitar Clip 1", 0.5, 2.0, "guitar_clip1.wav"); // Using new addClip
        sessionGuitarTrack.addClip("Guitar Clip 2", 3.0, 2.5, "guitar_clip2.wav");
        std::cout << "Added 2 audio clips to track: " << sessionGuitarTrack.getName() << std::endl;
    }

    if (!session.getMidiTracks().empty()) {
        MidiTrack& sessionPianoTrack = session.getMidiTracks()[0];
        MidiEvent pianoChord1 = {0.0, 60, 100, 2.0};
        MidiEvent pianoChord2 = {2.0, 64, 90, 2.0};
        sessionPianoTrack.addEvent(pianoChord1);
        sessionPianoTrack.addEvent(pianoChord2);
        std::cout << "Added 2 MIDI events to track: " << sessionPianoTrack.getName() << std::endl;
    }

    // 4. Instantiate and use Transport
    Transport transport;
    std::cout << "\n--- Transport Operations ---" << std::endl;
    transport.seek(1.0, session);
    transport.play(session); // This will run its internal conceptual playback loop
    // The transport.play() is now blocking for simulation purposes,
    // so direct calls to stop/record/play after it might not be reached as originally intended
    // or would need play() to be non-blocking (e.g. run in a separate thread).
    // For this conceptual test, play() runs a short loop and then stops itself.
    // So, further calls will demonstrate state after that simulated playback.

    std::cout << "Transport finished its simulated play." << std::endl;
    transport.record(session);
    // transport.play(session); // To start actual recording conceptually - this would run another loop
    // std::cout << "Simulating 1 second of recording..." << std::endl; // This line might not be reached if play() is blocking
    transport.stop(session);

    // 5. Instantiate effects and add them to tracks within the session
    // Effects are owned by the track once added.
    std::cout << "\n--- Effect Setup ---" << std::endl;
    ReverbEffect* globalReverb = new ReverbEffect(); // Uses default sample rate
    DelayEffect* vocalDelay = new DelayEffect(MAIN_SAMPLE_RATE);

    globalReverb->setEnabled(true);
    vocalDelay->setEnabled(true);

    if (session.getAudioTracks().size() > 1) {
        AudioTrack& sessionVocalTrack = session.getAudioTracks()[1]; // Assuming vocal is the second audio track
        sessionVocalTrack.addEffect(vocalDelay); // vocalDelay pointer is now managed by vocalTrack
         // sessionVocalTrack.addEffect(globalReverb); // Example: also add reverb to vocals
                                                    // Note: globalReverb would then be owned by vocalTrack.
                                                    // If adding same instance to multiple, use shared_ptr or manage lifetime carefully.
                                                    // For simplicity, let's assume globalReverb is not added to a specific track here,
                                                    // or if it is, it shouldn't be deleted separately later.
    } else {
        delete vocalDelay; // Not added to any track, so delete manually
    }
    // If globalReverb is not added to any track, it should be deleted manually.
    // If it were added to a "master bus" or a track, that entity would own it.
    // For this test, let's assume it's a conceptual global effect not owned by a track.


    // 6. Conceptual direct effect processing test (if still needed)
    // This is now primarily handled within Transport::processRealTimeAudio
    // But we can still test the process method directly for sanity check.
    std::cout << "\n--- Direct Effect Process Sanity Check ---" << std::endl;
    const int dummyNumSamples = 512;
    float dummyInput[dummyNumSamples];
    float dummyOutput[dummyNumSamples];
    for(int i=0; i<dummyNumSamples; ++i) dummyInput[i] = 0.1f * (i % 10); // Fill with some dummy data

    std::cout << "Directly processing with globalReverb:" << std::endl;
    if(globalReverb->process(dummyInput, dummyOutput, dummyNumSamples)) {
        std::cout << "globalReverb processed successfully." << std::endl;
    } else {
        std::cout << "globalReverb processing failed." << std::endl;
    }

    // Note: vocalDelay is now owned by sessionVocalTrack if added.
    // To test it directly, we'd need a pointer to it.
    // If we want to test the instance added to the track:
    if (session.getAudioTracks().size() > 1 && !session.getAudioTracks()[1].getEffects().empty()) {
        AudioEffect* trackVocalDelay = session.getAudioTracks()[1].getEffects()[0]; // Assuming it's the first
        std::cout << "Directly processing with vocalDelay from track:" << std::endl;
        if(trackVocalDelay->process(dummyInput, dummyOutput, dummyNumSamples)) {
            std::cout << "Track's vocalDelay processed successfully." << std::endl;
        } else {
            std::cout << "Track's vocalDelay processing failed." << std::endl;
        }
        trackVocalDelay->setEnabled(false);
        std::cout << "Track's vocalDelay disabled. Processing again:" << std::endl;
        if(trackVocalDelay->process(dummyInput, dummyOutput, dummyNumSamples)) { // Should pass through
             std::cout << "Track's vocalDelay (disabled) processed." << std::endl;
        }
    }

    // Cleanup for effects not added to tracks (or if shared ownership isn't used)
    delete globalReverb; // globalReverb was not added to a track in this example path.

    std::cout << "\n--- End of Conceptual DAW Test ---" << std::endl;
    return 0;
}
