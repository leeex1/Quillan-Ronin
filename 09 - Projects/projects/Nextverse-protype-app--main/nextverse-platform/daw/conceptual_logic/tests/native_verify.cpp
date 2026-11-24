// Native verification for the conceptual DAW signal path.
// Proves the transport renders non-silent blocks for active clips and silence
// when no clip is active, and that track ownership is move-only.
#include <cassert>
#include <iostream>
#include <type_traits>
#include <utility>

#include "EditSession.h"
#include "AudioTrack.h"
#include "Transport.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "[FAIL] " #cond " (line " << __LINE__ << ")\n";             \
            return 1;                                                                \
        }                                                                            \
    } while (0)

int main() {
    static_assert(!std::is_copy_constructible<AudioTrack>::value,
                  "AudioTrack must not be copyable (double-delete hazard)");
    static_assert(std::is_move_constructible<AudioTrack>::value,
                  "AudioTrack must be movable into EditSession");
    std::cout << "[PASS] AudioTrack is move-only\n";

    EditSession session("Verify Session");
    AudioTrack track("Tone Track");
    CHECK(track.addClip("Tone Clip", 0.0, 5.0, "tone.wav") >= 0);
    session.addAudioTrack(std::move(track));
    CHECK(session.getAudioTracks().size() == 1);
    std::cout << "[PASS] Track moved into session exactly once\n";

    Transport transport;
    transport.processRealTimeAudio(session, 1.0);
    const double active_peak = transport.getLastPeak();
    CHECK(active_peak > 0.01);
    std::cout << "[PASS] Active clip renders audio, peak = " << active_peak << "\n";

    transport.processRealTimeAudio(session, 10.0);
    const double inactive_peak = transport.getLastPeak();
    CHECK(inactive_peak == 0.0);
    std::cout << "[PASS] No active clip renders silence, peak = " << inactive_peak << "\n";

    std::cout << "[PASS] All native DAW checks passed\n";
    return 0;
}
