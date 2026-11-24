#pragma once
#include "AudioTrack.h" // Assuming AudioTrack.h is in the same include directory
#include "MidiTrack.h"  // Assuming MidiTrack.h is in the same include directory
#include <vector>
#include <string>
#include <iostream> // For logging

/**
 * @class EditSession
 * @brief Represents a DAW project session, containing multiple tracks and settings.
 *
 * An EditSession encapsulates all data related to a single project,
 * including audio and MIDI tracks, session-level settings (like tempo, time signature - not yet implemented),
 * and overall project information.
 */
class EditSession {
public:
    /**
     * @brief Constructor for EditSession.
     * @param name The name of the session.
     */
    EditSession(const std::string& name);
    /**
     * @brief Destructor for EditSession.
     */
    ~EditSession();

    /**
     * @brief Sets the name of the session.
     * @param name The new name for the session.
     */
    void setSessionName(const std::string& name);
    /**
     * @brief Gets the current name of the session.
     * @return The name of the session.
     */
    std::string getSessionName() const; // Suitable for Transport's conceptual getName()

    /**
     * @brief Adds an audio track to the session.
     * @param track The AudioTrack object to add.
     */
    void addAudioTrack(AudioTrack&& track);
    /**
     * @brief Adds a MIDI track to the session.
     * @param track The MidiTrack object to add.
     */
    void addMidiTrack(MidiTrack&& track);

    /**
     * @brief Gets a reference to the vector of audio tracks in the session.
     * @return A reference to the vector of AudioTracks.
     */
    std::vector<AudioTrack>& getAudioTracks(); // Used by Transport
    /**
     * @brief Gets a constant reference to the vector of audio tracks in the session.
     * @return A constant reference to the vector of AudioTracks.
     */
    const std::vector<AudioTrack>& getAudioTracks() const;


    /**
     * @brief Gets a reference to the vector of MIDI tracks in the session.
     * @return A reference to the vector of MidiTracks.
     */
    std::vector<MidiTrack>& getMidiTracks(); // Used by Transport
    /**
     * @brief Gets a constant reference to the vector of MIDI tracks in the session.
     * @return A constant reference to the vector of MidiTracks.
     */
    const std::vector<MidiTrack>& getMidiTracks() const;


    /**
     * @brief Gets the total duration of the session (conceptual).
     * In a real DAW, this might be calculated based on the end of the last clip/event
     * or a user-defined session length.
     * @return The total duration in seconds (placeholder value).
     */
    double getTotalDuration() const; // Used by Transport, placeholder implementation

    // Could add methods to find, remove, or reorder tracks later

private:
    std::string sessionName;
    std::vector<AudioTrack> audioTracks;
    std::vector<MidiTrack> midiTracks;

    /**
     * @brief Logs a message specific to EditSession operations.
     * @param message The message string to log.
     */
    void log(const std::string& message) const {
        std::cout << "[EditSession: " << sessionName << " LOG]: " << message << std::endl;
    }

    /**
     * @brief Logs an error message specific to EditSession operations.
     * @param message The error message string to log.
     */
    void logError(const std::string& message) const {
        std::cerr << "[EditSession: " << sessionName << " ERROR]: " << message << std::endl;
    }
};
