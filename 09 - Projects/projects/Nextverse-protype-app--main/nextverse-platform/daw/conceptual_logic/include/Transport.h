#pragma once
#include <string>   // For std::to_string if used in logging
#include <iostream> // For std::cout and std::cerr for logging

// Forward declaration
class EditSession;

/**
 * @class Transport
 * @brief Controls playback, recording, and seeking within an EditSession.
 *
 * The Transport class manages the state of playback (playing, stopped, recording)
 * and the current playback position within an EditSession. It simulates
 * real-time audio processing during playback.
 */
class Transport {
public:
    /**
     * @brief Constructor for Transport.
     */
    Transport();
    /**
     * @brief Destructor for Transport.
     */
    ~Transport();

    /**
     * @brief Starts or resumes playback of the given EditSession.
     * @param session The EditSession to play.
     */
    void play(EditSession& session);
    /**
     * @brief Stops playback or recording for the given EditSession.
     * @param session The EditSession to stop.
     */
    void stop(EditSession& session); // Added session parameter for context
    /**
     * @brief Toggles the recording state for the given EditSession.
     * This is a conceptual placeholder.
     * @param session The EditSession context for recording.
     */
    void record(EditSession& session); // Added session parameter for context

    /**
     * @brief Seeks to a specific time in the EditSession.
     * @param timeInSeconds The time to seek to, in seconds.
     * @param session The EditSession context for seeking.
     */
    void seek(double timeInSeconds, EditSession& session); // Added session parameter

    /**
     * @brief Gets the current playback or recording time.
     * @return The current time in seconds.
     */
    double getCurrentPosition() const; // Renamed from getCurrentTime for clarity
    /**
     * @brief Checks if playback is currently active.
     * @return True if playing, false otherwise.
     */
    bool isPlaying() const; // Renamed from getIsPlaying
    /**
     * @brief Checks if recording is currently active.
     * @return True if recording, false otherwise.
     */
    bool isRecording() const; // Renamed from getIsRecording

    /**
     * @brief Conceptually processes a slice of real-time audio for the session.
     * This method would be called repeatedly during playback to handle audio data
     * from tracks, apply effects, and mix them.
     * @param session The EditSession being processed.
     * @param currentTime The current playback time for this processing slice.
     */
    void processRealTimeAudio(EditSession& session, double currentTime);

    /**
     * @brief Peak absolute amplitude of the most recently rendered block.
     * Used by native verification to prove the engine is not producing silence.
     */
    double getLastPeak() const { return lastPeak_; }

private:
    double currentPosition_ = 0.0;   ///< Current playback/seek position in seconds.
    bool isPlaying_ = false;         ///< True if playback is active.
    bool isRecording_ = false;       ///< True if recording is active.
    double lastPeak_ = 0.0;          ///< Peak amplitude of the last rendered block.

    /**
     * @brief Logs a message specific to Transport operations.
     * @param message The message string to log.
     */
    void log(const std::string& message) const {
        std::cout << "[Transport LOG]: " << message << std::endl;
    }

    /**
     * @brief Logs an error message specific to Transport operations.
     * @param message The error message string to log.
     */
    void logError(const std::string& message) const {
        std::cerr << "[Transport ERROR]: " << message << std::endl;
    }
};
