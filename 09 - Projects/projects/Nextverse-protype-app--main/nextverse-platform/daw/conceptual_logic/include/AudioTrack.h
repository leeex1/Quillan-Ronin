#pragma once
#include <string>
#include <vector>
#include <iostream> // For logging in implementations, though not strictly needed in .h
#include <utility>  // For std::move
#include "EventTypes.h" // Include the actual definition

// Forward declaration
class AudioEffect;

/**
 * @struct AudioClip
 * @brief Represents a segment of audio data within an AudioTrack.
 *
 * An AudioClip has properties like name, start time, duration, file path,
 * and a unique ID within its track.
 */
struct AudioClip {
    int id;                     ///< Unique identifier for the clip within the track.
    std::string name;           ///< Name of the audio clip (e.g., "Verse 1 Vocals").
    double startTime;           ///< Start time of the clip on the track timeline, in seconds.
    double duration;            ///< Duration of the clip, in seconds.
    std::string filePath;       ///< Path to the audio file for this clip.

    /**
     * @brief Constructor for AudioClip.
     * @param _id Unique ID.
     * @param _name Name of the clip.
     * @param _startTime Start time in seconds.
     * @param _duration Duration in seconds.
     * @param _filePath Path to the audio file.
     */
    AudioClip(int _id, std::string _name, double _startTime, double _duration, std::string _filePath)
        : id(_id), name(std::move(_name)), startTime(_startTime), duration(_duration), filePath(std::move(_filePath)) {}
};

/**
 * @class AudioTrack
 * @brief Represents a track in the DAW that holds audio data.
 *
 * AudioTracks can contain multiple audio clips, each representing a segment
 * of audio. The track itself has properties like volume, pan, mute, and solo states.
 * It can also host a chain of audio effects.
 */
class AudioTrack {
public:
    /**
     * @brief Constructor for AudioTrack.
     * @param name The initial name of the track.
     */
    AudioTrack(const std::string& name);
    /**
     * @brief Destructor for AudioTrack.
     * Manages the cleanup of associated effects.
     */
    ~AudioTrack();

    // effects_ holds owning raw pointers: copying would double-delete them.
    // Tracks must be transferred, never copied, into an EditSession.
    AudioTrack(const AudioTrack&) = delete;
    AudioTrack& operator=(const AudioTrack&) = delete;
    AudioTrack(AudioTrack&&) = default;
    AudioTrack& operator=(AudioTrack&&) = default;

    /**
     * @brief Sets the name of the audio track.
     * @param name The new name for the track.
     */
    void setName(const std::string& name);
    /**
     * @brief Gets the current name of the audio track.
     * @return The name of the track.
     */
    std::string getName() const;

    // --- Existing Event-related methods (if they are to be kept separate from clips) ---
    /**
     * @brief Adds an audio event to the track.
     * @param event The AudioEvent to add.
     */
    void addEvent(const AudioEvent& event);
    /**
     * @brief Gets all audio events on this track.
     * @return A constant reference to the vector of AudioEvents.
     */
    const std::vector<AudioEvent>& getEvents() const;

    // --- AudioClip CRUD Operations ---
    /**
     * @brief Adds a new audio clip to the track.
     * @param clipName The name for the new clip.
     * @param startTime The start time of the clip on the track.
     * @param duration The duration of the clip.
     * @param filePath The path to the audio file for the clip.
     * @return The ID of the newly created clip, or -1 on failure.
     */
    int addClip(const std::string& clipName, double startTime, double duration, const std::string& filePath);

    /**
     * @brief Retrieves an audio clip by its ID.
     * @param clipId The ID of the clip to retrieve.
     * @return The AudioClip object. Throws std::out_of_range if not found.
     */
    AudioClip getClip(int clipId) const;

    /**
     * @brief Retrieves all audio clips from the track.
     * @return A constant reference to the vector of AudioClips.
     */
    const std::vector<AudioClip>& getAllClips() const;

    /**
     * @brief Updates an existing audio clip's properties.
     * @param clipId The ID of the clip to update.
     * @param newName The new name for the clip.
     * @param newStartTime The new start time for the clip.
     * @param newDuration The new duration for the clip.
     * @param newFilePath The new file path for the clip.
     * @return True if the update was successful, false if the clip was not found.
     */
    bool updateClip(int clipId, const std::string& newName, double newStartTime, double newDuration, const std::string& newFilePath);

    /**
     * @brief Removes an audio clip from the track by its ID.
     * @param clipId The ID of the clip to remove.
     * @return True if the clip was found and removed, false otherwise.
     */
    bool removeClip(int clipId);

    // --- Audio Effects ---
    /**
     * @brief Adds an audio effect to the track's effect chain.
     * The track takes ownership of the effect pointer.
     * @param effect Pointer to an AudioEffect object.
     */
    void addEffect(AudioEffect* effect);

    /**
     * @brief Retrieves the list of audio effects applied to this track.
     * @return A constant reference to the vector of AudioEffect pointers.
     */
    const std::vector<AudioEffect*>& getEffects() const;


    // --- Track Properties ---
    float volume = 1.0f;            ///< Volume of the track, typically 0.0 to 1.0 (or higher).
    float pan = 0.0f;               ///< Panning of the track, -1.0 (left) to 1.0 (right).
    bool isMuted = false;           ///< Mute state of the track.
    bool isSoloed = false;          ///< Solo state of the track.

private:
    std::string trackName;
    std::vector<AudioEvent> audioEvents; ///< Existing audio events.
    std::vector<AudioClip> clips_;       ///< Stores audio clips for this track.
    int nextClipId_ = 1;                 ///< Simple counter for generating unique clip IDs within this track.
    std::vector<AudioEffect*> effects_;  ///< Chain of audio effects for this track.

    void log(const std::string& message) const {
        std::cout << "[AudioTrack: " << trackName << " LOG]: " << message << std::endl;
    }
    void logError(const std::string& message) const {
        std::cerr << "[AudioTrack: " << trackName << " ERROR]: " << message << std::endl;
    }
};
