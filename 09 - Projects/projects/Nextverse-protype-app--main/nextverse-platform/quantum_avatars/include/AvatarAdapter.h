#ifndef AVATAR_ADAPTER_H
#define AVATAR_ADAPTER_H

#include <string>
#include <vector>
#include <iostream> // For basic logging
#include <stdexcept> // For exceptions
#include <utility> // For std::move

// Forward declaration for a potential Avatar class if needed for context
// class QuantumAvatar;

/**
 * @struct UserAction
 * @brief Represents a user action that might influence avatar behavior.
 */
struct UserAction {
    std::string type;       ///< Type of action, e.g., "interaction", "combat_move", "environment_change".
    std::string details;    ///< Specifics of the action, e.g., "attack_target_A", "friendly_greeting".
    // Potentially add player ID, target ID, etc. as needed.
    // int actorId;
    // int targetId;

    /**
     * @brief Default constructor.
     */
    UserAction() = default;

    /**
     * @brief Constructor for UserAction.
     * @param _type The type of the action.
     * @param _details The details specific to this action.
     */
    UserAction(std::string _type, std::string _details)
        : type(std::move(_type)), details(std::move(_details)) {}
};

/**
 * @class AvatarAdapter
 * @brief Adapts avatar behavior based on user actions and internal state.
 * Implements logic for adaptive behaviors, error checking, and logging.
 * This class serves as a bridge between external stimuli (like user actions)
 * and the avatar's behavioral responses.
 */
class AvatarAdapter {
public:
    /**
     * @brief Constructor for AvatarAdapter.
     * @param avatarId The ID of the avatar this adapter is for.
     */
    AvatarAdapter(const std::string& avatarId);
    /**
     * @brief Destructor for AvatarAdapter.
     */
    ~AvatarAdapter();

    /**
     * @brief Adapts the avatar's behavior based on a user action.
     * This method processes a given UserAction, determines an appropriate
     * behavioral response, updates the avatar's internal state, and logs the change.
     * @param action The UserAction that occurred.
     * @return A string describing the adapted behavior or status.
     *         Returns an error message if adaptation fails.
     */
    std::string adaptBehavior(const UserAction& action);

    /**
     * @brief Sets a new behavior state for the avatar.
     * @param newBehavior The new behavior string (e.g., "aggressive", "defensive", "exploring").
     *                    Cannot be empty.
     * @return True if behavior was set successfully, false otherwise (e.g., empty newBehavior).
     */
    bool setBehaviorState(const std::string& newBehavior);

    /**
     * @brief Gets the current behavior state of the avatar.
     * @return The current behavior string.
     */
    std::string getCurrentBehaviorState() const;

    // Basic Logging
    /**
     * @brief Logs a general message related to this avatar adapter.
     * @param message The message to log.
     */
    void log(const std::string& message) const;
    /**
     * @brief Logs an error message related to this avatar adapter.
     * @param errorMessage The error message to log.
     */
    void logError(const std::string& errorMessage) const;

private:
    std::string avatarId_; ///< The ID of the avatar this adapter manages.
    std::string currentBehaviorState_ = "idle"; ///< Default behavior state.

    /**
     * @brief Internal logic to determine a new behavior based on an action.
     * This method encapsulates the decision-making process for behavior changes.
     * @param action The user action to process.
     * @return A string representing the suggested new behavior.
     */
    std::string determineNewBehavior(const UserAction& action);
};

#endif // AVATAR_ADAPTER_H
