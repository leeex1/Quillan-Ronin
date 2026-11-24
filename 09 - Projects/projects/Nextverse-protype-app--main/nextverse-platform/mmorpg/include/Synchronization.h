#ifndef SYNCHRONIZATION_H
#define SYNCHRONIZATION_H

#include "PlayerData.h" // For PlayerData
#include "NPCBehavior.h" // For NPCState (if synchronizing NPC states)
#include <vector>
#include <string>
#include <memory>   // For std::shared_ptr
#include <iostream> // For logging


/**
 * @class Synchronization
 * @brief Handles synchronization of game state, like player positions and NPC states.
 *
 * This class provides methods to package and (conceptually) transmit
 * game state updates across the network or between server threads.
 */
class Synchronization {
public:
    Synchronization();
    ~Synchronization();

    /**
     * @brief Synchronizes player positions and other relevant player data.
     * Placeholder for actual network send/receive logic. In a real system, this would
     * involve serializing data and sending it to clients or a central server.
     * @param players A list of shared_ptr to PlayerData objects to synchronize.
     */
    void synchronizePlayerStates(const std::vector<std::shared_ptr<PlayerData>>& players);

    /**
     * @brief Synchronizes NPC states.
     * Placeholder for actual network send/receive logic.
     * @param npcStates A list of NPCState objects to synchronize.
     */
    void synchronizeNPCStates(const std::vector<NPCState>& npcStates);


    // Basic Logging
    void log(const std::string& message) const;
    void logError(const std::string& errorMessage) const;

private:
    void initialize();
    void cleanup();
};

#endif // SYNCHRONIZATION_H
