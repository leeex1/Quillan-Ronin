#ifndef PLAYER_MANAGER_H
#define PLAYER_MANAGER_H

#include "PlayerData.h" // Include the PlayerData struct
#include <vector>
#include <string>
#include <memory> // For std::shared_ptr or std::unique_ptr
#include <iostream> // For logging
#include <map>      // To store players

/**
 * @class PlayerManager
 * @brief Manages player connections, data, and basic interactions.
 */
class PlayerManager {
public:
    PlayerManager();
    ~PlayerManager();

    /**
     * @brief Handles a new player connecting to the server.
     * @param playerId A unique ID for the player.
     * @param username The username of the connecting player.
     * @return True if the player was successfully added, false otherwise (e.g., ID conflict).
     */
    bool playerConnect(int playerId, const std::string& username);

    /**
     * @brief Handles a player disconnecting from the server.
     * @param playerId The ID of the player who disconnected.
     * @return True if the player was found and marked as disconnected, false otherwise.
     */
    bool playerDisconnect(int playerId);

    /**
     * @brief Retrieves player data for a given player ID.
     * @param playerId The ID of the player.
     * @return A shared_ptr to PlayerData if found, nullptr otherwise.
     */
    std::shared_ptr<PlayerData> getPlayerData(int playerId) const;

    /**
     * @brief Updates a player's position.
     * @param playerId The ID of the player.
     * @param x The new x-coordinate.
     * @param y The new y-coordinate.
     * @param z The new z-coordinate.
     * @return True if player found and updated, false otherwise.
     */
    bool updatePlayerPosition(int playerId, float x, float y, float z);

    // Basic Logging
    void log(const std::string& message) const;
    void logError(const std::string& errorMessage) const;

private:
    std::map<int, std::shared_ptr<PlayerData>> players_; // Map player ID to PlayerData
    // int nextPlayerId_ = 1; // Simple way to generate IDs if not provided externally
    // Commented out nextPlayerId_ as playerConnect takes playerId as a parameter.
    // If IDs were to be generated internally, this would be needed.

    void initialize();
    void cleanup();
};

#endif // PLAYER_MANAGER_H
