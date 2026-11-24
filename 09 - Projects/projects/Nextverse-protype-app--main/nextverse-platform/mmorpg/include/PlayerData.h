#ifndef PLAYER_DATA_H
#define PLAYER_DATA_H

#include <string>
#include <vector>
#include <chrono>  // For timestamps
#include <utility> // For std::move
#include "game_data.hpp" // For InventoryItem and PlayerQuestStatus

/**
 * @struct PlayerQuestInfo
 * @brief Represents the status and key information of a quest for a player.
 * This mirrors PlayerQuestDBEntry but is defined here for clarity if PlayerData is a core DTO.
 */
struct PlayerQuestInfo {
    int questId;                        ///< Unique identifier for the quest.
    PlayerQuestStatus status;           ///< Current status of the quest for the player.
    // std::string questName;           // Optional: Can be added if frequently needed, otherwise fetched via G_QUEST_DEFINITIONS
    // std::string currentObjective;    // Optional: More detailed progress tracking

    PlayerQuestInfo(int qId, PlayerQuestStatus stat) : questId(qId), status(stat) {}
};


/**
 * @struct PlayerData
 * @brief Holds basic information and state for a player/character.
 * This structure aggregates all essential data for a player within the game world.
 */
struct PlayerData {
    // Identification & Connection
    int playerId;                       ///< Unique identifier for the player.
    std::string username;               ///< Player's username.
    bool isConnected = false;           ///< Current connection status.
    std::chrono::system_clock::time_point lastSeen; ///< Timestamp of last activity.

    // Position
    float x = 0.0f;                     ///< Player's x-coordinate in the game world.
    float y = 0.0f;                     ///< Player's y-coordinate in the game world.
    float z = 0.0f;                     ///< Player's z-coordinate in the game world.
    std::string zone_id = "starting_zone"; ///< ID of the zone the player is currently in.

    // Combat & Stats
    int max_health = 100;               ///< Player's maximum health points.
    int current_health = 100;           ///< Player's current health points.
    int level = 1;                      ///< Player's current level.
    int experience = 0;                 ///< Player's current experience points.
    int attack_power = 10;              ///< Player's base attack power.
    int defense = 5;                    ///< Player's base defense value.

    // Inventory & Quests
    std::vector<InventoryItem> current_inventory; ///< Player's current inventory.
    std::vector<PlayerQuestInfo> current_quests;    ///< Player's current active/completed quests.


    /**
     * @brief Default constructor for PlayerData.
     * Initializes members with default values.
     */
    PlayerData()
        : playerId(0), username("Unknown"), isConnected(false),
          x(0.0f), y(0.0f), z(0.0f), zone_id("default_zone"),
          max_health(100), current_health(100), level(1), experience(0),
          attack_power(10), defense(5) {
        lastSeen = std::chrono::system_clock::now();
    }

    /**
     * @brief Constructor for PlayerData.
     * @param id Unique ID for the player.
     * @param _username Username of the player.
     */
    PlayerData(int id, std::string _username)
        : playerId(id), username(std::move(_username)), isConnected(false),
          x(0.0f), y(0.0f), z(0.0f), zone_id("starting_zone"),
          max_health(100), current_health(100), level(1), experience(0),
          attack_power(10), defense(5) {
        lastSeen = std::chrono::system_clock::now();
        current_health = max_health;
    }
};

#endif // PLAYER_DATA_H
