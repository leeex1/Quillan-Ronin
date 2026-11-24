#include "PlayerManager.h" // Assuming PlayerManager.h is in include path
#include <utility> // For std::move
#include <chrono>  // For std::chrono::system_clock (already in PlayerData.h but good for clarity if used directly here)

PlayerManager::PlayerManager() {
    initialize();
    log("PlayerManager initialized.");
}

PlayerManager::~PlayerManager() {
    cleanup();
    log("PlayerManager destroyed.");
}

void PlayerManager::initialize() {
    log("PlayerManager systems initialized (placeholder).");
}

void PlayerManager::cleanup() {
    log("PlayerManager systems cleaned up (placeholder).");
    players_.clear();
}

bool PlayerManager::playerConnect(int playerId, const std::string& username) {
    if (players_.count(playerId)) {
        auto& existingPlayer = players_[playerId];
        if (existingPlayer->isConnected) {
            logError("Player ID " + std::to_string(playerId) + " (" + existingPlayer->username + ") already connected.");
            return false;
        } else {
            // Player is reconnecting
            existingPlayer->username = username; // Update username in case it changed
            existingPlayer->isConnected = true;
            existingPlayer->lastSeen = std::chrono::system_clock::now();
            log("Player reconnected: " + username + " (ID: " + std::to_string(playerId) + ")");
            return true;
        }
    }
    auto newPlayer = std::make_shared<PlayerData>(playerId, username);
    // newPlayer->isConnected = true; // Constructor now handles lastSeen, isConnected can be set here explicitly
    // newPlayer->lastSeen = std::chrono::system_clock::now(); // Constructor handles this
    newPlayer->isConnected = true; // Explicitly set connected state
    players_[playerId] = newPlayer;
    log("Player connected: " + username + " (ID: " + std::to_string(playerId) + ")");
    return true;
}

bool PlayerManager::playerDisconnect(int playerId) {
    auto it = players_.find(playerId);
    if (it != players_.end()) {
        if (it->second->isConnected) {
            it->second->isConnected = false;
            it->second->lastSeen = std::chrono::system_clock::now();
            log("Player disconnected: " + it->second->username + " (ID: " + std::to_string(playerId) + ")");
            // Optionally remove from map or keep for history:
            // players_.erase(it); // If removing, careful with shared_ptr elsewhere
        } else {
            log("Player " + it->second->username + " (ID: " + std::to_string(playerId) + ") was already marked as disconnected.");
        }
        return true;
    }
    logError("PlayerDisconnect: Player ID " + std::to_string(playerId) + " not found.");
    return false;
}

std::shared_ptr<PlayerData> PlayerManager::getPlayerData(int playerId) const {
    auto it = players_.find(playerId);
    if (it != players_.end()) {
        return it->second;
    }
    // logError("GetPlayerData: Player ID " + std::to_string(playerId) + " not found.");
    // Logging for a failed get might be too verbose depending on usage, could be optional.
    return nullptr;
}

bool PlayerManager::updatePlayerPosition(int playerId, float x, float y, float z) {
    auto it = players_.find(playerId);
    if (it != players_.end()) {
        if (it->second->isConnected) {
            it->second->x = x;
            it->second->y = y;
            it->second->z = z;
            it->second->lastSeen = std::chrono::system_clock::now();
            // Log for position update might be too verbose for frequent updates.
            // log("Player " + it->second->username + " position updated to (" +
            //     std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")");
            return true;
        } else {
            logError("UpdatePlayerPosition: Player ID " + std::to_string(playerId) + " (" + it->second->username + ") is not connected.");
            return false;
        }
    }
    logError("UpdatePlayerPosition: Player ID " + std::to_string(playerId) + " not found.");
    return false;
}

void PlayerManager::log(const std::string& message) const {
    std::cout << "[PlayerManager LOG]: " << message << std::endl;
}

void PlayerManager::logError(const std::string& errorMessage) const {
    std::cerr << "[PlayerManager ERROR]: " << errorMessage << std::endl;
}
