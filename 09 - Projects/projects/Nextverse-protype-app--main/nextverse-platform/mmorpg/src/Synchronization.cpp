#include "Synchronization.h"
#include <vector> // Included again for clarity, though header might suffice for types

Synchronization::Synchronization() {
    initialize();
    log("Synchronization service initialized.");
}

Synchronization::~Synchronization() {
    cleanup();
    log("Synchronization service destroyed.");
}

void Synchronization::initialize() {
    log("Synchronization systems initialized (placeholder).");
}

void Synchronization::cleanup() {
    log("Synchronization systems cleaned up (placeholder).");
}

void Synchronization::synchronizePlayerStates(const std::vector<std::shared_ptr<PlayerData>>& players) {
    if (players.empty()) {
        // log("No player states to synchronize."); // Can be too verbose
        return;
    }
    log("Synchronizing states for " + std::to_string(players.size()) + " players (placeholder)...");
    for (const auto& playerPtr : players) {
        if (playerPtr) { // Check if the shared_ptr is not null
            if (playerPtr->isConnected) {
                // In a real system, this would involve serializing player data
                // (ID, position, health, status, etc.)
                // and sending it over the network.
                // For now, just log key details.
                // log("  Syncing " + playerPtr->username + " (ID: " + std::to_string(playerPtr->playerId) +
                //     ") at (" + std::to_string(playerPtr->x) + ", " + std::to_string(playerPtr->y) +
                //     ", " + std::to_string(playerPtr->z) + "). Status: Connected.");
            } else {
                // Log disconnected players if relevant for synchronization (e.g., to inform clients of disconnection)
                // log("  Player " + playerPtr->username + " (ID: " + std::to_string(playerPtr->playerId) +
                //     ") is disconnected. Syncing disconnect status.");
            }
        } else {
            logError("Encountered a null PlayerData pointer during synchronization.");
        }
    }
    // log("Player states synchronization cycle complete (placeholder)."); // Can be too verbose
}

void Synchronization::synchronizeNPCStates(const std::vector<NPCState>& npcStates) {
    if (npcStates.empty()) {
        // log("No NPC states to synchronize."); // Can be too verbose
        return;
    }
    log("Synchronizing " + std::to_string(npcStates.size()) + " NPC states (placeholder)...");
    for (const auto& npcState : npcStates) {
        // In a real system, this would involve serializing NPC data (ID, position, behavior, health)
        // and sending it.
        // log("  Syncing NPC " + npcState.name + " (ID: " + std::to_string(npcState.npcId) +
        //     ") at (" + std::to_string(npcState.x) + ", " + std::to_string(npcState.y) +
        //     ", " + std::to_string(npcState.z) + "). Behavior: " + npcState.currentBehavior);
    }
    // log("NPC states synchronization cycle complete (placeholder)."); // Can be too verbose
}

void Synchronization::log(const std::string& message) const {
    std::cout << "[Synchronization LOG]: " << message << std::endl;
}

void Synchronization::logError(const std::string& errorMessage) const {
    std::cerr << "[Synchronization ERROR]: " << errorMessage << std::endl;
}
