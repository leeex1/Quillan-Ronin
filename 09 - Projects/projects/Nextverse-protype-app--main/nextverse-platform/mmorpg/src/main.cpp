#include "../include/mmorpg_server.hpp"
#include "../include/quantum_vm_mock.hpp"
#include "../include/game_data.hpp" // For initialize_game_data and G_NPC_DEFINITIONS
#include "db_connection.hpp"
#include "database_manager.hpp"
#include <iostream>
#include <cstdlib> // For std::getenv, srand
#include <ctime>   // For time

// Global LogMessage for main.cpp, or would be part of a shared utility
// Declaration or full definition before first use.
void LogMessage(const std::string& message);

int main() {
    // Initialize random seed for gameplay elements like NPC ambient speech
    srand(time(0));

    // Initialize static game data (items, recipes, NPCs, zones, quests)
    initialize_game_data();
    LogMessage("Static game data initialized.");

    // Database Connection Setup
    DBConnection* db_conn = nullptr;
    DatabaseManager* db_manager = nullptr;

    const char* conn_str_env = std::getenv("DB_CONNECTION_STRING");
    std::string db_conn_string = conn_str_env ? conn_str_env : "postgresql://testuser:testpass@localhost:5432/testdb"; // Default fallback

    std::cout << "Attempting to connect to database with connection string: " << db_conn_string << std::endl;

    try {
        db_conn = new DBConnection(db_conn_string);
        if (db_conn->isValid()) {
            std::cout << "DBConnection object created and connection is valid." << std::endl;
            db_manager = new DatabaseManager(*db_conn);

            if (db_manager->executeSQLFile("../sql/schema.sql")) {
                std::cout << "Database schema initialized (or verified)." << std::endl;
            } else {
                std::cerr << "Could not initialize database schema. Server might not function correctly with DB operations." << std::endl;
                // Depending on policy, one might choose to exit here.
            }

            // One-time NPC population if DB is empty
            std::vector<NPCInstanceData> current_npcs = db_manager->getAllNPCInstances();
            if (current_npcs.empty()) {
                std::cout << "No NPCs found in DB, populating with examples..." << std::endl;
                if(G_NPC_DEFINITIONS.count(1001)) db_manager->createNPCInstanceInDB(1001, "Guard Thompson", G_NPC_DEFINITIONS[1001].max_health, 10.0, 5.0, 0.0, "starting_zone_town_square");
                if(G_NPC_DEFINITIONS.count(1002)) db_manager->createNPCInstanceInDB(1002, "Mysterious Merchant", G_NPC_DEFINITIONS[1002].max_health, 15.0, -5.0, 0.0, "starting_zone_market");
                if(G_NPC_DEFINITIONS.count(2001)) {
                    db_manager->createNPCInstanceInDB(2001, "Forest Goblin Alpha", G_NPC_DEFINITIONS[2001].max_health, 100.0, 100.0, 0.0, "starting_zone_forest_edge");
                    db_manager->createNPCInstanceInDB(2001, "Forest Goblin Bravo", G_NPC_DEFINITIONS[2001].max_health, 105.0, 102.0, 0.0, "starting_zone_forest_edge");
                }
            }
        } else {
             std::cerr << "DBConnection failed to become valid." << std::endl;
             delete db_conn; // Clean up allocated DBConnection
             db_conn = nullptr;
             // Server cannot run without a DB manager in this design.
             std::cerr << "Server cannot start without a valid database connection. Exiting." << std::endl;
             return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during database setup: " << e.what() << std::endl;
        delete db_manager;
        delete db_conn;
        std::cerr << "Server cannot start due to database setup exception. Exiting." << std::endl;
        return 1;
    }

    if (!db_manager) { // Should have been caught by exceptions, but as a safeguard.
        std::cerr << "DatabaseManager not initialized. Server cannot start." << std::endl;
        delete db_conn;
        return 1;
    }

    std::cout << "Database setup finished. Starting MMORPG server..." << std::endl;

    try {
        QuantumVM qvm_mock;
        MMORPG_Server server(qvm_mock, "tcp://*:5555", "tcp://*:5556", *db_manager);
        server.run();
    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception in MMORPG_Server run: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown unhandled exception in MMORPG_Server run." << std::endl;
    }

    // Clean up resources
    std::cout << "Server shutting down. Cleaning up resources..." << std::endl;
    delete db_manager;
    delete db_conn;
    std::cout << "Cleanup finished. Exiting." << std::endl;

    return 0;
}

// Full definition of LogMessage
void LogMessage(const std::string& message) {
    std::cout << "[MAIN_LOG] " << message << std::endl;
}
