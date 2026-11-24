#ifndef DATABASE_MANAGER_HPP
#define DATABASE_MANAGER_HPP

#include "db_connection.hpp"
#include "PlayerData.h" // Use PlayerData. Path should be correct via CMake include_directories("include")
#include <pqxx/pqxx>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <optional>
#include "../include/game_data.hpp" // For PlayerQuestStatus, InventoryItem, GuildMemberInfo etc.

struct PlayerQuestDBEntry { // This struct is used by methods returning quest data from DB
    int quest_id;
    PlayerQuestStatus status;
};

// CharacterData struct is removed. PlayerData from PlayerData.h is used.

class DatabaseManager {
public:
    DatabaseManager(DBConnection& db_conn) : db_conn_(db_conn) {}

    bool executeSQLFile(const std::string& filepath) {
        // ... (implementation as before)
        if (!db_conn_.isValid()) {
            std::cerr << "Database connection is not valid. Cannot execute SQL file." << std::endl;
            return false;
        }
        std::ifstream sql_file(filepath);
        if (!sql_file.is_open()) {
            std::cerr << "Error: Could not open SQL file: " << filepath << std::endl;
            return false;
        }
        std::stringstream buffer;
        buffer << sql_file.rdbuf();
        std::string sql_script = buffer.str();
        sql_file.close();

        if (sql_script.empty()) {
            std::cerr << "SQL file is empty: " << filepath << std::endl;
            return false;
        }

        try {
            pqxx::work W(*(db_conn_.getConnection()));
            W.exec(sql_script);
            W.commit();
            std::cout << "Successfully executed SQL script: " << filepath << std::endl;
            return true;
        } catch (const std::exception &e) {
            std::cerr << "Error executing SQL script " << filepath << ": " << e.what() << std::endl;
            return false;
        }
    }

    bool createPlayerAccount(const std::string& username, const std::string& password_hash, const std::string& email) {
        // ... (implementation as before)
        if (!db_conn_.isValid()) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            W.exec_params("INSERT INTO player_accounts (username, password_hash, email) VALUES ($1, $2, $3)",
                          username, password_hash, email);
            W.commit();
            std::cout << "Account created for: " << username << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in createPlayerAccount for " << username << ": " << e.what() << std::endl;
            return false;
        }
    }

    std::optional<int> verifyPlayerLogin(const std::string& username, const std::string& password_hash) {
        // ... (implementation as before)
        if (!db_conn_.isValid()) return std::nullopt;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("SELECT account_id FROM player_accounts WHERE username = $1 AND password_hash = $2",
                                           username, password_hash);
            W.commit();
            if (!R.empty()) {
                return R[0]["account_id"].as<int>();
            }
            return std::nullopt;
        } catch (const std::exception& e) {
            std::cerr << "Error in verifyPlayerLogin for " << username << ": " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    std::vector<PlayerData> listCharactersForAccount(int account_id) {
        std::vector<PlayerData> characters;
        if (!db_conn_.isValid()) return characters;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("SELECT character_id, character_name, level, experience, health, position_x, position_y, position_z, zone_id, attack_power, defense FROM characters WHERE account_id = $1", account_id);
            W.commit();
            for (const auto& row : R) {
                PlayerData char_data(row["character_id"].as<int>(), row["character_name"].as<std::string>());
                char_data.level = row["level"].as<int>();
                char_data.experience = row["experience"].as<int>();
                char_data.current_health = row["health"].as<int>();
                char_data.x = row["position_x"].as<float>();
                char_data.y = row["position_y"].as<float>();
                char_data.z = row["position_z"].as<float>();
                char_data.zone_id = row["zone_id"].as<std::string>();
                char_data.attack_power = row["attack_power"].as<int>();
                char_data.defense = row["defense"].as<int>();
                // PlayerData.max_health is defaulted. If schema stores it, load here.
                // PlayerData.current_inventory and current_quests would need separate loading or different PlayerData structure.
                characters.push_back(char_data);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in listCharactersForAccount for account_id " << account_id << ": " << e.what() << std::endl;
        }
        return characters;
    }

    std::optional<int> createCharacter(int account_id, const std::string& char_name, int initial_health, float x, float y, float z, const std::string& zone_id) {
        // ... (implementation as before, uses PlayerData defaults for non-provided stats)
        if (!db_conn_.isValid()) return std::nullopt;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            // These are the fields directly in the 'characters' table as per schema.sql for INSERT
            // Other PlayerData fields like level, experience, attack_power, defense will use their defaults
            // from PlayerData struct when a PlayerData object is constructed by getCharacterData.
            // Or, the DB schema itself has defaults for these columns.
            pqxx::result R = W.exec_params(
                "INSERT INTO characters (account_id, character_name, health, position_x, position_y, position_z, zone_id, level, experience, attack_power, defense) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING character_id",
                account_id, char_name, initial_health, x, y, z, zone_id,
                1, 0, 10, 5); // Default level, xp, atk, def for new char in DB
            W.commit();
            if (!R.empty()) {
                return R[0]["character_id"].as<int>();
            }
            return std::nullopt;
        } catch (const std::exception& e) {
            std::cerr << "Error in createCharacter for " << char_name << ": " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    std::optional<PlayerData> getCharacterData(int character_id) {
        if (!db_conn_.isValid()) return std::nullopt;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("SELECT character_id, character_name, level, experience, health, position_x, position_y, position_z, zone_id, attack_power, defense FROM characters WHERE character_id = $1", character_id);
            // W.commit(); // Not needed for read-only query if using read_transaction or if work is destructed.

            if (!R.empty()) {
                const auto& row = R[0];
                PlayerData char_data(row["character_id"].as<int>(), row["character_name"].as<std::string>());
                char_data.level = row["level"].as<int>();
                char_data.experience = row["experience"].as<int>();
                char_data.current_health = row["health"].as<int>();
                // char_data.max_health could be loaded if stored, or rely on PlayerData default.
                char_data.x = row["position_x"].as<float>();
                char_data.y = row["position_y"].as<float>();
                char_data.z = row["position_z"].as<float>();
                char_data.zone_id = row["zone_id"].as<std::string>();
                char_data.attack_power = row["attack_power"].as<int>();
                char_data.defense = row["defense"].as<int>();

                // The following lines will cause type errors if PlayerData.current_inventory/quests are std::vector<std::string>
                // For now, they are commented out. A full refactor would change PlayerData or add conversion.
                // auto inventory_items = getCharacterInventory(character_id); // returns std::vector<InventoryItem>
                // for(const auto& item : inventory_items) { char_data.current_inventory.push_back(item.name_or_id_or_description); }

                // auto quest_entries = getAllPlayerQuests(character_id); // returns std::vector<PlayerQuestDBEntry>
                // for(const auto& quest : quest_entries) { char_data.current_quests.push_back(std::to_string(quest.quest_id) + ":" + std::to_string(static_cast<int>(quest.status))); }

                // PlayerData does not have guild_info member by default.
                // char_data.guild_info = getCharacterGuildInfo(character_id);
                W.commit(); // Commit after all reads if not a read-only transaction
                return char_data;
            }
            W.commit(); // Or W.abort() if no result
            return std::nullopt;
        } catch (const std::exception& e) {
            std::cerr << "Error in getCharacterData for ID " << character_id << ": " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    bool saveCharacterData(const PlayerData& data) {
        if (!db_conn_.isValid()) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            W.exec_params("UPDATE characters SET level=$1, experience=$2, health=$3, position_x=$4, position_y=$5, position_z=$6, zone_id=$7, attack_power=$8, defense=$9, last_seen=CURRENT_TIMESTAMP WHERE character_id=$10",
                          data.level, data.experience, data.current_health, data.x, data.y, data.z, data.zone_id, data.attack_power, data.defense, data.playerId);
            W.commit();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in saveCharacterData for " << data.username << ": " << e.what() << std::endl;
            return false;
        }
    }

    // ... (rest of the methods like getPlayerQuestStatus, getAllPlayerQuests, etc. remain as they are,
    // as they mostly deal with character_id (int) and specific DB structures like PlayerQuestDBEntry, InventoryItem) ...
    // Ensure they are compatible or adapt if PlayerData needs to be passed to them or consume their results more directly.
    PlayerQuestStatus getPlayerQuestStatus(int character_id, int quest_id) {
        if (!db_conn_.isValid()) return PlayerQuestStatus::NOT_ELIGIBLE;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("SELECT status FROM character_quests WHERE character_id = $1 AND quest_id = $2",
                                           character_id, quest_id);
            W.commit();
            if (!R.empty()) {
                return static_cast<PlayerQuestStatus>(R[0]["status"].as<int>());
            }
            return PlayerQuestStatus::NOT_ACCEPTED;
        } catch (const std::exception& e) {
            std::cerr << "Error in getPlayerQuestStatus for char_id " << character_id << ", quest_id " << quest_id << ": " << e.what() << std::endl;
            return PlayerQuestStatus::NOT_ELIGIBLE;
        }
    }

    std::vector<PlayerQuestDBEntry> getAllPlayerQuests(int character_id) {
        std::vector<PlayerQuestDBEntry> quests;
        if (!db_conn_.isValid()) return quests;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("SELECT quest_id, status FROM character_quests WHERE character_id = $1", character_id);
            W.commit();
            for (const auto& row : R) {
                quests.push_back({row["quest_id"].as<int>(), static_cast<PlayerQuestStatus>(row["status"].as<int>())});
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in getAllPlayerQuests for char_id " << character_id << ": " << e.what() << std::endl;
        }
        return quests;
    }

    bool updatePlayerQuestStatus(int character_id, int quest_id, PlayerQuestStatus new_status) {
        if (!db_conn_.isValid()) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            std::string sql = R"(
                INSERT INTO character_quests (character_id, quest_id, status, last_updated)
                VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                ON CONFLICT (character_id, quest_id) DO UPDATE
                SET status = EXCLUDED.status, last_updated = CURRENT_TIMESTAMP;
            )";
            W.exec_params(sql, character_id, quest_id, static_cast<int>(new_status));
            W.commit();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in updatePlayerQuestStatus for char_id " << character_id << ", quest_id " << quest_id << ": " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<InventoryItem> getCharacterInventory(int character_id) {
        std::vector<InventoryItem> inventory;
        if (!db_conn_.isValid()) return inventory;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("SELECT item_id, quantity FROM character_inventories WHERE character_id = $1", character_id);
            W.commit();
            for (const auto& row : R) {
                inventory.push_back({row["item_id"].as<int>(), row["quantity"].as<int>()});
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in getCharacterInventory for char_id " << character_id << ": " << e.what() << std::endl;
        }
        return inventory;
    }

    bool addItemToInventory(int character_id, int item_id, int quantity) {
        if (!db_conn_.isValid() || quantity <= 0) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            std::string sql = R"(
                INSERT INTO character_inventories (character_id, item_id, quantity)
                VALUES ($1, $2, $3)
                ON CONFLICT (character_id, item_id) DO UPDATE
                SET quantity = character_inventories.quantity + EXCLUDED.quantity;
            )";
            W.exec_params(sql, character_id, item_id, quantity);
            W.commit();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in addItemToInventory for char_id " << character_id << ", item_id " << item_id << ": " << e.what() << std::endl;
            return false;
        }
    }

    bool removeItemFromInventory(int character_id, int item_id, int quantity_to_remove) {
        if (!db_conn_.isValid() || quantity_to_remove <= 0) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("SELECT quantity FROM character_inventories WHERE character_id = $1 AND item_id = $2",
                                           character_id, item_id);
            if (R.empty() || R[0]["quantity"].as<int>() < quantity_to_remove) {
                W.abort();
                std::cerr << "Item not found or insufficient quantity to remove for char_id " << character_id << ", item_id " << item_id << std::endl;
                return false;
            }
            int new_quantity = R[0]["quantity"].as<int>() - quantity_to_remove;
            if (new_quantity <= 0) {
                W.exec_params("DELETE FROM character_inventories WHERE character_id = $1 AND item_id = $2",
                              character_id, item_id);
            } else {
                W.exec_params("UPDATE character_inventories SET quantity = $1 WHERE character_id = $2 AND item_id = $3",
                              new_quantity, character_id, item_id);
            }
            W.commit();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in removeItemFromInventory for char_id " << character_id << ", item_id " << item_id << ": " << e.what() << std::endl;
            return false;
        }
    }

    bool checkAndConsumeIngredients(int character_id, const std::vector<Ingredient>& ingredients) {
        if (!db_conn_.isValid() || ingredients.empty()) return false;
        pqxx::work W(*(db_conn_.getConnection()));
        try {
            for (const auto& ingredient : ingredients) {
                pqxx::result R = W.exec_params(
                    "SELECT quantity FROM character_inventories WHERE character_id = $1 AND item_id = $2 FOR UPDATE",
                    character_id, ingredient.item_id);
                if (R.empty() || R[0]["quantity"].as<int>() < ingredient.quantity) {
                    W.abort();
                    std::cerr << "Insufficient ingredient: item_id " << ingredient.item_id << " for char_id " << character_id << std::endl;
                    return false;
                }
                int new_quantity = R[0]["quantity"].as<int>() - ingredient.quantity;
                if (new_quantity < 0) {
                     W.abort();
                     std::cerr << "Error: new quantity negative for item_id " << ingredient.item_id << std::endl;
                     return false;
                }
                if (new_quantity == 0) {
                    W.exec_params("DELETE FROM character_inventories WHERE character_id = $1 AND item_id = $2",
                                  character_id, ingredient.item_id);
                } else {
                    W.exec_params("UPDATE character_inventories SET quantity = $1 WHERE character_id = $2 AND item_id = $3",
                                  new_quantity, character_id, ingredient.item_id);
                }
            }
            W.commit();
            return true;
        } catch (const std::exception& e) {
            W.abort();
            std::cerr << "Error in checkAndConsumeIngredients for char_id " << character_id << ": " << e.what() << std::endl;
            return false;
        }
    }

    bool updateCharacterExperience(int character_id, int experience_gain) {
        if (!db_conn_.isValid() || experience_gain == 0) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            W.exec_params("UPDATE characters SET experience = experience + $1 WHERE character_id = $2",
                          experience_gain, character_id);
            W.commit();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in updateCharacterExperience for char_id " << character_id << ": " << e.what() << std::endl;
            return false;
        }
    }


    std::vector<NPCInstanceData> getAllNPCInstances() {
        // ... (implementation as before)
        std::vector<NPCInstanceData> instances;
        if (!db_conn_.isValid()) return instances;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params(
                "SELECT instance_id, definition_id, unique_name, current_health, pos_x, pos_y, pos_z, zone_id FROM npc_instances"
            );
            W.commit();
            for (const auto& row : R) {
                 NPCInstanceData npc_data;
                 npc_data.instance_id = row["instance_id"].as<int>();
                 npc_data.definition_id = row["definition_id"].as<int>();
                 npc_data.unique_name = row["unique_name"].is_null() ? "" : row["unique_name"].as<std::string>();
                 npc_data.current_health = row["current_health"].as<int>();
                 npc_data.pos_x = row["pos_x"].as<double>();
                 npc_data.pos_y = row["pos_y"].as<double>();
                 npc_data.pos_z = row["pos_z"].as<double>();
                 npc_data.zone_id = row["zone_id"].as<std::string>();
                instances.push_back(npc_data);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in getAllNPCInstances: " << e.what() << std::endl;
        }
        return instances;
    }

    std::optional<int> createNPCInstanceInDB(int definition_id, const std::string& unique_name, int health, double x, double y, double z, const std::string& zone_id) {
        // ... (implementation as before)
        if (!db_conn_.isValid()) return std::nullopt;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params(
                "INSERT INTO npc_instances (definition_id, unique_name, current_health, pos_x, pos_y, pos_z, zone_id) VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING instance_id",
                definition_id, unique_name, health, x, y, z, zone_id);
            W.commit();
            if (!R.empty()) {
                return R[0]["instance_id"].as<int>();
            }
            return std::nullopt;
        } catch (const std::exception& e) {
            std::cerr << "Error in createNPCInstanceInDB for def_id " << definition_id << ": " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    bool setNPCInstanceHealth(int npc_instance_id, int new_health) {
        // ... (implementation as before)
        if (!db_conn_.isValid()) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            W.exec_params("UPDATE npc_instances SET current_health = $1 WHERE instance_id = $2",
                          new_health, npc_instance_id);
            W.commit();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error in setNPCInstanceHealth for npc_id " << npc_instance_id << ": " << e.what() << std::endl;
            return false;
        }
    }

private:
    DBConnection& db_conn_;

public:
    // Guild Methods - these should be fine as they use character_id (int) mostly.
    // ... (guild method implementations as before) ...
    std::optional<int> createGuild(const std::string& guild_name, int leader_character_id ) {
        if (!db_conn_.isValid()) return std::nullopt;
        pqxx::work W(*(db_conn_.getConnection()));
        try {
            pqxx::result R_guild = W.exec_params(
                "INSERT INTO guilds (guild_name, leader_character_id) VALUES ($1, $2) RETURNING guild_id",
                guild_name, leader_character_id);

            if (R_guild.empty()) {
                W.abort(); return std::nullopt;
            }
            int new_guild_id = R_guild[0]["guild_id"].as<int>();

            W.exec_params(
                "INSERT INTO guild_members (guild_id, character_id, guild_rank) VALUES ($1, $2, $3)",
                new_guild_id, leader_character_id, static_cast<int>(GuildRank::LEADER));

            W.commit();
            return new_guild_id;
        } catch (const pqxx::unique_violation& e) {
            W.abort();
            std::cerr << "Error creating guild: Guild name '" << guild_name << "' likely already exists. Detail: " << e.what() << std::endl;
            return std::nullopt;
        }
        catch (const std::exception& e) {
            W.abort();
            std::cerr << "Error in createGuild for " << guild_name << ": " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    std::vector<GuildMemberData> getGuildMembers(int guild_id) {
        std::vector<GuildMemberData> members;
        if (!db_conn_.isValid()) return members;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params(
                "SELECT gm.character_id, c.character_name, gm.guild_rank, gm.joined_at "
                "FROM guild_members gm JOIN characters c ON gm.character_id = c.character_id "
                "WHERE gm.guild_id = $1 ORDER BY gm.guild_rank DESC, c.character_name ASC",
                guild_id);
            W.commit();
            for (const auto& row : R) {
                members.push_back({
                    row["character_id"].as<int>(),
                    row["character_name"].as<std::string>(),
                    static_cast<GuildRank>(row["guild_rank"].as<int>()),
                    row["joined_at"].as<std::string>()
                });
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in getGuildMembers for guild_id " << guild_id << ": " << e.what() << std::endl;
        }
        return members;
    }

    std::optional<GuildData> getGuildById(int guild_id, bool fetch_members = true) {
        if (!db_conn_.isValid()) return std::nullopt;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params(
                "SELECT g.guild_id, g.guild_name, g.leader_character_id, c.character_name as leader_name, g.created_at "
                "FROM guilds g JOIN characters c ON g.leader_character_id = c.character_id "
                "WHERE g.guild_id = $1", guild_id);
            W.commit();

            if (!R.empty()) {
                const auto& row = R[0];
                GuildData guild = {
                    row["guild_id"].as<int>(),
                    row["guild_name"].as<std::string>(),
                    row["leader_character_id"].as<int>(),
                    row["leader_name"].as<std::string>(),
                    row["created_at"].as<std::string>(),
                    {}
                };
                if (fetch_members) {
                    guild.members = getGuildMembers(guild_id);
                }
                return guild;
            }
            return std::nullopt;
        } catch (const std::exception& e) {
            std::cerr << "Error in getGuildById for guild_id " << guild_id << ": " << e.what() << std::endl;
            return std::nullopt;
        }
    }


    bool addGuildMember(int guild_id, int character_id , GuildRank rank) {
        if (!db_conn_.isValid()) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            W.exec_params(
                "INSERT INTO guild_members (guild_id, character_id, guild_rank) VALUES ($1, $2, $3)",
                guild_id, character_id, static_cast<int>(rank));
            W.commit();
            return true;
        } catch (const pqxx::unique_violation& e) {
             std::cerr << "Error adding guild member: Character " << character_id << " might already be in a guild. Detail: " << e.what() << std::endl;
            return false;
        }
        catch (const std::exception& e) {
            std::cerr << "Error in addGuildMember for char_id " << character_id << " to guild_id " << guild_id << ": " << e.what() << std::endl;
            return false;
        }
    }

    bool removeGuildMember(int guild_id, int character_id ) {
        if (!db_conn_.isValid()) return false;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params("DELETE FROM guild_members WHERE guild_id = $1 AND character_id = $2 RETURNING guild_member_id",
                                guild_id, character_id);
            W.commit();
            return R.affected_rows() > 0;
        } catch (const std::exception& e) {
            std::cerr << "Error in removeGuildMember for char_id " << character_id << " from guild_id " << guild_id << ": " << e.what() << std::endl;
            return false;
        }
    }

    std::optional<GuildMemberInfo> getCharacterGuildInfo(int character_id ) {
        if (!db_conn_.isValid()) return std::nullopt;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params(
                "SELECT gm.guild_id, g.guild_name, gm.guild_rank FROM guild_members gm "
                "JOIN guilds g ON gm.guild_id = g.guild_id "
                "WHERE gm.character_id = $1", character_id);
            W.commit();
            if (!R.empty()) {
                return GuildMemberInfo{
                    R[0]["guild_id"].as<int>(),
                    R[0]["guild_name"].as<std::string>(),
                    static_cast<GuildRank>(R[0]["guild_rank"].as<int>())
                };
            }
            return std::nullopt;
        } catch (const std::exception& e) {
            std::cerr << "Error in getCharacterGuildInfo for char_id " << character_id << ": " << e.what() << std::endl;
            return std::nullopt;
        }
    }

    std::vector<GuildData> getAllGuildsSummary() {
        std::vector<GuildData> guilds;
        if(!db_conn_.isValid()) return guilds;
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params(
                 "SELECT g.guild_id, g.guild_name, g.leader_character_id, c.character_name as leader_name, g.created_at "
                "FROM guilds g LEFT JOIN characters c ON g.leader_character_id = c.character_id ORDER BY g.guild_name ASC"
            );
            W.commit();
            for(const auto& row : R) {
                guilds.push_back({
                    row["guild_id"].as<int>(),
                    row["guild_name"].as<std::string>(),
                    row["leader_character_id"].is_null() ? 0 : row["leader_character_id"].as<int>(),
                    row["leader_name"].is_null() ? "Unknown" : row["leader_name"].as<std::string>(),
                    row["created_at"].as<std::string>(),
                    {}
                });
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in getAllGuildsSummary: " << e.what() << std::endl;
        }
        return guilds;
    }


    bool updateGuildMemberRank(int guild_id, int character_id , GuildRank new_rank) {
        if (!db_conn_.isValid()) return false;
        if (new_rank == GuildRank::LEADER) {
             std::cerr << "Cannot promote to LEADER using updateGuildMemberRank. Use a specific change leader function." << std::endl;
             return false;
        }
        try {
            pqxx::work W(*(db_conn_.getConnection()));
            pqxx::result R = W.exec_params(
                "UPDATE guild_members SET guild_rank = $1 WHERE guild_id = $2 AND character_id = $3 AND guild_rank != $4",
                static_cast<int>(new_rank), guild_id, character_id, static_cast<int>(GuildRank::LEADER)
            );
            W.commit();
            return R.affected_rows() > 0;
        } catch (const std::exception& e) {
            std::cerr << "Error in updateGuildMemberRank for char_id " << character_id << " in guild " << guild_id << ": " << e.what() << std::endl;
            return false;
        }
    }

};

#endif // DATABASE_MANAGER_HPP
