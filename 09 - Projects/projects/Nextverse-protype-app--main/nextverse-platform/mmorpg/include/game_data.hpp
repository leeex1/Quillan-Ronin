#ifndef GAME_DATA_HPP
#define GAME_DATA_HPP

#include <string>
#include <vector>
#include <map>

// --- Item, Recipe, Inventory ---
struct ItemDefinition {
    int item_id;
    std::string name;
    std::string description;
};

struct Ingredient {
    int item_id;
    int quantity;
};

struct RecipeDefinition {
    int recipe_id;
    std::string recipe_name;
    int result_item_id;
    int result_quantity;
    std::vector<Ingredient> ingredients;
    bool is_quantum_craft;

    RecipeDefinition() : recipe_id(0), result_item_id(0), result_quantity(0), is_quantum_craft(false) {}
    RecipeDefinition(int id, std::string name, int res_id, int res_qty, std::vector<Ingredient> ings, bool is_q = false)
        : recipe_id(id), recipe_name(std::move(name)), result_item_id(res_id), result_quantity(res_qty), ingredients(std::move(ings)), is_quantum_craft(is_q) {}
};

struct InventoryItem {
    int item_id;
    int quantity;
};

// --- NPC ---
struct NPCDefinition {
    int definition_id;
    std::string name_template;
    std::string description;
    std::string type;
    int max_health;
    int base_attack_power;
    int base_defense;

    NPCDefinition() : definition_id(0), name_template(""), description(""), type(""), max_health(100), base_attack_power(5), base_defense(2) {}
    NPCDefinition(int id, std::string nt, std::string d, std::string t, int mh, int atk, int def)
        : definition_id(id), name_template(std::move(nt)), description(std::move(d)),
          type(std::move(t)), max_health(mh), base_attack_power(atk), base_defense(def) {}
};

extern std::map<int, NPCDefinition> G_NPC_DEFINITIONS; // Forward declaration

struct NPCInstanceData {
    int instance_id;
    int definition_id;
    std::string unique_name;
    int current_health;
    double pos_x, pos_y, pos_z;
    std::string zone_id;
    int attack_power;
    int defense;

    NPCInstanceData() : instance_id(0), definition_id(0), unique_name(""), current_health(0), pos_x(0), pos_y(0), pos_z(0), zone_id(""), attack_power(0), defense(0) {}

    std::string getDisplayName() const {
        if (!unique_name.empty()) return unique_name;
        // Accessing G_NPC_DEFINITIONS here requires it to be defined or linked.
        // This is fine for a .hpp if G_NPC_DEFINITIONS is extern.
        auto def_it = G_NPC_DEFINITIONS.find(definition_id);
        if (def_it != G_NPC_DEFINITIONS.end()) return def_it->second.name_template;
        return "Unknown NPC";
    }
};

// --- Zone ---
struct ZoneDefinition {
    std::string zone_id_str;
    std::string friendly_name;
    std::string description;
};

// --- Quest ---
enum class QuestObjectiveType {
    COLLECT_ITEM,
    TALK_TO_NPC,
};

struct QuestObjectiveDefinition {
    QuestObjectiveType type;
    int target_id;
    int required_quantity;
    std::string description;
};

struct QuestRewardDefinition {
    int experience_points;
    int item_id_reward;
    int item_quantity_reward;
};

struct QuestPrerequisiteDefinition {
    int minimum_level;
    int required_completed_quest_id;
};

struct QuestDefinition {
    int quest_id;
    std::string title;
    std::string description_start;
    std::string description_inprogress;
    QuestPrerequisiteDefinition prerequisites;
    std::vector<QuestObjectiveDefinition> objectives;
    QuestRewardDefinition rewards;
    int starting_npc_definition_id;
    int completion_npc_definition_id;
};

enum class PlayerQuestStatus : int {
    NOT_ELIGIBLE = 0,
    AVAILABLE = 1,
    NOT_ACCEPTED = 2,
    ACCEPTED = 3,
    COMPLETED = 4,
};

// --- Global Data Stores (declared extern) ---
extern std::map<int, ItemDefinition> G_ITEM_DEFINITIONS;
extern std::map<int, RecipeDefinition> G_RECIPE_DEFINITIONS;
// G_NPC_DEFINITIONS already declared above
extern std::map<std::string, ZoneDefinition> G_ZONE_DEFINITIONS;
extern std::map<int, QuestDefinition> G_QUEST_DEFINITIONS;

// --- Guild Data Structures ---
enum class GuildRank : int {
    MEMBER = 0,
    OFFICER = 1,
    LEADER = 2
};

// Forward declaration for GuildMemberData for GuildData struct
struct GuildMemberData;

struct GuildData {
    int guild_id;
    std::string name;
    int leader_character_id;
    std::string leader_name; // Add this field
    std::string created_at_str;
    std::vector<GuildMemberData> members;
};

struct GuildMemberInfo {
    int guild_id;
    std::string guild_name;
    GuildRank rank;
};

struct GuildMemberData {
    int character_id;
    std::string character_name;
    GuildRank rank;
    std::string joined_at_str;
};


// --- Initialization Function ---
void initialize_game_data();

#endif // GAME_DATA_HPP
