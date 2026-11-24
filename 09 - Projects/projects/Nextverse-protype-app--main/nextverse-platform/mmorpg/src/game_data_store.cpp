#include "../include/game_data.hpp"

// Define Global Data Stores
std::map<int, ItemDefinition> G_ITEM_DEFINITIONS;
std::map<int, RecipeDefinition> G_RECIPE_DEFINITIONS;
std::map<int, NPCDefinition> G_NPC_DEFINITIONS;
std::map<std::string, ZoneDefinition> G_ZONE_DEFINITIONS;
std::map<int, QuestDefinition> G_QUEST_DEFINITIONS;

void initialize_game_data() {
    // Placeholder Items
    G_ITEM_DEFINITIONS[1] = {1, "Iron Ore", "A chunk of raw iron ore."};
    G_ITEM_DEFINITIONS[2] = {2, "Wood Log", "A sturdy log of wood."};
    G_ITEM_DEFINITIONS[3] = {3, "Iron Ingot", "A bar of refined iron."};
    G_ITEM_DEFINITIONS[4] = {4, "Wooden Handle", "A simple wooden handle."};
    G_ITEM_DEFINITIONS[5] = {5, "Basic Sword", "A crude but effective sword."};
    G_ITEM_DEFINITIONS[6] = {6, "Health Potion", "Restores a bit of health."};
    G_ITEM_DEFINITIONS[7] = {7, "Quantum Shard", "A fragment resonating with unstable energy."};
    G_ITEM_DEFINITIONS[8] = {8, "Unstable Matrix", "A containment unit struggling to hold chaotic forces."};
    G_ITEM_DEFINITIONS[9] = {9, "Proto-Blade", "A basic blade hilt, ready for quantum infusion."};
    G_ITEM_DEFINITIONS[10] = {10, "Flux-Edged Blade", "A Proto-Blade with minor quantum instability."};
    G_ITEM_DEFINITIONS[11] = {11, "Damage-Infused Blade", "A Proto-Blade enhanced with bonus damage."};
    G_ITEM_DEFINITIONS[12] = {12, "Critical Strike Blade", "A Proto-Blade honed for critical hits."};

    // Placeholder Recipes
    G_RECIPE_DEFINITIONS[101] = {101, "Smelt Iron Ingot", 3, 1, {{1, 2}}, false};
    G_RECIPE_DEFINITIONS[102] = {102, "Craft Wooden Handle", 4, 1, {{2, 1}}, false};
    G_RECIPE_DEFINITIONS[103] = {103, "Craft Basic Sword", 5, 1, {{3, 2}, {4, 1}}, false};
    G_RECIPE_DEFINITIONS[901] = {901, "Infuse Proto-Blade (Quantum)", 9, 1, {{7, 5}, {8, 1}}, true};

    // Placeholder NPC Definitions (id, name, desc, type, health, attack, defense)
    G_NPC_DEFINITIONS[1001] = {1001, "Town Guard", "A vigilant guard patrolling the town.", "FRIENDLY_GUARD", 150, 15, 10};
    G_NPC_DEFINITIONS[1002] = {1002, "Mysterious Merchant", "A merchant dealing in rare goods.", "NEUTRAL_VENDOR", 75, 5, 5};
    G_NPC_DEFINITIONS[2001] = {2001, "Forest Goblin", "A pesky goblin lurking in the woods.", "HOSTILE_MOB", 50, 8, 3};

    // Zone Definitions
    G_ZONE_DEFINITIONS["starting_zone_town_square"] = {"starting_zone_town_square", "Town Square", "The bustling heart of the starting town."};
    G_ZONE_DEFINITIONS["starting_zone_market"] = {"starting_zone_market", "Market District", "A place of commerce and trade."};
    G_ZONE_DEFINITIONS["starting_zone_forest_edge"] = {"starting_zone_forest_edge", "Forest Edge", "The dark woods begin here."};
    G_ZONE_DEFINITIONS["dungeon_level_1"] = {"dungeon_level_1", "Abandoned Mines - Level 1", "Dark and treacherous tunnels."};
    G_ZONE_DEFINITIONS["unknown_zone"] = {"unknown_zone", "The Void", "An undefined area."};

    // Sample Quests
    G_QUEST_DEFINITIONS[1] = {
        1, "Collect Iron Ore", "The blacksmith needs iron ore...", "Gather 3 Iron Ores...",
        {1, 0}, {{QuestObjectiveType::COLLECT_ITEM, 1, 3, "Gather 3 Iron Ores"}}, {100, 0, 0}, 1001, 1001
    };
    G_QUEST_DEFINITIONS[2] = {
        2, "The Lost Sword", "Guard Thompson lost his spare sword...", "Find Guard Thompson's Lost Sword...",
        {2, 1}, {{QuestObjectiveType::COLLECT_ITEM, 5, 1, "Find 1 Basic Sword"}}, {250, 6, 1}, 1001, 1001
    };
    G_QUEST_DEFINITIONS[3] = {
        3, "Urgent Delivery", "A Town Guard needs you to deliver a message...", "Deliver the 'Sealed Message' to the Merchant.",
        {1, 0}, {{QuestObjectiveType::TALK_TO_NPC, 1002, 1, "Deliver message to Mysterious Merchant"}}, {150, 0, 0}, 1001, 1002
    };
}
