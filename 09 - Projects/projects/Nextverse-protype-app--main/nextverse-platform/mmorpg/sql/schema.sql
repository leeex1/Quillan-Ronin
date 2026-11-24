-- File: nextverse-platform/mmorpg/sql/schema.sql

-- Player Accounts Table
CREATE TABLE IF NOT EXISTS player_accounts (
    account_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL, -- Store hashed passwords, not plaintext
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Characters Table
CREATE TABLE IF NOT EXISTS characters (
    character_id SERIAL PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES player_accounts(account_id),
    character_name VARCHAR(50) UNIQUE NOT NULL,
    level INTEGER DEFAULT 1,
    experience INTEGER DEFAULT 0,
    health INTEGER DEFAULT 100,
    position_x DOUBLE PRECISION DEFAULT 0.0,
    position_y DOUBLE PRECISION DEFAULT 0.0,
    position_z DOUBLE PRECISION DEFAULT 0.0,
    zone_id VARCHAR(50) DEFAULT 'starting_zone',
    attack_power INTEGER DEFAULT 10,
    defense INTEGER DEFAULT 5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_characters_account_id ON characters(account_id);
CREATE INDEX IF NOT EXISTS idx_player_accounts_username ON player_accounts(username);
CREATE INDEX IF NOT EXISTS idx_characters_character_name ON characters(character_name);

-- Character Inventories Table
CREATE TABLE IF NOT EXISTS character_inventories (
    inventory_id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES characters(character_id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT uq_character_item UNIQUE (character_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_character_inventories_character_id ON character_inventories(character_id);

-- NPC Instances Table
CREATE TABLE IF NOT EXISTS npc_instances (
    instance_id SERIAL PRIMARY KEY,
    definition_id INTEGER NOT NULL,
    unique_name VARCHAR(100),
    current_health INTEGER NOT NULL,
    pos_x DOUBLE PRECISION DEFAULT 0.0,
    pos_y DOUBLE PRECISION DEFAULT 0.0,
    pos_z DOUBLE PRECISION DEFAULT 0.0,
    zone_id VARCHAR(50) NOT NULL,
    spawned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_npc_instances_zone_id ON npc_instances(zone_id);

-- Character Quests Table
CREATE TABLE IF NOT EXISTS character_quests (
    character_quest_id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES characters(character_id) ON DELETE CASCADE,
    quest_id INTEGER NOT NULL,
    status INTEGER NOT NULL DEFAULT 2,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_character_quest UNIQUE (character_id, quest_id)
);
CREATE INDEX IF NOT EXISTS idx_character_quests_character_id ON character_quests(character_id);

-- Guilds Table
CREATE TABLE IF NOT EXISTS guilds (
    guild_id SERIAL PRIMARY KEY,
    guild_name VARCHAR(100) UNIQUE NOT NULL,
    leader_character_id INTEGER NOT NULL REFERENCES characters(character_id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_guilds_guild_name ON guilds(guild_name);
CREATE INDEX IF NOT EXISTS idx_guilds_leader_character_id ON guilds(leader_character_id);

-- Guild Members Table
CREATE TABLE IF NOT EXISTS guild_members (
    guild_member_id SERIAL PRIMARY KEY,
    guild_id INTEGER NOT NULL REFERENCES guilds(guild_id) ON DELETE CASCADE,
    character_id INTEGER NOT NULL REFERENCES characters(character_id) ON DELETE CASCADE,
    guild_rank INTEGER NOT NULL DEFAULT 0, -- 0: Member, 1: Officer, 2: Leader
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_character_guild_membership UNIQUE (character_id), -- Enforces one guild per character
    CONSTRAINT uq_guild_character_combo UNIQUE (guild_id, character_id) -- Ensures char not listed twice in same guild
);
CREATE INDEX IF NOT EXISTS idx_guild_members_guild_id ON guild_members(guild_id);
-- Index on character_id is implicitly created by the UNIQUE constraint uq_character_guild_membership
-- CREATE INDEX IF NOT EXISTS idx_guild_members_character_id ON guild_members(character_id); -- So this is not strictly needed
