# NextVerse Persistent MMORPG Universe — Design Specification 2026

**Document status:** Active design specification
**Scope:** `nextverse-platform/mmorpg` and its integration surfaces across the NextVerse platform
**Path convention:** Every path below is relative to the repository root `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main`. Line references are included where they pin a specific claim.
**Grounding rule:** Statements about code that exists today cite a real file. Statements about future work are labeled **Planned** and never presented as implemented.

---

## 1. Vision and Player Fantasy

NextVerse is explicitly "Inspired by *Ready Player One*" (`README.md:9`). The intended player fantasy is a sovereign, persistent universe in which a player owns an identity, a customizable avatar, a home guild, and a place in a living world whose NPCs and events react to player action. The canonical architecture table names the MMORPG World as the "Associative Cortex" with JSSC as its engine (`README.md:38`), and the beta architecture expands this into sharded zones, dynamic NPCs, player-driven markets, and emergent quests (`beta files/custom_mmorpg_module_architecture.md:9-62`).

The design target for 2026 is a **single authoritative world** with:

- Persistent characters and inventories stored in PostgreSQL, not in process memory alone.
- Zone-scoped event dissemination over a publish/subscribe bus.
- Server-authoritative combat and crafting so that the client never decides outcomes.
- AI-assisted NPCs that use the existing AI Assistant service for contextual dialogue.
- Avatar behavior adapted through the quantum-avatars adapter layer.

The current implementation is an **early vertical slice** of that vision: a C++ server with real PostgreSQL persistence and a real ZeroMQ transport, joined to placeholder behavior layers. Section 2 records exactly what is present.

---

## 2. Ground Truth: What Exists Today

| Component | Real file | What the code actually does today |
| :--- | :--- | :--- |
| Server host and loop | `nextverse-platform/mmorpg/include/mmorpg_server.hpp:19-148`, `:151-296` | Defines `MMORPG_Server`, binds a ZeroMQ REP socket and a ZeroMQ PUB socket, and runs a single-threaded polling loop that parses space-delimited text commands. |
| Process entry point | `nextverse-platform/mmorpg/src/main.cpp:14-96` | Initializes static data, reads `DB_CONNECTION_STRING`, opens PostgreSQL, applies `schema.sql`, seeds NPCs when the table is empty, then constructs the server with `QuantumVM` and calls `run()`. |
| Player manager | `nextverse-platform/mmorpg/include/PlayerManager.h:15-64`, `nextverse-platform/mmorpg/src/PlayerManager.cpp:24-95` | Tracks `playerId` to `PlayerData` in a `std::map`, handles connect, reconnect, disconnect, lookup, and position updates. Connected state and `lastSeen` are maintained. |
| Player state DTO | `nextverse-platform/mmorpg/include/PlayerData.h:30-81` | Holds identity, position, `zone_id`, health, level, experience, attack/defense, inventory, and quest state with two constructors defaulting to level 1, health 100, attack 10, defense 5. |
| NPC behavior interface | `nextverse-platform/mmorpg/include/NPCBehavior.h:12-101` | Declares `NPCState`, abstract `NPCBehavior::update`, and `SimpleIdleNPCBehavior`, whose update marks `currentBehavior` as `idle_processed`. |
| Synchronization | `nextverse-platform/mmorpg/include/Synchronization.h:19-47`, `nextverse-platform/mmorpg/src/Synchronization.cpp:22-64` | Provides `synchronizePlayerStates` and `synchronizeNPCStates`; both log counts and check null pointers. Serialization and network send are explicitly described as placeholder in the source comments. |
| Quantum crafting mock | `nextverse-platform/mmorpg/include/quantum_vm_mock.hpp:15-79` | `QuantumVM::processQuantumCrafting` accepts recipe 901 only, applies a `rand() % 100` roll, and returns `SUCCESS:minor_stability_flux:0`, `SUCCESS:damage_bonus:N`, or `SUCCESS:crit_chance_bonus:N`; non-901 recipes return `FAILURE:recipe_not_quantum_compatible`. |
| Static game data | `nextverse-platform/mmorpg/include/game_data.hpp:9-170`, `nextverse-platform/mmorpg/src/game_data_store.cpp:10-56` | Defines item, recipe, NPC, zone, quest, and guild structs and populates global maps with 12 items, 4 recipes, 3 NPC definitions, 5 zones, and 3 quests. |
| Database layer | `nextverse-platform/mmorpg/src/database_manager.hpp:21-628`, `nextverse-platform/mmorpg/src/db_connection.hpp:9-54` | Wraps libpqxx. Implements account creation, login verification, character CRUD, inventory add/remove/consume, quest status, NPC instance CRUD, and full guild membership operations. All value-bearing statements use `exec_params`. |
| Schema | `nextverse-platform/mmorpg/sql/schema.sql:1-93` | Creates seven tables: `player_accounts`, `characters`, `character_inventories`, `npc_instances`, `character_quests`, `guilds`, `guild_members`, plus five named indexes and two uniqueness constraints on guild membership. |
| Build | `nextverse-platform/mmorpg/CMakeLists.txt:1-62` | C++17 target `mmorpg_server` from four `.cpp` files; links `NetworkModule`, `PkgConfig::ZMQ`, `PkgConfig::PQXX`, and PostgreSQL libraries. |
| Integration test client | `nextverse-platform/mmorpg/test_client.py:5-126` | Python REQ client sending a 30-command lifecycle, plus a SUB thread subscribed to all PUB topics for 45 seconds. |
| Container image | `nextverse-platform/mmorpg/Dockerfile:1-36` | Ubuntu 22.04 image installing g++, cmake, libzmq3-dev, libpqxx-dev, and pyzmq; builds the server and exposes ports 5555 and 5556. |
| Health probe | `nextverse-platform/mmorpg/health.py:1-3` | Prints `{"status": "healthy", "module": "mmorpg"}`. |
| Avatar adapter | `nextverse-platform/quantum_avatars/include/AvatarAdapter.h:17-104`, `nextverse-platform/quantum_avatars/src/AvatarAdapter.cpp:12-101` | Maps `UserAction.type` and `details` strings to behavior states such as `aggressive`, `defensive`, `evasive_fleeing`, `neutral_friendly`, `neutral_trading`, `evasive_aoe`, and `passive_safe`. |
| UI surface | `nextverse-platform/web-ui/src/components/MMRPGWorld.tsx:4-15` | A React/MUI card labeled "MMORPG World - SocialFabric" describing a "JSSC 500-player simulation with Rust backend"; it is a static presentation card, not a game client. |

**Critical documented divergence:** `nextverse-platform/docker-compose.yml:55-72` describes the MMORPG service as a "Rust SocialFabric" with `ACTIX_WORKERS=8`, `REDIS_URL=redis://redis:6379`, HTTP port 3005, and a WebSocket port 8080. The code that exists is the C++17 ZeroMQ/PostgreSQL server described above; there is no Rust source, no Actix runtime, and no Redis client in `nextverse-platform/mmorpg`. This document treats the compose entry as the **planned deployment shape** and the C++ tree as the **current implementation**.

---

## 3. World Topology and Zones

### 3.1 Zones that exist today

`nextverse-platform/mmorpg/src/game_data_store.cpp:37-41` registers five `ZoneDefinition` entries in `G_ZONE_DEFINITIONS`:

| Zone ID (actual key) | Friendly name | Description in code |
| :--- | :--- | :--- |
| `starting_zone_town_square` | Town Square | The bustling heart of the starting town. |
| `starting_zone_market` | Market District | A place of commerce and trade. |
| `starting_zone_forest_edge` | Forest Edge | The dark woods begin here. |
| `dungeon_level_1` | Abandoned Mines - Level 1 | Dark and treacherous tunnels. |
| `unknown_zone` | The Void | An undefined area. |

NPC instances are seeded into three of these zones at `nextverse-platform/mmorpg/src/main.cpp:48-53`: Guard Thompson (definition 1001) into `starting_zone_town_square`, Mysterious Merchant (definition 1002) into `starting_zone_market`, and Forest Goblin Alpha and Bravo (definition 2001) into `starting_zone_forest_edge`.

Zone identity is carried in three places: `PlayerData::zone_id` (`nextverse-platform/mmorpg/include/PlayerData.h:41`), `characters.zone_id` (`nextverse-platform/mmorpg/sql/schema.sql:24`), and `NPCInstanceData::zone_id` (`nextverse-platform/mmorpg/include/game_data.hpp:62`). A minor inconsistency exists today: the field initializer at `PlayerData.h:41` uses `"starting_zone"`, the default constructor at `:62` uses `"default_zone"`, and the schema default at `schema.sql:24` uses `"starting_zone"`. Standardizing on `starting_zone_town_square` is part of the beta work.

### 3.2 Event topology that exists today

`MMORPG_Server::publishWorldEvent` (`nextverse-platform/mmorpg/include/mmorpg_server.hpp:68-82`) builds the PUB topic as the zone ID concatenated with a dot and the base topic when a zone filter is supplied, and as `GLOBAL.` concatenated with the base topic otherwise. Live topics emitted by the current code are `COMBAT_LOG` (`:230`, `:248`, `:339`), `NPC_AMBIENT_SPEECH` (`:320`), and `SIMULATION_UPDATE` (`:330`). `nextverse-platform/mmorpg/test_client.py:75-107` subscribes to all topics and prints them.

### 3.3 Planned topology

**Planned** world topology extends the five real zones without renaming them, adding:

| Tier | Regions | Purpose |
| :--- | :--- | :--- |
| Starter continuum | `starting_zone_town_square`, `starting_zone_market`, `starting_zone_forest_edge` | Onboarding, commerce, and first combat; already seeded today. |
| Dungeon ladder | `dungeon_level_1` plus new `dungeon_level_2` through `dungeon_level_5` | Instanced PvE progression with per-party shards. |
| Civic hub | `nexus_atrium` | Cross-zone social lobby; feeds the social-hub design in `nextverse-platform/docs/SOCIAL_HUBS_DESIGN_2026.md`. |
| Guild territory | `guild_temple_1` as the first instance, with one room per guild ID | Persistent guild halls backed by the existing `guilds` and `guild_members` tables. |
| Event arena | `live_event_arena` | Time-boxed global events published on `GLOBAL.EVENT_*` topics. |

**Planned** sharding and spatial partitioning follow `beta files/custom_mmorpg_module_architecture.md:11-14`: separate server instances own separate regions, and a zone-to-shard registry routes `ENTER_WORLD` and `MOVE` commands to the owning instance. The current `MMORPG_Server` owns all zones in one `active_npcs_` map (`nextverse-platform/mmorpg/include/mmorpg_server.hpp:147`).

---

## 4. Avatar and Identity Model

### 4.1 Identity today

Account and character are already distinct entities in `nextverse-platform/mmorpg/sql/schema.sql:4-29`: `player_accounts` owns `account_id`, `username`, `password_hash`, and `email`; `characters` owns `character_id`, a foreign key to `account_id`, `character_name`, level, experience, health, position, zone, attack, and defense. `DatabaseManager::createCharacter` inserts a new character with level 1, experience 0, attack 10, and defense 5 (`nextverse-platform/mmorpg/src/database_manager.hpp:120-142`), matching the defaults in `PlayerData` (`nextverse-platform/mmorpg/include/PlayerData.h:60-80`).

`MMORPG_Server::run` accepts `ENTER_WORLD 1` as exercised by `test_client.py:22` (`nextverse-platform/mmorpg/include/mmorpg_server.hpp:177-198`), loads the character through `getCharacterData`, and inserts it into `active_characters_`. Duplicate entry is rejected. A character removed from the world is not persisted on disconnect by the current loop, so `saveCharacterData` (`:186-198` in `database_manager.hpp`) is the hook the beta must call consistently.

### 4.2 Avatar behavior adapter today

`AvatarAdapter` (`nextverse-platform/quantum_avatars/include/AvatarAdapter.h:45-104`) holds an `avatarId_` and a `currentBehaviorState_` defaulting to `idle`. `adaptBehavior` validates that the action type is non-empty, calls `determineNewBehavior`, and returns either a new-state or unchanged-state string; exceptions return an error string and preserve the current state (`AvatarAdapter.cpp:48-75`). The mapping table is string-based:

| Action type | Detail substring | Resulting state | Source |
| :--- | :--- | :--- | :--- |
| `combat_move` | `attack`, `defend`, `flee` | `aggressive`, `defensive`, `evasive_fleeing` | `AvatarAdapter.cpp:15-22` |
| `interaction` | `friendly_greeting`, `threaten`, `query_trade` | `neutral_friendly`, `cautious_hostile`, `neutral_trading` | `AvatarAdapter.cpp:23-30` |
| `environment_change` | `imminent_danger_aoe`, `safe_zone_entered` | `evasive_aoe`, `passive_safe` | `AvatarAdapter.cpp:31-36` |
| `low_health_event` | any | `defensive_wounded` | `AvatarAdapter.cpp:37-39` |

**Planned** avatar roster: `beta files/QUANTUM_AVATAR_README.md:17-31` specifies three archetypes — Excalibur (player-controlled, quantum sword attacks), Volt (NPC, accelerated movement), and Mag (NPC, spatial distortion) — with a `QuantumEngine` and `QuantumAvatar` class pair at `:74-83`. Those classes are **not present** in `nextverse-platform/quantum_avatars`, which currently contains only `AvatarAdapter.h` and `AvatarAdapter.cpp`. The 2026 work maps the planned archetypes onto the real adapter states: Excalibur starts at `idle` and enters `aggressive` on `combat_move` containing `attack`; Volt enters `evasive_aoe`; Mag enters `neutral_trading` during trade interactions.

### 4.3 Planned identity contract

**Planned** unified identity resolution: `account_id` from `player_accounts` is the sovereign identity; `character_id` from `characters` is the in-world identity; `playerId` in `PlayerData` is bound to `character_id`, not to an account. This removes the current ambiguity where `test_client.py:18-22` sends `CREATE_ACCOUNT`, `LOGIN`, and `CREATE_CHAR` as text commands but `mmorpg_server.hpp` implements only `ENTER_WORLD`, `ATTACK`, and a reply gate that names `CRAFT`, `HAIL`, `QUEST_*` (`:177-268`). Reconciling the client command list with the server command set is an alpha acceptance item.

---

## 5. Persistence and Database Model

The persistence layer is the strongest part of the current slice.

| Table in `sql/schema.sql` | Columns of note | Consuming code |
| :--- | :--- | :--- |
| `player_accounts:4-11` | `account_id`, unique `username`, `password_hash`, unique `email`, `created_at`, `last_login` | `createPlayerAccount` and `verifyPlayerLogin` at `database_manager.hpp:58-90` |
| `characters:14-29` | `character_id`, `account_id` FK, unique `character_name`, level, experience, health, x/y/z, `zone_id`, attack, defense, timestamps | `listCharactersForAccount`, `createCharacter`, `getCharacterData`, `saveCharacterData` at `database_manager.hpp:92-198` |
| `character_inventories:37-44` | `inventory_id`, `character_id` FK with cascade delete, `item_id`, `quantity`, unique pair constraint | `getCharacterInventory`, `addItemToInventory`, `removeItemFromInventory`, `checkAndConsumeIngredients` at `database_manager.hpp:255-351` |
| `npc_instances:47-58` | `instance_id`, `definition_id`, `unique_name`, `current_health`, x/y/z, `zone_id`, `spawned_at` | `getAllNPCInstances`, `createNPCInstanceInDB`, `setNPCInstanceHealth` at `database_manager.hpp:368-428` |
| `character_quests:61-69` | `character_quest_id`, `character_id` FK, `quest_id`, `status` default 2, `last_updated`, unique pair constraint | `getPlayerQuestStatus`, `getAllPlayerQuests`, `updatePlayerQuestStatus` at `database_manager.hpp:203-253` |
| `guilds:72-79` | `guild_id`, unique `guild_name`, `leader_character_id` FK, `created_at` | `createGuild`, `getGuildById`, `getAllGuildsSummary` at `database_manager.hpp:436-465`, `:492-522`, `:581-605` |
| `guild_members:82-92` | `guild_id` FK, `character_id` FK, `guild_rank` default 0, `joined_at`, one-guild-per-character constraint | `getGuildMembers`, `addGuildMember`, `removeGuildMember`, `getCharacterGuildInfo`, `updateGuildMemberRank` at `database_manager.hpp:467-626` |

`executeSQLFile` reads the entire schema file into a string and executes it in one `pqxx::work` transaction (`database_manager.hpp:25-56`); `main.cpp:37-42` calls it with `../sql/schema.sql`, so the server's working directory must be the module root or build directory that resolves that relative path.

**Planned** persistence additions:

1. `world_events` table for durable global event history, keyed by the `GLOBAL.` topic prefix convention in `mmorpg_server.hpp:68-82`.
2. `character_avatar_states` table storing the `AvatarAdapter` behavior string from `AvatarAdapter.h:95` so avatar mood survives relogin.
3. `audit_log` table for administrative actions and moderation decisions, satisfying the accountability requirement in Section 10.
4. A periodic flush of `active_characters_` (`mmorpg_server.hpp:146`) to `saveCharacterData`, since the live loop currently persists hp changes only inside the player-versus-player branch at `:253`.

---

## 6. NPC Behavior and AI Assistant Integration

### 6.1 NPC behavior today

`NPCBehavior` defines a pure virtual `update(const NPCState&, double)` and two static loggers (`nextverse-platform/mmorpg/include/NPCBehavior.h:43-71`). `SimpleIdleNPCBehavior::update` copies the state and sets `currentBehavior` to `idle_processed` without moving the NPC (`:80-101`). Because `NPCBehavior.h` contains no `.cpp` and is not part of the compiled source list in `CMakeLists.txt:22-31`, this interface is a **design seam**, not a live simulation today.

The live NPC simulation is in `MMORPG_Server`: `loadInitialNPCs` hydrates `active_npcs_` from PostgreSQL and copies attack and defense from `G_NPC_DEFINITIONS` (`mmorpg_server.hpp:39-66`), while `performAmbientNPCActions` rolls a 2 percent chance per NPC and emits vendor, guard, or goblin flavor lines on `NPC_AMBIENT_SPEECH` (`:298-325`). `handleNPCDefeat` removes the NPC from the active map and publishes a combat event but leaves XP granting commented out (`:333-350`).

### 6.2 AI Assistant integration points

The AI Assistant exposes two concrete interfaces that the world can use:

- SocketIO events `connect`, `disconnect`, and `user_message` on the Flask-SocketIO server (`nextverse-platform/ai-assistant/api/chat_api.py:79-130`), with `ai_response` emitted back to the originating client. `user_message` expects `message`, `user_id`, and `session_id`; the handler defaults `user_id` to `default_web_user` and `session_id` to the SocketIO SID (`:104-106`).
- HTTP `POST /contextual_query` accepting `event_type`, `object_id`, and optional `player_location` (`chat_api.py:142-180`), routed to `AIBrain.process_game_context_query` (`nextverse-platform/ai-assistant/core/ai_brain.py:291-321`). That method currently returns predefined strings for `inspect_object` on `crystal_A` and `ancient_tablet_1`, and for `character_interaction_hint` on `npc_wizard_01` (`:302-316`).

`AIBrain` also holds a `GameEngineClient` (`ai_brain.py:88-105`) that sends newline-delimited JSON commands over TCP (`nextverse-platform/ai-assistant/core/game_engine_client.py:23-61`) to `GAME_ENGINE_HOST` and `GAME_ENGINE_COMMAND_PORT`, defaulting to `localhost:8081` (`nextverse-platform/ai-assistant/utils/config.py:65-67`).

**Planned** NPC-AI loop: `HAIL 1 1` as exercised by `test_client.py:38` and `inspect_object` events publish into `/contextual_query`; the returned `ai_generated_message` is emitted on the zone-scoped `NPC_DIALOGUE` topic; the response is filtered by the same ethical gate used for chat. This realizes the "AI-driven NPCs" and "adaptive storytelling" goals in `beta files/custom_mmorpg_module_architecture.md:36-38` and `:54-62` without inventing a new transport.

---

## 7. Networking and Synchronization

### 7.1 Transport today

| Endpoint | Code | Role |
| :--- | :--- | :--- |
| `tcp://*:5555` | `nextverse-platform/mmorpg/src/main.cpp:81`, `nextverse-platform/mmorpg/include/mmorpg_server.hpp:28-29` | ZeroMQ REP socket for synchronous commands. |
| `tcp://*:5556` | `nextverse-platform/mmorpg/src/main.cpp:81`, `mmorpg_server.hpp:30-31` | ZeroMQ PUB socket for zone-scoped and global events. |
| `tcp://localhost:5555` and `tcp://localhost:5556` | `nextverse-platform/mmorpg/test_client.py:9`, `:75` | Reference REQ and SUB clients. |

`MMORPG_Server::run` polls REP with `dontwait`, sleeps 1 millisecond when idle, and triggers ambient NPC actions after 500000 idle iterations (`mmorpg_server.hpp:162-284`). The reply gate at `:265-275` permits replies for `CRAFT`, `ATTACK`, `HAIL`, `QUEST_COMPLETE`, `QUESTS_AVAILABLE`, `QUEST_ACCEPT`, `QUEST_LOG`, and `HAIL_SENT`.

### 7.2 Synchronization module today

`Synchronization::synchronizePlayerStates` and `synchronizeNPCStates` iterate the supplied vectors, reject null player pointers, and log counts (`nextverse-platform/mmorpg/src/Synchronization.cpp:22-64`). The comments at `:31-33` and `:57-58` explicitly describe serialization and sending as future work. The class is compiled into the target (`nextverse-platform/mmorpg/CMakeLists.txt:26`) but is not invoked from `main.cpp`.

### 7.3 Network module today

`nextverse-platform/network/include/NetworkHandler.h:20-115` declares connect, disconnect, send, receive, and callback-based asynchronous receive with `connected_` and `receiving_` flags; the socket internals are commented placeholders (`:106-110`). `SecureConnection.h:16-97` declares TLS-style establish, send-encrypted, and receive-encrypted methods with certificate setters commented out (`:63-75`). `LoadBalancer.h:41-94` implements a round-robin `getNextServer` over `ServerNode` entries with no load metric (`:19-21`). The MMORPG binary links this module through `nextverse-platform/mmorpg/CMakeLists.txt:50`.

### 7.4 Planned networking evolution

**Planned** sequencing:

1. **Alpha:** Implement binary or JSON framing inside `Synchronization` and call it from the REP loop so every connected client receives a consistent player/NPC snapshot. Keep ZeroMQ endpoints unchanged.
2. **Beta:** Introduce the WebSocket endpoint on port 8080 declared in `nextverse-platform/docker-compose.yml:59-60`, backed by the same event topics, so the React UI in `nextverse-platform/web-ui/src/App.tsx:176-183` can subscribe to world events instead of rendering a static card.
3. **2026 launch:** Realize `SecureConnection` with TLS, wire `LoadBalancer::getNextServer` to zone ownership, and shard by zone per `beta files/custom_mmorpg_module_architecture.md:11-14` and `:44-52`.

---

## 8. Economy, Progression, and Social Systems

### 8.1 Economy today

`nextverse-platform/mmorpg/src/game_data_store.cpp:12-23` defines 12 items including Iron Ore (1), Wood Log (2), Iron Ingot (3), Health Potion (6), Quantum Shard (7), Unstable Matrix (8), and Proto-Blade (9). Recipes at `:26-29` are Smelt Iron Ingot (101), Craft Wooden Handle (102), Craft Basic Sword (103), and Infuse Proto-Blade (901, flagged `is_quantum_craft = true`). Ingredient consumption is transactional with `SELECT ... FOR UPDATE` in `checkAndConsumeIngredients` (`nextverse-platform/mmorpg/src/database_manager.hpp:317-351`), which provides a correct basis for server-authoritative crafting.

Quantum crafting is the differentiator: recipe 901 routes to `QuantumVM::processQuantumCrafting` (`quantum_vm_mock.hpp:32-58`) and yields one of three outcome strings. The logistics of actually adding the resulting item to inventory are not implemented in the current server loop and are a beta item.

### 8.2 Progression today

`QuestDefinition` carries prerequisites, objectives of type `COLLECT_ITEM` or `TALK_TO_NPC`, rewards, and start/completion NPC definition IDs (`nextverse-platform/mmorpg/include/game_data.hpp:86-119`). Three quests are defined at `game_data_store.cpp:44-55`: Collect Iron Ore (1), The Lost Sword (2), and Urgent Delivery (3). `PlayerQuestStatus` enumerates five states from `NOT_ELIGIBLE` to `COMPLETED` (`game_data.hpp:121-127`). `checkQuestPrerequisites` and `checkQuestObjectives` in `mmorpg_server.hpp:95-131` evaluate prerequisite quest completion and inventory counts against the database.

`experience` is stored in `characters.experience` (`schema.sql:19`) and updated by `updateCharacterExperience` (`database_manager.hpp:353-365`), but `handleNPCDefeat` currently leaves the XP grant commented out (`mmorpg_server.hpp:343-349`). Wiring that call is the minimum viable combat-to-progression loop.

### 8.3 Social systems today

Guilds are fully modeled in SQL and in the C++ data layer: `GuildRank` has MEMBER, OFFICER, and LEADER (`game_data.hpp:137-141`), `GuildData` and `GuildMemberData` carry names, leaders, timestamps, and ranks (`:146-166`), and `DatabaseManager` supplies create, lookup, member add/remove, rank update, and summaries (`database_manager.hpp:436-626`). The one-guild-per-character rule is enforced by `uq_character_guild_membership` (`schema.sql:88`). No guild command is currently parsed by `MMORPG_Server::run`; the guild layer is a ready backend waiting for a command surface.

Player-driven markets, auctions, and mentor systems remain **Planned** per `beta files/custom_mmorpg_module_architecture.md:40-42`.

---

## 9. Anti-Cheat and Moderation

No anti-cheat or moderation subsystem exists today. The following controls are proposed on top of surfaces that already exist.

| Threat | Current surface | Proposed control | Acceptance signal |
| :--- | :--- | :--- | :--- |
| Position teleport | `updatePlayerPosition` writes client-supplied x/y/z directly (`PlayerManager.cpp:76-95`) | Server-side speed and bounds validation against `PlayerData::zone_id` (`PlayerData.h:41`), rejecting deltas beyond a per-tick maximum. | A test client sending a 1000-unit jump receives an error and position is unchanged. |
| Damage spoofing | `ATTACK` parses target IDs from text and computes damage server-side using a hardcoded 10 (`mmorpg_server.hpp:223-248`) | Replace hardcoded `10` with the attacker's persisted `attack_power` from `characters.attack_power` (`schema.sql:25`) plus a server-side combat formula; never accept damage from the client. | Two characters with different `attack_power` produce different damage values. |
| Item duplication | `addItemToInventory` increments quantities (`database_manager.hpp:271-288`) | Require every award to carry an idempotency key derived from the event topic and NPC instance ID already published at `mmorpg_server.hpp:339`. | Replaying the same `handleNPCDefeat` event twice awards loot once. |
| Quest farming | `updatePlayerQuestStatus` accepts any `PlayerQuestStatus` (`database_manager.hpp:236-253`) | Enforce forward-only transitions through the five states in `game_data.hpp:121-127` inside the server, never in the client. | An attempt to move from `NOT_ACCEPTED` to `COMPLETED` is rejected. |
| Chat harassment | AI Assistant handles `user_message` with no user trust level (`chat_api.py:90-130`) | Route all hub and world chat through `EthicalFramework.validate_content` (`ai-assistant/core/ethical_framework.py:243-326`), using the `social` context weights at `:92` that emphasize harmlessness 1.5 and privacy 1.4. | A message triggering a forbidden-topic regex at `:97-115` produces a refusal identical in shape to `ai_brain.py:397-425`. |
| Botting | `test_client.py:17-53` demonstrates scripted command replay | Per-session command-rate limits on the REP loop at `mmorpg_server.hpp:162-284`, keyed by character ID. | A client exceeding the rate receives a throttling error and is disconnected after repeated violations. |

Moderation actions must be written to the proposed `audit_log` table so that every mute, kick, or rollback has an actor, reason, and timestamp.

---

## 10. Security Posture (C13-WARDEN style)

The C13-WARDEN discipline is to threat-model the worst case and hard-gate anything hostile. Findings are ordered by severity.

| Finding | Evidence in current code | Risk | Mitigation |
| :--- | :--- | :--- | :--- |
| Hardcoded database fallback credentials | `nextverse-platform/mmorpg/src/main.cpp:26-27` falls back to `postgresql://testuser:testpass@localhost:5432/testdb` and prints the full connection string at `:29` | Critical in any shared environment: credentials and connection metadata leak to stdout | Require `DB_CONNECTION_STRING`, fail closed when absent, and log only the host and database name. |
| Login compares a supplied hash string, not a verified hash | `verifyPlayerLogin` selects by `username = $1 AND password_hash = $2` (`database_manager.hpp:74-90`) | High: a stored hash functions as a bearer secret; no salt or KDF is applied in this code path | Move to a per-user salt plus Argon2id or bcrypt verification, returning the account only after a constant-time comparison. |
| Secret keys checked into source | `chat_api.py:45` uses a literal Flask fallback secret; `ai-assistant/utils/config.py:34` contains a literal development secret | High: session forgery | Load both from environment, refuse to start in production mode when unset. |
| Wildcard CORS | `chat_api.py:49-53` applies `CORS(app)` and `SocketIO(..., cors_allowed_origins="*")` | High: any origin can drive the assistant and read responses in a browser | Restrict to the explicit web-ui origin from `nextverse-platform/web-ui/src/App.tsx`. |
| Unsafe deserialization on a local database | `dialogue_manager.py:13` imports `pickle`, and `:220` and `:330` serialize and unpickle embedding BLOBs | Medium today, high if the database file is ever shared | Replace pickle BLOBs with a fixed-format float array serialized as bytes. |
| Unauthenticated ZeroMQ sockets | `main.cpp:81` binds REP and PUB with no identity check; `test_client.py:12` connects unauthenticated | High at launch: any local process can issue `ATTACK` or `ENTER_WORLD` | Add an authenticated session token issued after login and validate it in the REP loop before dispatch. |
| Unbounded SQL script execution | `executeSQLFile` executes the whole file in one transaction (`database_manager.hpp:25-56`) | Medium: a modified schema file can alter or drop data | Restrict execution to an idempotent migration directory and checksum each file before execution. |
| Placeholder TLS | `SecureConnection.h:14-15` documents placeholder logic; certificate APIs are commented out at `:63-75` | High for production transport | Implement TLS before exposing port 8080 beyond localhost. |
| Ambient personality in production logs | `mmorpg_server.hpp:84-86` logs with console output and test commands are replayed in `test_client.py:17-53` | Low | Route all logging through a structured logger with correlation IDs. |

Parameterized queries are already used consistently in `database_manager.hpp`, which contains the SQL-injection class of risk for all value-bearing statements. That pattern must be preserved in every new query.

---

## 11. Phased Roadmap and Acceptance Criteria

### Phase A — Alpha: one coherent vertical slice

Goal: make the C++ server, the schema, and the test client agree.

| Step | Work | Verify |
| :--- | :--- | :--- |
| A1 | Replace the hardcoded fallback connection string in `main.cpp:26-27` with a required environment variable. | Starting the server without `DB_CONNECTION_STRING` exits with the message at `main.cpp:60`. |
| A2 | Implement the commands listed in `test_client.py:17-53` that are missing from `mmorpg_server.hpp:177-268`: `CREATE_ACCOUNT`, `LOGIN`, `LIST_CHARS`, `CREATE_CHAR`, `GET_STATE`, `INVENTORY`, `PICKUP`, `SAY`, `MOVE`, `LOOK`, `HAIL`, `DISCONNECT`. | `python nextverse-platform/mmorpg/test_client.py` completes all commands with no `ERROR: Unknown command` reply. |
| A3 | Normalize `zone_id` across `PlayerData.h:41`, `:62`, and `schema.sql:24`. | A freshly created character and a reloaded character report the same zone string. |
| A4 | Persist combat outcomes: call `updateCharacterExperience` on NPC defeat and `saveCharacterData` on disconnect. | Character XP and health in PostgreSQL match the in-memory values after a `DISCONNECT` and a fresh `ENTER_WORLD`. |
| A5 | Compile and link `NPCBehavior.h` implementations into `CMakeLists.txt:22-31` and register them with `active_npcs_`. | `SimpleIdleNPCBehavior::update` changes `currentBehavior` to `idle_processed` on a live NPC. |

### Phase B — Beta: synchronized, social, and observable

Goal: make state visible to clients and connect the AI Assistant.

| Step | Work | Verify |
| :--- | :--- | :--- |
| B1 | Implement serialization in `Synchronization.cpp:22-64` and invoke it from the REP loop. | A subscribed client receives player and NPC snapshots with matching `playerId` and `npcId` values. |
| B2 | Wire `HAIL` and `inspect_object` to `POST /contextual_query` (`chat_api.py:142-180`) and publish the reply on a zone-scoped `NPC_DIALOGUE` topic. | Hailing Guard Thompson (instance seeded at `main.cpp:48`) produces a dialogue event in the `starting_zone_town_square` topic. |
| B3 | Route all world chat through `EthicalFramework.validate_content` with `context="social"` (`ethical_framework.py:92`, `:243`). | A forbidden-topic message is refused before it reaches the PUB socket. |
| B4 | Add guild commands over the existing `DatabaseManager` guild methods at `database_manager.hpp:436-626`. | A character creates a guild, a second character joins at MEMBER, and `getGuildMembers` returns both. |
| B5 | Replace the static `MMORPGWorld.tsx:4-15` card with a SUB consumer of port 5556 events, fronted by the WebSocket endpoint declared in `docker-compose.yml:59-60`. | The web UI shows a live combat line within 2 seconds of an `ATTACK` command. |
| B6 | Apply the Section 10 mitigations for environment secrets, CORS, and ZeroMQ session tokens. | A request without a valid token receives an error and no state mutation. |

### Phase C — 2026 launch: sharded, secure, and persistent at scale

Goal: realize the topology in Section 3.3 and the deployment shape in `docker-compose.yml:55-72`.

| Step | Work | Verify |
| :--- | :--- | :--- |
| C1 | Introduce a zone-to-shard registry; route `ENTER_WORLD` and `MOVE` to the owning instance. | Two characters in different zones are served by different processes and both remain persistent. |
| C2 | Implement TLS in `network/include/SecureConnection.h` and replace plaintext REP traffic for client-facing connections. | A packet capture shows no cleartext character data. |
| C3 | Wire `LoadBalancer::getNextServer` (`network/include/LoadBalancer.h:72`) to live load metrics instead of pure round-robin. | Adding a node changes distribution within one heartbeat, and removing a drained node drops its traffic to zero. |
| C4 | Connect quantum crafting results from `quantum_vm_mock.hpp:39-52` to real inventory mutations through `addItemToInventory` (`database_manager.hpp:271-288`). | Recipe 901 consumes 5 Quantum Shards and 1 Unstable Matrix and adds exactly one outcome item. |
| C5 | Publish durable `world_events` and `audit_log` records for every moderation and high-value economy action. | Every admin action is replayable by timestamp and actor. |
| C6 | Achieve the 500-player target named in `README.md:73` and `docker-compose.yml:62` under the same schema. | A soak test with 500 connected characters holds PUB delivery below the timeout used in `test_client.py:59`. |

---

## 12. Risks and Mitigations

| Risk | Basis | Mitigation |
| :--- | :--- | :--- |
| Documentation and code diverge further | `docker-compose.yml:55-72` and `MMORPGWorld.tsx:8-11` claim a Rust backend that does not exist | Treat `mmorpg/CMakeLists.txt:22-31` as the build authority; update compose and UI only when the Rust service actually lands. |
| Single-threaded REP loop becomes the bottleneck | `mmorpg_server.hpp:162-284` polls one socket and sleeps 1 ms when idle | Move to a poller plus worker pool before phase C; keep `Synchronization` as the single state interface. |
| Placeholder layers are mistaken for production | `Synchronization.cpp:31-33`, `SecureConnection.h:14-15`, `quantum_vm_mock.hpp:1-2` all declare themselves mocks | Keep the Section 2 table current in review; require an acceptance signal for each promoted layer. |
| XP and loot never reach the database | XP grant is commented out at `mmorpg_server.hpp:343-349` | Phase A4 makes the write path mandatory and testable. |
| Credential leakage | `main.cpp:26-29` | Phase A1 and the Section 10 secret controls. |
| NPC dialogue floods the assistant | `performAmbientNPCActions` fires per-NPC on a 2 percent roll (`mmorpg_server.hpp:309`) | Rate-limit `NPC_AMBIENT_SPEECH` and cap contextual queries per zone per minute. |
| Quantum crafting is random and unreproducible | `rand() % 100` in `quantum_vm_mock.hpp:40` | Replace with a seeded server-side roll recorded in `world_events` before launch. |

---

## 13. Explicit Current-versus-Planned Statement

**Exists today (verifiable in code):** the C++17 `mmorpg_server` executable; ZeroMQ REP on 5555 and PUB on 5556; `ENTER_WORLD` and `ATTACK` handling; player connect, disconnect, and position updates; PostgreSQL schema with seven tables and full CRUD for characters, inventory, quests, NPCs, accounts, and guilds; five named zones; three quest definitions; twelve items and four recipes; the recipe-901 quantum crafting mock; ambient NPC speech; a Python REQ/SUB integration client; the `AvatarAdapter` behavior mapper; a health probe; and a Dockerfile.

**Planned (not present in code):** Rust SocialFabric, Actix workers, Redis caching, WebSocket port 8080, sharded zones, TLS, anti-cheat enforcement, guild commands, market and auction systems, the Excalibur/Volt/Mag `QuantumEngine` and `QuantumAvatar` classes, durable world events, moderation audit logs, and a live React world client. The planned items are grounded in `docker-compose.yml:55-72`, `beta files/QUANTUM_AVATAR_README.md:74-83`, `beta files/custom_mmorpg_module_architecture.md:11-62`, and `web-ui/src/components/MMRPGWorld.tsx:8-11`.

**Document relationship:** social gathering spaces that sit inside this universe are specified in `nextverse-platform/docs/SOCIAL_HUBS_DESIGN_2026.md`; this document remains the authority for world state, persistence, and combat.

---

## Appendix A — Primary Source Index

| Area | Files cited |
| :--- | :--- |
| Server | `nextverse-platform/mmorpg/include/mmorpg_server.hpp`, `nextverse-platform/mmorpg/src/main.cpp` |
| Players | `nextverse-platform/mmorpg/include/PlayerData.h`, `nextverse-platform/mmorpg/include/PlayerManager.h`, `nextverse-platform/mmorpg/src/PlayerManager.cpp` |
| NPCs and sync | `nextverse-platform/mmorpg/include/NPCBehavior.h`, `nextverse-platform/mmorpg/include/Synchronization.h`, `nextverse-platform/mmorpg/src/Synchronization.cpp` |
| Data and quantum | `nextverse-platform/mmorpg/include/game_data.hpp`, `nextverse-platform/mmorpg/src/game_data_store.cpp`, `nextverse-platform/mmorpg/include/quantum_vm_mock.hpp` |
| Persistence | `nextverse-platform/mmorpg/src/database_manager.hpp`, `nextverse-platform/mmorpg/src/db_connection.hpp`, `nextverse-platform/mmorpg/sql/schema.sql` |
| Build and test | `nextverse-platform/mmorpg/CMakeLists.txt`, `nextverse-platform/mmorpg/test_client.py`, `nextverse-platform/mmorpg/Dockerfile`, `nextverse-platform/mmorpg/health.py` |
| Cross-module | `nextverse-platform/quantum_avatars/include/AvatarAdapter.h`, `nextverse-platform/quantum_avatars/src/AvatarAdapter.cpp`, `nextverse-platform/network/include/NetworkHandler.h`, `nextverse-platform/network/include/SecureConnection.h`, `nextverse-platform/network/include/LoadBalancer.h`, `nextverse-platform/ai-assistant/api/chat_api.py`, `nextverse-platform/ai-assistant/core/ai_brain.py`, `nextverse-platform/ai-assistant/core/ethical_framework.py`, `nextverse-platform/ai-assistant/core/game_engine_client.py`, `nextverse-platform/web-ui/src/components/MMRPGWorld.tsx` |
| Documentation | `README.md`, `nextverse-platform/README.md`, `nextverse-platform/docker-compose.yml`, `beta files/custom_mmorpg_module_architecture.md`, `beta files/QUANTUM_AVATAR_README.md` |
