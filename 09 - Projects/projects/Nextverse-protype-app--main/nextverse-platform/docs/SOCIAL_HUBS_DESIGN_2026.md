# NextVerse Social Hubs — Design Specification 2026

**Document status:** Active design specification
**Scope:** Social gathering spaces across `nextverse-platform`, with explicit integration points into the MMORPG world, game engine, AI Assistant, DAW, and quantum VM
**Path convention:** Every path below is relative to the repository root `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main`. Line references pin specific claims.
**Grounding rule:** Statements about code that exists today cite a real file. Future capabilities are labeled **Planned** and are never described as shipped.

---

## 1. Vision and Hub Typology

The root README frames NextVerse as a local-first sovereign metaverse inspired by *Ready Player One* (`README.md:3-9`), where users own their compute, data, and experience (`README.md:13-26`). Social hubs are the places where that sovereignty becomes visible: persistent rooms in which an identity is recognized, a voice and text channel is available, creators can perform, and guilds can gather — all without a mandatory cloud dependency.

Six hub types are specified. Each is tied to a surface that already exists in the repository.

| Hub type | Purpose | Existing surface it extends |
| :--- | :--- | :--- |
| Lobby | Low-friction arrival space with presence, text, and navigation to other hubs | SocketIO `connect` and `disconnect` handlers in `nextverse-platform/ai-assistant/api/chat_api.py:79-88`; nav drawer in `nextverse-platform/web-ui/src/App.tsx:108-115` |
| Creator stages | Streamed performances with an audience and a moderated chat lane | Static DAW presentation card `nextverse-platform/web-ui/src/components/DAWStudio.tsx:4-15`; JUCE/Tracktion host `nextverse-platform/daw/source/NextVerseDAW/Source/MainComponent.h:17-46` |
| Live events | Time-boxed global happenings announced across zones | `MMORPG_Server::publishWorldEvent` topic rules in `nextverse-platform/mmorpg/include/mmorpg_server.hpp:68-82`; PUB subscriber in `nextverse-platform/mmorpg/test_client.py:75-107` |
| Temple and guild halls | Guild-owned persistent rooms with rank-gated access | `guilds` and `guild_members` tables in `nextverse-platform/mmorpg/sql/schema.sql:72-92`; `GuildRank` in `nextverse-platform/mmorpg/include/game_data.hpp:137-141`; guild CRUD in `nextverse-platform/mmorpg/src/database_manager.hpp:436-626` |
| DAW jam rooms | Synchronized multi-user music sessions | Tracktion `Engine` and `Edit` members in `MainComponent.h:35-44`; conceptual track and transport classes described in `nextverse-platform/daw/conceptual_logic/README.md:9-15` |
| AI-assisted gathering spaces | Rooms with a persistent, context-aware facilitator | `AIBrain` orchestration in `nextverse-platform/ai-assistant/core/ai_brain.py:52-105`; SQLite conversation and preference store in `nextverse-platform/ai-assistant/core/dialogue_manager.py:99-134` |

---

## 2. Ground Truth: What Exists Today

There is no social-hub module, hub schema, voice transport, or creator-payment code in `nextverse-platform` today. The following real surfaces are the available foundation.

| Capability | Real file | Actual state today |
| :--- | :--- | :--- |
| Real-time bidirectional text | `nextverse-platform/ai-assistant/api/chat_api.py:79-130` | Flask-SocketIO emits `connection_status` on connect, logs disconnect, and routes `user_message` to `AIBrain.generate_response`, replying with `ai_response`. Defaults `user_id` to `default_web_user` and `session_id` to the SocketIO SID at `:104-106`. |
| Contextual HTTP query | `nextverse-platform/ai-assistant/api/chat_api.py:142-180` | `POST /contextual_query` accepts `event_type`, `object_id`, and optional `player_location`; validated responses are returned as `ai_generated_message`. |
| Conversation persistence | `nextverse-platform/ai-assistant/core/dialogue_manager.py:99-134` | SQLite tables `conversations`, `user_preferences`, and `conversation_embeddings`, with an index on `conversation_embeddings(session_id)` at `:131-134`. Short-term memory is capped at 20 turns (`:37`). |
| Ethical filtering | `nextverse-platform/ai-assistant/core/ethical_framework.py:62-115` | Four moral factors with keyword sets, a `social` context weighting harmlessness 1.5 and privacy 1.4 (`:92`), and 15 regex-matched forbidden topics (`:97-115`). `validate_content` is at `:243-326`. |
| World identity | `nextverse-platform/mmorpg/sql/schema.sql:4-29` | `player_accounts` and `characters` separate account identity from in-world identity. |
| Guild identity | `nextverse-platform/mmorpg/sql/schema.sql:72-92` | `guilds`, `guild_members`, a one-guild-per-character uniqueness rule, and ranks 0, 1, 2 documented at `:86`. |
| World event fan-out | `nextverse-platform/mmorpg/include/mmorpg_server.hpp:68-82` | Topics are prefixed with the zone ID when supplied and with `GLOBAL` otherwise; live topics include `COMBAT_LOG` and `NPC_AMBIENT_SPEECH`. |
| Audio host | `nextverse-platform/daw/source/NextVerseDAW/Source/MainComponent.h:17-46` | A JUCE `AudioAppComponent` with `prepareToPlay`, `getNextAudioBlock`, `releaseResources`, and a Tracktion `Engine` plus `Edit`. |
| Conceptual audio logic | `nextverse-platform/daw/conceptual_logic/README.md:7-15` | Audio tracks, MIDI tracks, transport, and effects exist as C++ structures; the README states there is no actual audio processing, GUI, or real-time I/O. |
| UI routing | `nextverse-platform/web-ui/src/App.tsx:176-183` | Routes exist for dashboard, game engine, AI assistant, DAW studio, MMORPG world, and monitor. No hub route is registered. |
| DAW presentation | `nextverse-platform/web-ui/src/components/DAWStudio.tsx:4-15` | A static MUI card advertising "LSSS 96kHz @ 50ms latency with JUCE backend". |
| Quantum task gateway | `nextverse-platform/quantum-vm/quantum-master.py:9-72` | `QuantumMaster` exposes `GET /health` and `POST /execute`, splits task data across 11 instances, and merges results. |
| Avatar behavior | `nextverse-platform/quantum_avatars/src/AvatarAdapter.cpp:12-46` | String-driven behavior states including `neutral_friendly`, `neutral_trading`, and `passive_safe`. |
| Service topology | `nextverse-platform/docker-compose.yml:3-131` | Defines quantum-core, ai-assistant, game-engine, mmorpg, daw, orchestrator, redis, and monitoring services on a shared bridge network. |
| Health probes | `nextverse-platform/ai-assistant/health.py`, `nextverse-platform/daw/health.py`, `nextverse-platform/game-engine/health.py`, `nextverse-platform/orchestrator/health.py`, `nextverse-platform/mmorpg/health.py` | Each prints a one-line JSON status object with its module name. |

**Conclusion for this section:** presence, text, identity, guild structure, event fan-out, and an audio host all exist in separable pieces. The hub layer is an orchestration and persistence product built on those pieces, not a replacement for them.

---

## 3. Presence, Voice, Text, and Identity

### 3.1 Identity binding

**Planned** hub identity resolution uses two existing identifiers without inventing a third:

1. `account_id` from `player_accounts` (`schema.sql:5`) is the sovereign identity, used for billing, entitlements, and moderation history.
2. `character_id` from `characters` (`schema.sql:15`) is the visible in-world identity, used for guild rank and avatar state.

The SocketIO server already accepts `user_id` and `session_id` on every message (`chat_api.py:104-106`). **Planned** change: require both fields, validate `user_id` against an authenticated account, reject the `default_web_user` fallback in production, and store the resolved `account_id` and `character_id` together with the SID emitted at `chat_api.py:83`.

### 3.2 Presence model

Today, connection state exists only as SocketIO connect and disconnect log lines (`chat_api.py:79-88`) and as `PlayerData::isConnected` with `lastSeen` in `nextverse-platform/mmorpg/include/PlayerData.h:34-35`, maintained by `PlayerManager::playerConnect` and `playerDisconnect` (`nextverse-platform/mmorpg/src/PlayerManager.cpp:24-64`).

**Planned** presence record fields: `account_id`, `character_id`, `hub_id`, `presence_state` with values `online`, `away`, `busy`, `performing`, `in_combat`, and `last_seen` sourced from the existing `lastSeen` timestamp. The `in_combat` state derives from the `COMBAT_LOG` topic emitted by `mmorpg_server.hpp:230`; `performing` derives from an audio transport state in the DAW host.

### 3.3 Text and voice

Text is **available today** through the `user_message` and `ai_response` SocketIO events (`chat_api.py:90-130`) and the short-term history buffer of 20 turns (`dialogue_manager.py:37`). **Planned** hub text adds room scoping: a `hub_id` field in the message payload, and a per-room history that reuses the SQLite `conversations` schema with `session_id` set to `hub_id` plus a turn range.

Voice is **not implemented today**. No audio capture, codec, or WebRTC dependency appears in any file under `nextverse-platform`. **Planned** voice design:

| Layer | Design | Rationale |
| :--- | :--- | :--- |
| Transport | An SFU-style relay deployed beside the existing `redis` service in `docker-compose.yml:110-118`, with media on a dedicated port added to the compose file | Keeps media separate from the AI Assistant port mapping at `docker-compose.yml:22-23` |
| Signaling | SocketIO room events extending `connect` and `disconnect` at `chat_api.py:79-88` | Reuses an authenticated channel already present |
| Mixing | The JUCE `AudioAppComponent` lifecycle in `MainComponent.h:25-27` is the local render endpoint | `getNextAudioBlock` already exists as the audio callback |
| Safety | Mute, deafen, and push-to-talk flags stored per `account_id` in `user_preferences` (`dialogue_manager.py:110-119`) | Requires no new preference schema |

Voice remains a 2026-launch milestone; alpha and beta hubs are text-only.

---

## 4. Moderation and Safety

The AI Assistant already implements a content gate that can be reused directly. `EthicalFramework.calculate_dess_score` normalizes four moral factors into a 0.0 to 1.0 score (`ai-assistant/core/ethical_framework.py:161-219`), and `validate_content` combines the DESS score with simulated ethical and LMCB scores at thresholds 0.4 and 0.5 (`:243-326`). The `social` context weights at `:92` raise harmlessness to 1.5 and privacy to 1.4, which is the correct default for hub chat.

**Planned** moderation stack:

1. **Pre-post filter:** every hub text message is passed through `validate_content(text, context="social")` before fan-out. A forbidden-topic match from `:97-115` is refused using the same response shape as `AIBrain.generate_response` at `ai_brain.py:397-425`.
2. **Avatar-state signal:** a participant whose `AvatarAdapter` state is `cautious_hostile` or `aggressive` (`AvatarAdapter.cpp:16-29`) is not automatically moderated; the state is a feature for downstream review, not a verdict.
3. **User actions:** warning, mute, kick, and ban, each recorded with actor, target, reason, and timestamp in a proposed `hub_moderation_actions` table.
4. **Trust tiers:** guild rank from `GuildRank` (`game_data.hpp:137-141`) supplies the first tier — LEADER and OFFICER can moderate temple and guild halls. Lobby and event arenas use platform moderators.
5. **Hard security gates:** the wildcard CORS policy in `chat_api.py:49-53` and the development secret at `:45` must be replaced before any hub is exposed beyond localhost; the same applies to `utils/config.py:34`.

The C2-VIR and C13-WARDEN posture is load-bearing here: no creator economy feature ships without a working mute, report, and audit path.

---

## 5. Creator Economy

No payment, subscription, wallet, or marketplace code exists in `nextverse-platform` today. The following design is therefore **Planned** and grounded only in existing structural primitives.

| Feature | Design | Existing primitive |
| :--- | :--- | :--- |
| Creator profile | One row per `account_id`, with stage slot, handle, and payout preference | `player_accounts.account_id` at `schema.sql:5` |
| Stage reservation | Time-boxed bookings in a proposed `hub_stages` table | DAW `Edit` session concept in `MainComponent.h:36` |
| Ticketed events | Event records referencing a stage and a capacity, with entitlement checked at join | Guild membership pattern in `schema.sql:82-92` |
| In-world goods | Creator items minted as ordinary `item_id` entries and granted through the existing inventory path | `G_ITEM_DEFINITIONS` in `nextverse-platform/mmorpg/src/game_data_store.cpp:12-23` and `addItemToInventory` in `database_manager.hpp:271-288` |
| Revenue share | Computed from event attendance and recorded, never embedded in client messages | Proposed `hub_events` and `audit_log` tables |
| Anti-fraud | Idempotency keys on every grant and payout, mirroring the transactional inventory model | `checkAndConsumeIngredients` uses `SELECT ... FOR UPDATE` at `database_manager.hpp:317-351` |

The creator economy deliberately reuses the game economy rather than introducing a second currency. A quantum crafting outcome from `nextverse-platform/mmorpg/include/quantum_vm_mock.hpp:39-52` can be issued as a creator reward item using the same `item_id` namespace.

---

## 6. Events and Discovery

### 6.1 Event propagation today

`MMORPG_Server::publishWorldEvent` (`mmorpg_server.hpp:68-82`) produces zone-prefixed topics formed from a zone ID, a dot, and a base topic, or `GLOBAL.`-prefixed topics for world-wide messages, on the PUB socket bound at `main.cpp:81`. `test_client.py:75-107` demonstrates a subscriber that reads both frames of each message. This is the event backbone a hub discovery service can consume.

### 6.2 Discovery design

**Planned** discovery index:

| Index dimension | Source |
| :--- | :--- |
| Zone proximity | `zone_id` in `PlayerData.h:41` and the five `G_ZONE_DEFINITIONS` entries at `game_data_store.cpp:37-41` |
| Guild affiliation | `getCharacterGuildInfo` at `database_manager.hpp:558-579` |
| Live activity count | `GLOBAL` and zone-prefixed PUB topics from `mmorpg_server.hpp:68-82` |
| Creator ranking | `POST /execute` on `quantum-master.py:58-61`, used to rank candidate hubs by a reproducible score rather than by client-side ordering |
| Ethical safety | DESS score from `ethical_framework.py:161-219`, so unsafe rooms rank below safe ones |

A hub card returned by discovery carries `hub_id`, `hub_type`, `zone_id`, `participant_count`, `guild_id` when applicable, and a `rank_score`. The React navigation in `App.tsx:108-115` gains one new route for the hub directory; existing routes remain unchanged.

### 6.3 Live events

A live event is a scheduled `GLOBAL.EVENT_*` publication plus a room reservation. The `SIMULATION_UPDATE` topic already emitted by `handle_players_simulation` (`mmorpg_server.hpp:327-330`) is the template: it demonstrates event text plus a context payload. **Planned** event lifecycle is scheduled, announced, joined, active, and archived, with the archived record written to the proposed `hub_events` table.

---

## 7. Integration Points

| Target | Concrete interface in code today | Hub usage |
| :--- | :--- | :--- |
| MMORPG | `ENTER_WORLD` and `ATTACK` commands in `nextverse-platform/mmorpg/include/mmorpg_server.hpp:177-262`; guild methods in `database_manager.hpp:436-626` | Derive `in_combat` presence, gate temple halls by `GuildRank`, and place hub rooms in existing zones |
| Game engine | TCP JSON command protocol in `nextverse-platform/ai-assistant/core/game_engine_client.py:23-61`, targeting `GAME_ENGINE_HOST` and port 8081 by default (`utils/config.py:65-67`) | Move AI facilitator avatars in a hub; commands use `avatar_id`, `action`, and `target_x/y/z` |
| AI assistant | SocketIO events at `chat_api.py:79-130`, HTTP `POST /contextual_query` at `:142-180`, `AIBrain.process_game_context_query` at `ai_brain.py:291-321` | Facilitator dialogue, room chat, object inspection inside a hub |
| Quantum VM | `GET /health` and `POST /execute` at `nextverse-platform/quantum-vm/quantum-master.py:55-71` | Ranking and batch aggregation of hub telemetry; the simulated nature of the boost is documented in `quantum-vm/README.md:5-7` |
| DAW | JUCE `AudioAppComponent` lifecycle and Tracktion `Engine`/`Edit` at `daw/source/NextVerseDAW/Source/MainComponent.h:17-46`; conceptual track/transport classes per `daw/conceptual_logic/README.md:9-15` | Local audio endpoint for jam rooms; real shared audio requires the voice milestone in Section 3.3 |
| Quantum avatars | `AvatarAdapter::adaptBehavior` and its state table at `quantum_avatars/src/AvatarAdapter.cpp:12-46` | Visual reactivity for avatars in hubs, including `neutral_friendly` greetings and `passive_safe` behavior in safe zones |
| Service deployment | `docker-compose.yml:3-131` on the `nextverse-net` bridge | Host hub services beside the eight existing services; keep Redis at `:110-118` as the presence cache |
| Web client | `web-ui/src/App.tsx:176-183` routes and the static cards at `MMORPGWorld.tsx:4-15` and `DAWStudio.tsx:4-15` | Replace static cards with live hub and stage views |

---

## 8. Data Model

### 8.1 Reused tables

No existing table is modified. The hub layer reads `player_accounts` and `characters` (`schema.sql:4-29`), `guilds` and `guild_members` (`:72-92`), `character_inventories` (`:37-44`), and the AI Assistant's `conversations`, `user_preferences`, and `conversation_embeddings` tables (`dialogue_manager.py:99-134`).

### 8.2 Proposed tables

| Table | Key columns | Notes |
| :--- | :--- | :--- |
| `hubs` | `hub_id` primary key, `hub_type`, `name`, `zone_id`, `owner_account_id` references `player_accounts(account_id)`, `guild_id` references `guilds(guild_id)` nullable, `capacity`, `is_active` | `hub_type` values are the six typologies in Section 1. `zone_id` uses the keys from `game_data_store.cpp:37-41`. |
| `hub_presence` | `account_id`, `character_id`, `hub_id`, `presence_state`, `socket_sid`, `last_seen` | `socket_sid` is the value emitted at `chat_api.py:83`; `last_seen` aligns with `PlayerData::lastSeen` at `PlayerData.h:35`. |
| `hub_memberships` | `hub_id`, `account_id`, `role`, `joined_at` | `role` values `owner`, `moderator`, `member`, mirroring `GuildRank` semantics at `game_data.hpp:137-141`. |
| `hub_stages` | `stage_id`, `hub_id`, `creator_account_id`, `starts_at`, `ends_at`, `capacity`, `status` | Backs creator bookings and the Tracktion `Edit` concept at `MainComponent.h:36`. |
| `hub_jam_sessions` | `session_id`, `hub_id`, `stage_id`, `tracktion_edit_id`, `tempo_bpm`, `started_at` | `tracktion_edit_id` is the identifier associated with the `Edit` object at `MainComponent.h:36`. |
| `hub_events` | `event_id`, `hub_id`, `topic`, `scheduled_at`, `closed_at`, `attendance`, `rank_score` | `topic` follows the `GLOBAL.` prefix convention at `mmorpg_server.hpp:74`. |
| `hub_messages` | `message_id`, `hub_id`, `session_id`, `account_id`, `body`, `dess_score`, `is_compliant`, `created_at` | `dess_score` and `is_compliant` come from `validate_content` at `ethical_framework.py:318-326`. |
| `hub_moderation_actions` | `action_id`, `hub_id`, `actor_account_id`, `target_account_id`, `action_type`, `reason`, `created_at` | Provides the audit trail required by Section 4. |

All tables use parameterized access through the same libpqxx pattern already used in `database_manager.hpp`, and every foreign key uses `ON DELETE CASCADE` or `ON DELETE SET NULL` consistently with `schema.sql:39` and `:75`.

---

## 9. Phased Roadmap and Acceptance Criteria

### Phase A — Alpha: one working hub

Goal: a single text-only lobby with real identity and real moderation.

| Step | Work | Verify |
| :--- | :--- | :--- |
| A1 | Create the `hubs`, `hub_presence`, `hub_memberships`, and `hub_messages` tables in a new idempotent migration file under `nextverse-platform/mmorpg/sql/`, following the `CREATE TABLE IF NOT EXISTS` style of `schema.sql:4-92`. | Applying the migration twice leaves row and index counts unchanged. |
| A2 | Require `user_id` and `session_id` on `user_message`, replacing the defaults at `chat_api.py:104-106`. | A message without both fields receives an `error_response` from the handler at `:108-111`. |
| A3 | Add `hub_id` to the SocketIO message payload and persist it through `DialogueManager.store_conversation` at `dialogue_manager.py:184-241`. | A reconnect with the same `session_id` retrieves the hub-scoped history via `get_conversation_history` at `:244-282`. |
| A4 | Call `EthicalFramework.validate_content(text, context="social")` before emitting `ai_response` from `chat_api.py:127`. | A forbidden-topic message produces a refusal and no normal `ai_response`. |
| A5 | Register one hub route in `App.tsx:176-183` and replace the static lobby text in `MMORPGWorld.tsx:4-15` with a live participant count. | Two browser sessions see the same count within one 5-second monitor interval (`App.tsx:92`). |

### Phase B — Beta: guild halls, events, and stages

Goal: persistent guild rooms, live event discovery, and a scheduled creator stage.

| Step | Work | Verify |
| :--- | :--- | :--- |
| B1 | Add `hub_stages`, `hub_events`, and `hub_moderation_actions`; gate temple hall joins with `getCharacterGuildInfo` (`database_manager.hpp:558-579`) and `GuildRank` (`game_data.hpp:137-141`). | A MEMBER cannot enter a LEADER-only hall; an OFFICER can. |
| B2 | Subscribe a hub service to the PUB socket at `tcp://localhost:5556` from `test_client.py:75`, and index `GLOBAL.EVENT_*` topics into `hub_events`. | An event published by `publishWorldEvent` (`mmorpg_server.hpp:68-82`) appears in discovery within 2 seconds. |
| B3 | Add a stage booking flow that reserves the Tracktion `Edit` hosted by `MainComponent.h:36`. | Two creators cannot book the same `stage_id` and time window; the second request is rejected. |
| B4 | Connect AI facilitators to `POST /contextual_query` (`chat_api.py:142-180`) with `event_type="hub_greet"` and a new object ID handled in `AIBrain.process_game_context_query` (`ai_brain.py:291-321`). | Joining a hub yields a contextual facilitator message rather than the unknown-event response at `ai_brain.py:317-321`. |
| B5 | Replace the wildcard CORS origins at `chat_api.py:49-53` with the explicit web-ui origin and load the Flask secret from the environment at `:45`. | A request from an unapproved origin is rejected. |
| B6 | Route discovery ranking through `POST /execute` on `quantum-master.py:58-61`. | Ranking results are identical for the same input and instance count, satisfying the simulated-determinism expectation in `quantum-vm/README.md:5-7`. |

### Phase C — 2026 launch: voice, creators, and scale

Goal: the complete hub platform on the service topology in `docker-compose.yml:3-131`.

| Step | Work | Verify |
| :--- | :--- | :--- |
| C1 | Implement the SFU-style voice relay described in Section 3.3 and add its port to `docker-compose.yml`. | Two clients exchange audio with the JUCE callback at `MainComponent.h:26` active and no audio on the AI Assistant port mapping at `docker-compose.yml:22-23`. |
| C2 | Put `hub_jam_sessions` into service with tempo and edit state; stream events through PUB topics from `mmorpg_server.hpp:68-82`. | A participant joining mid-session receives the current tempo before any audio frame. |
| C3 | Ship the creator payout ledger using `hub_events.attendance` and the idempotent inventory grant pattern at `database_manager.hpp:271-288`. | Replaying a payout event twice produces one ledger row and one inventory grant. |
| C4 | Enforce presence TTL against `PlayerData::lastSeen` (`PlayerData.h:35`) and `PlayerManager` disconnect handling (`PlayerManager.cpp:48-64`). | A client that disappears without a SocketIO disconnect is marked offline within the TTL. |
| C5 | Publish per-module health through the existing probe contract used by `ai-assistant/health.py`, `daw/health.py`, `game-engine/health.py`, `orchestrator/health.py`, and `mmorpg/health.py`. | The monitor surface in `App.tsx:131-140` reports the hub service as healthy or degraded. |
| C6 | Complete the audit path: every moderation action in `hub_moderation_actions` is replayable by actor, target, and timestamp. | An export of one month of actions contains no row with a null actor or reason. |

---

## 10. Risks and Mitigations

| Risk | Basis | Mitigation |
| :--- | :--- | :--- |
| Static UI is mistaken for a live feature | `MMORPGWorld.tsx:4-15` and `DAWStudio.tsx:4-15` are static cards | Phase A5 and B3 require live state before a hub feature is marked complete |
| Text identity remains anonymous | `chat_api.py:104-106` defaults `user_id` and `session_id` | Phase A2 removes the defaults in production |
| Unsafe rooms outrank safe rooms | No ranking logic exists today | Include `dess_score` from `ethical_framework.py:161-219` in the discovery score |
| Voice arrives before moderation | No voice code exists | Gate Phase C1 on Phase A4 and B1 passing |
| Audio backend is conceptual | `daw/conceptual_logic/README.md:7-15` states there is no real audio processing | Treat `MainComponent.h:17-46` as the only production audio seam until jam rooms are implemented |
| Quantum ranking is simulated | `quantum-vm/README.md:5-7` describes conceptual acceleration | Keep ranking deterministic and never present the boost as measured performance |
| Guild data has no command surface | Guild methods exist at `database_manager.hpp:436-626` but `mmorpg_server.hpp:177-268` parses no guild command | Phase B1 adds the command surface before temple halls enter the hub directory |
| Presence duplicates state | SocketIO SID at `chat_api.py:83` and `PlayerData::isConnected` at `PlayerData.h:34` are independent | Treat `hub_presence` as the join table and reconcile both sources on each heartbeat |

---

## 11. Explicit Current-versus-Planned Statement

**Exists today (verifiable in code):** SocketIO text messaging with connect and disconnect events; AI response generation with input and output ethical checks; a 20-turn short-term memory and three-table SQLite long-term store; a DESS scorer with a `social` context and forbidden-topic regexes; account, character, inventory, and guild tables with full guild CRUD; zone-scoped and global PUB topic conventions; a JUCE audio host with a Tracktion engine and edit; conceptual DAW data structures; a quantum task gateway with health and execute endpoints; a string-driven avatar behavior adapter; a six-route React navigation shell; an eight-service Docker topology; and one-line health probes for five modules.

**Planned (not present in code):** any `hubs`, `hub_presence`, `hub_stages`, `hub_jam_sessions`, `hub_events`, `hub_messages`, or `hub_moderation_actions` table; voice capture, codecs, or an SFU relay; room-scoped chat; creator profiles, bookings, payouts, or ticketing; a live event index; a hub directory route; and guild-gated temple halls. These are specified in Sections 3 through 9 and grounded in the cited surfaces, not in shipped implementations.

**Document relationship:** the authoritative design for world state, zones, combat, and persistence is `nextverse-platform/docs/MMORPG_UNIVERSE_DESIGN_2026.md`. This document owns hub typology, presence, moderation, creator economy, and discovery, and it consumes the MMORPG event topics and identity tables rather than redefining them.

---

## Appendix A — Primary Source Index

| Area | Files cited |
| :--- | :--- |
| Real-time API | `nextverse-platform/ai-assistant/api/chat_api.py`, `nextverse-platform/ai-assistant/main.py` |
| Assistant core | `nextverse-platform/ai-assistant/core/ai_brain.py`, `core/dialogue_manager.py`, `core/ethical_framework.py`, `core/game_engine_client.py`, `utils/config.py` |
| World | `nextverse-platform/mmorpg/include/mmorpg_server.hpp`, `include/PlayerData.h`, `include/game_data.hpp`, `src/PlayerManager.cpp`, `src/game_data_store.cpp`, `src/database_manager.hpp`, `sql/schema.sql`, `test_client.py` |
| Audio | `nextverse-platform/daw/source/NextVerseDAW/Source/MainComponent.h`, `nextverse-platform/daw/conceptual_logic/README.md` |
| Compute and avatars | `nextverse-platform/quantum-vm/quantum-master.py`, `nextverse-platform/quantum-vm/README.md`, `nextverse-platform/quantum_avatars/include/AvatarAdapter.h`, `nextverse-platform/quantum_avatars/src/AvatarAdapter.cpp` |
| Client and deployment | `nextverse-platform/web-ui/src/App.tsx`, `nextverse-platform/web-ui/src/components/MMORPGWorld.tsx`, `nextverse-platform/web-ui/src/components/DAWStudio.tsx`, `nextverse-platform/docker-compose.yml`, module-level `health.py` files |
| Vision documents | `README.md`, `nextverse-platform/README.md`, `beta files/custom_mmorpg_module_architecture.md` |
