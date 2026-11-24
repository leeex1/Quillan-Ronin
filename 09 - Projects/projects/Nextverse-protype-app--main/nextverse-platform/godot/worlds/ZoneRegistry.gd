extends Node
## Phase 4: zone catalogue grounded in
## nextverse-platform/docs/MMORPG_UNIVERSE_DESIGN_2026.md section 3.

const ZONES: Array[Dictionary] = [
	{"id": "starting_zone_town_square", "name": "Town Square", "tier": "starter", "exists_in_legacy": true},
	{"id": "starting_zone_market", "name": "Market District", "tier": "starter", "exists_in_legacy": true},
	{"id": "starting_zone_forest_edge", "name": "Forest Edge", "tier": "starter", "exists_in_legacy": true},
	{"id": "dungeon_level_1", "name": "Abandoned Mines - Level 1", "tier": "dungeon", "exists_in_legacy": true},
	{"id": "unknown_zone", "name": "The Void", "tier": "special", "exists_in_legacy": true},
	{"id": "dungeon_level_2", "name": "Abandoned Mines - Level 2", "tier": "dungeon", "exists_in_legacy": false},
	{"id": "nexus_atrium", "name": "Nexus Atrium", "tier": "civic", "exists_in_legacy": false},
	{"id": "guild_temple_1", "name": "Guild Temple 1", "tier": "guild", "exists_in_legacy": false},
	{"id": "live_event_arena", "name": "Live Event Arena", "tier": "event", "exists_in_legacy": false},
]

const DEFAULT_ZONE: String = "starting_zone_town_square"
const SCENE_PATH: String = "res://worlds/Zone.tscn"

func has_zone(zone_id: String) -> bool:
	for zone in ZONES:
		if zone.id == zone_id:
			return true
	return false

func zone_name(zone_id: String) -> String:
	for zone in ZONES:
		if zone.id == zone_id:
			return String(zone.name)
	return ""

func legacy_zone_count() -> int:
	var count: int = 0
	for zone in ZONES:
		if zone.exists_in_legacy:
			count += 1
	return count

func all_ids() -> PackedStringArray:
	var ids: PackedStringArray = PackedStringArray()
	for zone in ZONES:
		ids.append(zone.id)
	return ids
