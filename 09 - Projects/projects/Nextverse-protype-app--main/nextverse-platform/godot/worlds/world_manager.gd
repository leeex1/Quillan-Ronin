extends Node
## World manager: zone transitions, NPC spawning, quest state and
## persistent hub presence via the REST bridge.

@export var hub_url: String = "http://127.0.0.1:8085"
@export var character_id: String = "godot-hero"
@export var enable_hub_network: bool = true

var current_zone: String = "starting_zone_town_square"
var remote_players: Array = []
var quests: Node
var player_state: Node
var zone_transitions: int = 0
var _zone_http: HTTPRequest = null

const NPC_SEEDS := [
	{"name": "Guard Thompson", "definition_id": 1001, "zone": "starting_zone_town_square"},
	{"name": "Mysterious Merchant", "definition_id": 1002, "zone": "starting_zone_market"},
	{"name": "Forest Goblin Alpha", "definition_id": 2001, "zone": "starting_zone_forest_edge"},
	{"name": "Forest Goblin Bravo", "definition_id": 2001, "zone": "starting_zone_forest_edge"},
]

func _ready() -> void:
	quests = load("res://worlds/quest_system.gd").new()
	quests.name = "QuestSystem"
	add_child(quests)
	player_state = load("res://worlds/persistent_player.gd").new()
	player_state.name = "PersistentPlayer"
	add_child(player_state)
	player_state.load_saved()
	enter_zone(player_state.zone_id)

func enter_zone(zone_id: String) -> bool:
	current_zone = zone_id
	zone_transitions += 1
	if player_state != null:
		player_state.transition_to(zone_id)
	_report_zone_to_hub(zone_id)
	for child in get_children():
		if child.name.begins_with("NPC_"):
			child.queue_free()
	for seed in NPC_SEEDS:
		if seed.zone != zone_id:
			continue
		var npc: Node = CharacterBody3D.new()
		npc.set_script(load("res://worlds/npc.gd"))
		npc.name = "NPC_%s" % String(seed.name).replace(" ", "_")
		npc.set("display_name", seed.name)
		npc.set("definition_id", seed.definition_id)
		npc.set("zone_id", zone_id)
		add_child(npc)
	return true

func _report_zone_to_hub(zone_id: String) -> void:
	if not enable_hub_network:
		return
	if _zone_http == null:
		_zone_http = HTTPRequest.new()
		_zone_http.name = "ZoneTransitionRequest"
		add_child(_zone_http)
	var payload := {"character_id": character_id, "zone_id": zone_id}
	_zone_http.request(hub_url + "/api/hub/" + zone_id + "/enter",
		["Content-Type: application/json"], HTTPClient.METHOD_POST, JSON.stringify(payload))

func npc_count() -> int:
	var count: int = 0
	for child in get_children():
		if child.name.begins_with("NPC_"):
			count += 1
	return count

func migrate_quest(quest_id: int, target: int) -> bool:
	return quests.transition(quest_id, target)
