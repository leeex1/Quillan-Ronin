extends Node
## Persistent player state: zone, position, quests and entitlements snapshot.
## Stored locally in user:// and mirrored to the hub REST bridge.

const SAVE_PATH := "user://nextverse_player.json"

@export var character_id: String = "godot-hero"
@export var zone_id: String = "starting_zone_town_square"
@export var hub_url: String = "http://127.0.0.1:8085"

var position: Vector3 = Vector3(0.0, -0.5, 4.0)
var quest_snapshot: Dictionary = {}
var entitlement_snapshot: Dictionary = {}

func to_dict() -> Dictionary:
	return {
		"character_id": character_id,
		"zone_id": zone_id,
		"position": {"x": position.x, "y": position.y, "z": position.z},
		"quests": quest_snapshot.duplicate(),
		"entitlements": entitlement_snapshot.duplicate(),
	}

func from_dict(data: Dictionary) -> void:
	character_id = String(data.get("character_id", character_id))
	zone_id = String(data.get("zone_id", zone_id))
	var p: Dictionary = data.get("position", {})
	position = Vector3(float(p.get("x", 0.0)), float(p.get("y", -0.5)), float(p.get("z", 0.0)))
	quest_snapshot = data.get("quests", {}).duplicate()
	entitlement_snapshot = data.get("entitlements", {}).duplicate()

func save() -> bool:
	var file := FileAccess.open(SAVE_PATH, FileAccess.WRITE)
	if file == null:
		return false
	file.store_string(JSON.stringify(to_dict()))
	file.close()
	return true

func load_saved() -> bool:
	if not FileAccess.file_exists(SAVE_PATH):
		return false
	var file := FileAccess.open(SAVE_PATH, FileAccess.READ)
	if file == null:
		return false
	var parsed: Variant = JSON.parse_string(file.get_as_text())
	file.close()
	if parsed == null or not parsed is Dictionary:
		return false
	from_dict(parsed)
	return true

func transition_to(new_zone: String) -> bool:
	if new_zone.is_empty():
		return false
	zone_id = new_zone
	position = Vector3(0.0, -0.5, 4.0)
	return save()
