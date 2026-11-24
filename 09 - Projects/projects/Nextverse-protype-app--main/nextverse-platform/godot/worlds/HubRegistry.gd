extends Node
## Phase 4: six hub typologies from
## nextverse-platform/docs/SOCIAL_HUBS_DESIGN_2026.md section 1.

const HUB_TYPES: Array[String] = [
	"Lobby",
	"Creator stages",
	"Live events",
	"Temple and guild halls",
	"DAW jam rooms",
	"AI-assisted gathering spaces",
]

const SCENE_PATH: String = "res://worlds/Hub.tscn"

func has_type(hub_type: String) -> bool:
	return HUB_TYPES.has(hub_type)

func count() -> int:
	return HUB_TYPES.size()

func all_types() -> Array[String]:
	return HUB_TYPES.duplicate()
