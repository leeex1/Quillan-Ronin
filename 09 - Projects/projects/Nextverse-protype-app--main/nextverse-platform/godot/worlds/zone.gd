extends Node3D
## Generic zone root. Zone identity comes from ZoneRegistry, not a duplicate list.

@export var zone_id: String = "starting_zone_town_square"

func initialize(id: String) -> bool:
	var registry = load("res://worlds/ZoneRegistry.gd").new()
	var valid: bool = registry.has_zone(id)
	if valid:
		zone_id = id
	name = registry.zone_name(id)
	registry.free()
	return valid
