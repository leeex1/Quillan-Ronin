extends Node3D
## Generic social hub root; hub_type must be one of HubRegistry.HUB_TYPES.

@export var hub_type: String = "Lobby"
@export var zone_id: String = "nexus_atrium"

func initialize(type: String, in_zone: String) -> bool:
	var registry = load("res://worlds/HubRegistry.gd").new()
	var valid: bool = registry.has_type(type)
	if valid:
		hub_type = type
		zone_id = in_zone
		name = type
	registry.free()
	return valid
