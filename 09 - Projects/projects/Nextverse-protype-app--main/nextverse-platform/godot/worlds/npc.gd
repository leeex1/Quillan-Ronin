extends CharacterBody3D
## NPC with seeded identity and simple patrol behaviour.

@export var display_name: String = "Guard Thompson"
@export var definition_id: int = 1001
@export var zone_id: String = "starting_zone_town_square"

var dialogue_seed: String = ""
var _origin: Vector3 = Vector3.ZERO
var _elapsed: float = 0.0

func _ready() -> void:
	_origin = position
	dialogue_seed = "I am %s, definition %d, watching %s." % [display_name, definition_id, zone_id]

func _physics_process(delta: float) -> void:
	_elapsed += delta
	position.x = _origin.x + sin(_elapsed * 0.6) * 1.5
	position.z = _origin.z + cos(_elapsed * 0.4) * 1.0

func interact() -> String:
	return dialogue_seed
