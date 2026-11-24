extends Node3D
## Minimal Phase 0 world: proves the autoload graph loads and the runtime boots.

func _ready() -> void:
	print("[TestWorld] ready; tier=", PerformanceDirector.current_tier)
