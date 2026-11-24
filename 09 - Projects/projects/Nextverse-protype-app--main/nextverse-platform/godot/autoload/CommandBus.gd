extends Node
## Typed command channel replacing the legacy AICommandServer raw socket queue.

signal avatar_command(avatar_id: String, action: String, x: float, y: float, z: float)
signal session_event(event: String, payload: Dictionary)

var history: Array[Dictionary] = []

func publish_avatar_command(avatar_id: String, action: String, x: float, y: float, z: float) -> void:
	var command: Dictionary = {
		"avatar_id": avatar_id,
		"action": action,
		"x": x,
		"y": y,
		"z": z,
	}
	history.append(command)
	avatar_command.emit(avatar_id, action, x, y, z)

func clear() -> void:
	history.clear()
