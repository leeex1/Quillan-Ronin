extends Node
## user://save.json - replaces Wayward localStorage `wayward-ronin-save-v1`.
## Autoload: SaveManager

const PATH := "user://save.json"
const VERSION := 1

func save_game() -> void:
	var data := {
		"version": VERSION,
		"rep": GameState.rep,
		"flags": GameState.flags,
		"kills": GameState.kills,
	}
	var f := FileAccess.open(PATH, FileAccess.WRITE)
	f.store_string(JSON.stringify(data))
	f.close()

func load_game() -> bool:
	if not FileAccess.file_exists(PATH): return false
	var f := FileAccess.open(PATH, FileAccess.READ)
	var data: Dictionary = JSON.parse_string(f.get_as_text())
	f.close()
	if data == null or data.get("version", 0) != VERSION: return false
	GameState.rep = data["rep"]
	GameState.flags = data["flags"]
	GameState.kills = data["kills"]
	return true

func has_save() -> bool:
	return FileAccess.file_exists(PATH)

func clear() -> void:
	if FileAccess.file_exists(PATH):
		DirAccess.remove_absolute(PATH)
