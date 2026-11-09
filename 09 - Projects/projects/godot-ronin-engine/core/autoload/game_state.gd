extends Node
## Ported from Wayward Ronin story.js - reputation, flags, endings, kills.
## Autoload: GameState

var rep: Dictionary = {"clan": 0, "rebel": 0, "town": 0}
var kills: Dictionary = {"clan": 0, "rebel": 0}
var flags: Dictionary = {}

func adjust(faction: String, amt: int) -> void:
	if not rep.has(faction): return
	rep[faction] = clampi(rep[faction] + amt, -100, 100)

func add_kill(faction: String) -> void:
	if kills.has(faction): kills[faction] += 1

func set_flag(k: String, v = true) -> void:
	flags[k] = v

func evaluate_ending() -> String:
	if kills["clan"] + kills["rebel"] >= 7 and rep["clan"] < 50 and rep["rebel"] < 50:
		return "butcher"
	if rep["clan"] >= 50: return "clan"
	if rep["rebel"] >= 50: return "rebel"
	if rep["town"] >= 35: return "town"
	return "road"

func reset() -> void:
	rep = {"clan": 0, "rebel": 0, "town": 0}
	kills = {"clan": 0, "rebel": 0}
	flags = {}
