extends Node
## Lean port of Metamon-Survival-Rogue metamon.ts / parts.ts
## Autoload: MetamonData - extend via skills/port-metamon.SKILL.md

enum Element { FIRE, WATER, GRASS, ELECTRIC, PSYCHIC, DARK, NORMAL }

const ELEMENT_NAMES := ["Fire","Water","Grass","Electric","Psychic","Dark","Normal"]

const PART_TYPES := ["head","body","arms","legs","tail","weapon"]

# id -> {name, element, hp, atk, spd, parts}
const SPECIES := {
	"ember": {"name": "Ember", "element": 0, "hp": 40, "atk": 12, "spd": 5, "parts": 3},
	"aqua": {"name": "Aqua", "element": 1, "hp": 50, "atk": 9, "spd": 6, "parts": 3},
	"leaf": {"name": "Leaf", "element": 2, "hp": 55, "atk": 8, "spd": 4, "parts": 4},
}

func get_species(id: String) -> Dictionary:
	return SPECIES.get(id, {})

func element_of(id: String) -> int:
	return get_species(id).get("element", 6)
