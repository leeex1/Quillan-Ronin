# Skill: port-metamon
Port Metamon-Survival-Rogue mechanics to Godot.

Source: `09 - Projects/projects/Metamon-Survival-Rogue/` (React/Canvas)
Target: `core/autoload/metamon_data.gd`

- Elements (7): Fire, Water, Grass, Electric, Psychic, Dark, Normal
- Parts (6): head, body, arms, legs, tail, weapon -> Godot Resources
- Keep stats small. No 30-sprite dump; re-skin later.
- Survival wave = spawn timer + object pool in Godot.
