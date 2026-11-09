# Skill: port-wayward
Port Wayward Ronin story/combat to Godot.

Source: `09 - Projects/projects/wayward-ronin/wayward-ronin/js/story.js`
Target: `core/autoload/game_state.gd` + `core/systems/combat_tuning.gd`

- Rep: clan / rebel / town (-100..100). Endings: clan>=50, rebel>=50, town>=35, kills>=7.
- Combat: 3-hit combo, heavy guard-break, hold-block, tap-parry, dodge i-frames.
- Dialogue: trees with choices + conditional `if` + side effects `do()`.
- Keep Three.js procedural rig out; use Godot CharacterBody2D/3D.
