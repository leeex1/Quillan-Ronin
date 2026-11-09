# Godot Ronin Engine - lean custom scaffold

Godot 4.x base + Editor Plugin + MCP bridge + Skills. Built to port
**Metamon-Survival-Rogue** (mechanics) and **Wayward Ronin** (story/combat)
into native Godot without copying web bloat. ~100KB source, no binaries.

## Open it
1. Godot 4.2+ -> Import -> select this folder (`project.godot`)
2. Enable plugin: Project Settings > Plugins > `Ronin AI` ✅
3. Restart editor. Dock `Ronin AI` appears on the right.
4. Press F5 - Main scene runs.

## Structure
- `addons/ronin_ai/` - editor plugin + AI dock + MCP TCP bridge (127.0.0.1:7777)
- `mcp/server.py` - stdlib-only MCP server (no pip needed). Run:
  `python mcp/server.py`  or wire as MCP server in your client.
- `skills/*.SKILL.md` - copy into `.opencode/skills/` or `.agents/skills/` to
  activate: scene-builder, debug, port-metamon, port-wayward
- `core/autoload/` - `GameState` (Wayward rep/flags/endings), `SaveManager`,
  `MetamonData` (Metamon stats/elements/parts)
- `core/systems/combat_tuning.gd` - Wayward combat constants (3-hit combo, parry, block)

## MCP tools exposed
`list_project, create_scene, add_node, set_property, save_scene, run_game, get_logs`
Godot plugin polls `mcp/queue.jsonl`. No sockets needed for MVP = robust on low space.

## Space note
No engine fork, no assets duplicated. Point your existing Godot binary at
this folder. To get Godot: https://godotengine.org/download (use standard
non-mono build, ~1GB, don't download template pack until you need export).

## Next steps (quality order)
1. Test open + run (5 min)
2. `python mcp/server.py` + plugin dock ping
3. Activate `scene-builder` skill, ask AI to build a test level
4. Port one Metamon + one Wayward dialogue tree as proof

Do not add `.godot/` to git - see `.gitignore`.
