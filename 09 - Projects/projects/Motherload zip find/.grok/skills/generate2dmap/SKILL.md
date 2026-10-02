---
name: generate2dmap
description: >
  Generate production-oriented 2D game maps with `imagine_text_to_image`: RPG/top-down maps,
  side-scroller parallax stages, tilemaps, layered raster maps, prop packs,
  collision zones, and walkable areas. Use when building browser games that
  need real map art (not pure code-drawn tiles), layered props, or map
  collision metadata. Triggers on "map", "level", "stage", "tilemap",
  "overworld", "dungeon", "side scroller background", "prop pack", "2D map".
metadata:
  short-description: "2D game maps: layered art, props, collision metadata"
user-invocable: false
---

# Generate2dmap

## App-builder / Grok environment

| Item | Value |
| --- | --- |
| Skill dir / scripts | `.grok/skills/generate2dmap/`, run as `python3 .grok/skills/generate2dmap/scripts/<script>.py …` |
| Image tools | `imagine_text_to_image` / `imagine_image_to_image` (path-based; see **`imagine`**); inspect output with `read_file` on the PNG path (not Codex view_image) |
| Generated image path | `imagine_text_to_image` → sandbox `file_path` → copy into `assets/map/`; Pillow is preinstalled |
| Default `engine_target` | `raw_canvas` or `Phaser` for this TanStack browser sandbox — only use Godot/Unity/Tiled when the user explicitly wants those exports |
| Related skills | **`generate2dsprite`** (character/FX sprites; prop packs still use this skill's extract script), **`building-games`**, **`imagine`** |

## Decide the pipeline first

Build the smallest playable map bundle that satisfies the game: choose a
product-level `map_mode`, then the lower-level axes (`visual_model`,
`runtime_object_model`, `collision_model`, `engine_target`).

- `tile_mode` — editable tile/grid maps: Pokemon-like routes, top-down RPG towns, platformer tilemaps, or any project already on Tiled/LDtk/Godot/Unity/Phaser tilemaps.
- `scene_mode` — foundation base plus separate props: tower defense, survivors-like arenas, cozy top-down showcase maps.
- `side_scroll_mode` — parallax side-scroller stages: Mega Man-like, action platformers, Metroidvania rooms, runners, brawlers.
- `grid_mode` — rule-heavy grids: tactical RPGs, factory/automation, board/card battlers, build grids.
- `room_chunk_mode` — modular rooms/chunks: roguelike dungeons, Metroidvania networks, procedural assembly.
- `baked_scene_mode` — explicitly flat, non-playable scenes only: title/menu screens, battle backdrops, visual-novel scenes, concept art.

Use user-specified parameters when present; otherwise infer the lightest playable
pipeline from the existing game, camera, collision needs, map scale, and editing
needs. When mode and axes disagree, the mode's playable/editable contract wins.
Genre routing, per-mode axis defaults, presets, and the escalation heuristic are in
`references/map-strategies.md` — read it whenever the choice is not obvious.

**A playable map is never one baked image.** For any request implying a playable
map, level, stage, room, prototype, or engine scene, the deliverable must expose
gameplay geometry and objects as separate layers, props, tile/object data,
collision, zones, or engine-native scene nodes. A baked image may be a background,
reference, or preview artifact — never the runtime map — unless the user
explicitly asked for a flat background only.

**Scenes and maps only.** Do not generate character, enemy, boss, NPC, player,
projectile, or animation sprites here; those belong to `$generate2dsprite`. Maps
carry scene hooks (spawn markers, patrol/encounter zones, arena entrances, gates,
exits, camera triggers) as **metadata**, not as drawn art.

## Art comes from image generation, and you write the prompts

- `imagine_text_to_image` is the default art source for base maps, parallax plates,
  references, prop sheets, and tileset art. Default `art_style` is `clean_hd`
  (hand-painted HD, sharp readable shapes, low texture noise, no chunky pixels);
  use `pixel_inspired` or `retro_pixel` only when asked.
- **Write every creative prompt yourself.** Scripts may assemble, slice,
  chroma-key, crop, validate, compose previews, and emit JSON/engine files — never
  write creative prompts or draw final art. Procedural/placeholder art only when
  the user explicitly asks for placeholders, fixtures, debug maps, or scaffolding.
  With a tile engine target, generate the tileset art first, then script only the
  layers, collision, zones, and scene wiring.
- Save each prompt beside its asset as `<asset>.prompt.txt` (or an explicit
  manifest field) whenever the run creates new visual assets.
- **A reference handoff is a file path, not a sentence.** To build on an earlier
  image, pass its sandbox `file_path` to `imagine_image_to_image` (and `read_file`
  it so you can see it), then name the concrete features to preserve: camera
  framing, horizon, road/water shapes, terrain boundaries, entrances/exits,
  landmarks. A filename, "based on the map", or the image merely being visible in
  conversation is **not** a handoff — stop and pass the path.

## Keep runtime objects out of the base layer

The first generated base/background/foundation image may hold only stable, non-interactive
foundation art — ground material, paths, roads, water, cliffs, floor patterns, lane
markings and build pads; for side views sky, far/mid scenery, silhouettes, atmosphere; for
tilemaps tileset art as editable layers. It must **not** contain tall props, buildings,
trees, rocks, crates, signs, doors, gates, pickups, chests, checkpoints, hazards, traps,
turrets, ladders, foreground occluders, destructibles, actors, enemies, NPCs, UI, labels,
or anything needing collision, interaction, reuse, y-sorting, animation, or its own render
order — regenerate a foundation-only base, or demote such an image to a reference
artifact.

## Reference mockups are checkpoints, not deliverables

Dressed references (top-down) and stage references (side-view) plan placement in-world:
natural game-world objects or subtle blockout geometry, at most **9 distinct visible
object candidates** (repeats count once, then recur in placement metadata), **no
annotation graphics** (circles, arrows, outlines, labels, text, callouts, legends,
measurement lines), and no non-visual metadata — spawns, triggers, patrol hints, camera
bounds are written later as scene hooks.

**Having generated one, do not stop there.** Continue through
`references/object-production-gate.md`: re-`read_file` both images, build the object
list, generate the final separate objects, write placement / collision / scene-hook
metadata, compose the QA preview. Reference-only output is an incomplete run unless
the user explicitly asked for a concept image.

## Depth for the pipeline you picked — open these before producing assets

- Layered raster maps → `references/layered-map-contract.md` (layer types, base and
  prop prompt patterns, prop metadata, render order, collision, QA checklist).
- `side_scroll_mode` → `references/side-scroll-stages.md`: the `stage_canvas`
  decision, the named scenery-only parallax layers, and the mandatory in-world
  stage reference before any platform/object work.
- Any prop or scene-object generation → classify each object first, then follow
  `references/prop-pack-contract.md`: only compact props may share a square prop
  pack; platforms, floors, bridges, gates, buildings and anything collision-aligned
  go one-by-one, as a platform strip, a custom wide pack, or tile/object layers.
- Parameters, the step-by-step workflow, and the `extract_prop_pack.py` /
  `compose_layered_preview.py` commands → `references/pipeline.md`.
- Deliverable lists per pipeline and the validation checklist →
  `references/deliverables.md`; run both before calling a map done.
