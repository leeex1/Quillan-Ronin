# Map Pipeline: parameters, workflow, and scripts

Read this once the `map_mode` is chosen and you are about to produce assets.

## Parameter Contract

User-facing parameters may be stated in natural language:

- `map_mode`: tile_mode | scene_mode | side_scroll_mode | grid_mode | room_chunk_mode | baked_scene_mode
- `map_kind`: overworld | town | dungeon | shrine | arena | battle_bg | side_scroller | side_view_action | platformer | metroidvania | brawler | tower_defense | survivors_like | tactical | factory | card_board | room_chunk
- `visual_model`: baked raster | layered raster | tilemap | layered tilemap | parallax
- `size`: pixel dimensions, tile dimensions, or camera-relative size
- `stage_canvas`: exact pixel dimensions and aspect ratio for side-scroll/parallax layers, references, and previews
- `perspective`: top-down | 3/4 top-down | side-view | isometric-like
- `art_style`: clean_hd | pixel_inspired | retro_pixel | hand_painted | project-native
- `visual_asset_source`: imagine_text_to_image | existing_assets | procedural_placeholder
- `collision_precision`: none | coarse | precise | tile | walkmesh
- `prop_generation`: none | one_by_one | prop_pack_2x2 | prop_pack_3x3 | prop_pack_4x4 | platform_strip_1x3 | platform_strip_1x4 | custom_wide_pack
- `output_format`: PNG only | layered preview | manifest JSON | engine-native map data

When unspecified:

- Use `imagine_text_to_image` as the visual asset source.
- Infer `map_mode` from genre and editing needs before selecting lower-level axes.
- Use `tile_mode` for Pokemon-like, top-down RPG, monster-taming, editor/grid-perfect, or tilemap requests.
- Use `scene_mode` for tower defense, survivors-like, cozy/top-down showcase maps, and base-map-plus-props requests.
- Use `side_scroll_mode` for side-scrollers, platformers, runners, side-view action, brawlers, Metroidvania side rooms, Mega Man-like, Castlevania-like, Contra-like, and parallax background requests.
- For `side_scroll_mode`, choose a canonical `stage_canvas` before image generation. Use the project camera/viewport aspect when available; otherwise default to a 16:9 side-scroller canvas such as `1536x864`. All primary parallax plates, stage references, and previews must preserve this same size/aspect.
- Use `grid_mode` for tactical RPGs, factory/automation maps, board/card battlers, build grids, and terrain-cost maps.
- Use `room_chunk_mode` for modular rooms, roguelike rooms, procedural room assembly, or Metroidvania room-chunk planning.
- Use `baked_scene_mode` only for non-playable visual scenes or explicitly flat images.
- Use `baked_raster + coarse_shapes` only for battle backgrounds, title/menu scenes, cutscenes, decorative backdrops, non-playable previews, or when the user explicitly asks for a single flat image.
- Use `layered_raster + y_sorted_props + precise_shapes` for top-down RPG exploration with tall props, occlusion, interactables, or reusable props; the base must be foundation-only and the props/interactables must remain separate.
- Use `tilemap` or `layered_tilemap` only when the engine/editor already uses tiles or the user asks for editable tiles; do not flatten gameplay objects into one background image.
- Use `parallax_layers + platform_objects + interactive_scene_objects + scene_hooks + precise_shapes` for playable side-view scrolling stages, platformers, runners, shooters, and horizontal action scenes; the parallax/background image is scenery-only and is not the runtime map by itself.
- Use square prop packs only when 4 or more compact small/medium static props share one style and fit comfortably inside equal square cells.
- Use one-by-one, platform strips, tile/object layers, or custom wide packs for hero props, buildings, gates, irregular large props, wide/tall props, platforms, terrain chunks, bridges, walls, ladders, long hazards, animated props, or props needing strong identity or collision alignment.
- Use `clean_hd` for generated exploration maps unless the project or user asks for pixel art. This means clean hand-painted top-down 2D RPG game map, HD game asset style, sharp readable terrain shapes, low texture noise, and no chunky pixels.
- Use `pixel_inspired` only when the user wants a pixel-adjacent look without retro chunkiness.
- Use `retro_pixel` only when the user explicitly asks for 16-bit, retro JRPG, or classic pixel-art maps.

## Workflow

1. Inspect the target game.
   - Find camera size, map dimensions, coordinate system, render order, asset loading, collision support, zone data, and existing map formats.
   - Preserve the engine's existing style and data contracts.

2. Choose the pipeline axes.
   - Choose `map_mode` first. Use the genre routing table in `map-strategies.md` when the user describes a game type instead of a technical map format.
   - Select `visual_model`, `runtime_object_model`, `collision_model`, and `engine_target`.
   - If the request is for a playable map, stage, level, room, prototype, or game scene, choose a pipeline with explicit runtime objects. Do not downgrade to `baked_raster` unless the user asked for a background-only image.
   - If the request implies a playable side-view scrolling/action stage, such as a side-scroller, platformer, runner, shooter, brawler, scrolling combat stage, Megaman-like stage, Castlevania-like stage, or Contra-like stage, lock the map pipeline to `parallax_layers + platform_objects + interactive_scene_objects + scene_hooks + precise_shapes` unless the engine already requires a tilemap.
   - Select `art_style`. Prefer readable gameplay shapes over decorative texture density.
   - Select `visual_asset_source`. Default to `imagine_text_to_image`; use `existing_assets` only when the project already has suitable art; use `procedural_placeholder` only when explicitly requested.
   - Treat `hybrid` as a result of combining axes, not as a primary category.

3. Produce assets.
   - Write the creative prompts manually and use built-in `imagine_text_to_image` for visible map art unless the user explicitly chose existing assets or procedural placeholders.
   - For baked raster maps, generate one background with built-in `imagine_text_to_image`, or edit/use an existing image when supplied, then add optional collision/zones metadata.
   - For playable or editable layered maps, generate a foundation-only base/background first. The base must not contain runtime-controlled props, interactables, hazards, doors, gates, pickups, actors, or foreground occluders. If it does, regenerate or demote it to a reference artifact.
   - For layered raster maps, generate a ground-only/foundation-only base map first. Then perform the visual reference handoff and generate an in-world dressed reference mockup from the visible base before making final props and placements.
   - For tilemaps, generate or reuse tileset art first, then follow the engine/editor format for layers, objects, collision, and scene files. Do not script-draw the tileset as the final art source, and do not flatten object layers into a single runtime image.
   - For `grid_mode`, generate or reuse grid/tileset visual art first, then write cell metadata such as walkable/buildable flags, move cost, terrain effects, resource nodes, and object layers.
   - For `room_chunk_mode`, define chunk dimensions, exits, connection sockets, collision contract, and spawn/trigger metadata before final art assembly. Chunks must be reusable and validated at their seams.
   - For playable side-view scrolling/action stages, define the canonical `stage_canvas` before generating art. Generate named scenery-only parallax layers first: `sky`, `far_bg`, `mid_bg`, `near_bg`, and optional `foreground_overlay`. Every primary parallax layer must use the same pixel dimensions, aspect ratio, camera framing, horizon line, and top-left anchor as the `stage_canvas`; do not accept mismatched image sizes that require guesswork to stack. Do not treat one full-width background image as a complete `side_scroll_mode` background stack unless the user explicitly asks for a flat/non-parallax background. These parallax passes must not contain playable foreground platforms, walkable floors, terrain chunks, hazards, pickups, doors, gates, checkpoints, crates, fences, spikes, or other runtime objects. Then perform the visual reference handoff and generate an in-world stage reference mockup that visually places up to 9 distinct intended platform/object candidates before generating final separate scene objects and metadata.
   - If a side-view background already contains collidable-looking foreground geometry, walkable floors, or reusable gameplay props, reject it as a runtime background and regenerate a cleaner scenery-only background before continuing.
   - Treat the reference mockup as a checkpoint, not a deliverable. Do not stop after generating it. After the relevant `dressed-reference` or `stage-reference` exists, inspect it and continue into the post-reference object production gate (`object-production-gate.md`).
   - Do not present a rerunnable script that creates the whole art pack as the main solution unless the user asked for procedural placeholder art.

4. Build metadata.
   - Store prop placement, player spawns, actor spawn marker metadata, interactable scene objects, blockers, walk bounds, encounter zones, exits, camera bounds, and triggers as structured data.
   - For `grid_mode`, store grid dimensions, cell size, tile ids, terrain types, walkable/buildable flags, movement cost, collision, resource nodes, and object/entity slots.
   - For `room_chunk_mode`, store chunk id, size, entrances/exits, connection sockets, collision, spawn markers, camera bounds, and validation hints for seam alignment.
   - For `side_scroll_mode`, store `stage_canvas`, parallax layer source size, display size, anchor, render order, scroll factors, loop/repeat policy, camera bounds, platform collision, hazards, exits, checkpoints, and actor spawn marker metadata.
   - Keep collision independent from pixels unless the target engine explicitly uses tile collision.

5. Validate and preview.
   - Compose a flattened preview for layered maps.
   - Validate image sizes, alpha channels, prop pack extraction metadata, JSON parseability, and critical walkability points when collision matters.
   - For `side_scroll_mode`, reject or normalize mismatched primary parallax layer sizes before runtime integration. The stage reference and QA preview must match `stage_canvas` exactly. Deterministic resizing/cropping/padding is allowed only as a normalization step on generated art, not as a way to invent missing art.

## Prop Generation Rules

Use `$generate2dsprite` for reusable transparent props and visible scene objects, but the agent must write the prop prompt itself using the selected map `art_style`. Do not use a script to generate the creative prompt. For `clean_hd` maps, explicitly request clean hand-painted HD 2D game assets and explicitly forbid pixel art. For `pixel_inspired`, request clean modern pixel-art-inspired props without retro chunkiness. For `retro_pixel`, request 16-bit or retro JRPG pixel art.

Before any prop/object image generation, classify each visible runtime object from the reference mockup:

- `compact_prop`: small/medium, roughly square or vertical, decorative or simple blocker, no exact alignment requirement
- `wide_or_long_object`: expected aspect ratio wider than about `1.6:1`, such as platforms, floor pieces, bridges, wall runs, fence rows, long traps, long signs, pipes, rails, ledges, or roads
- `tall_or_large_object`: expected aspect ratio taller than about `1.6:1` or visually dominant, such as large trees, gates, towers, buildings, banners, doors, statues, or boss-room props
- `collision_bearing_object`: must line up with collision, walkable edges, build pads, doors, checkpoints, gates, hazards, or engine editor handles
- `tileset_or_strip_piece`: should repeat seamlessly or assemble from left/middle/right caps, corners, slopes, tops, sides, or tile pieces

Generation strategy is determined by that classification:

- Only `compact_prop` objects may use square `prop_pack_2x2`, `prop_pack_3x3`, or `prop_pack_4x4`.
- Do not put `wide_or_long_object`, `tall_or_large_object`, `collision_bearing_object`, or `tileset_or_strip_piece` into square prop packs.
- Use `one_by_one` for important, large, tall, irregular, identity-sensitive, or collision-aligned objects.
- Use `platform_strip_1x3` or `platform_strip_1x4` for repeatable floors/platforms: left cap, middle repeat, right cap, plus optional corner/slope/end variant.
- Use `custom_wide_pack` only for several similar wide objects that share one category and can use wide cells such as `768x256`, `1024x384`, or another explicit non-square cell size.
- Never mix compact decorative props with platforms, terrain chunks, gates, doors, hazards, or other collision-critical objects in the same generated sheet.
- If a square pack fails because a wide/tall object touches an edge, do not retry the same square pack with looser QC. Reclassify that object and regenerate it one-by-one, as a platform strip, as a custom wide pack, or as tile/object-layer art.

Choose the generation shape deliberately:

- `one_by_one`: safest for large, important, animated, or irregular props.
- `prop_pack_2x2`: 4 related compact props, safest square batch size.
- `prop_pack_3x3`: 9 compact small/medium props, good quality/time tradeoff.
- `prop_pack_4x4`: 16 very simple compact small props; fastest but most likely to drift or touch edges.
- `platform_strip_1x3`: repeatable non-actor platform/floor strip with left cap, middle repeat, and right cap.
- `platform_strip_1x4`: repeatable non-actor platform/floor strip with left cap, middle repeat, right cap, and one extra slope/corner/end variant. This is not an animation-frame format and must not be used for characters, enemies, creatures, NPCs, summons, or animated body assets.
- `custom_wide_pack`: several related wide objects using explicit wide cells, not square cells.

Prop packs save image-generation calls and prompt overhead, but reduce per-prop control. Use square prop packs for rocks, shrubs, barrels, small signs, lamps, crates, floor ornaments, plants, and repeated compact environmental props. Do not use square prop packs for buildings, gates, trees with wide canopies, bridges, platforms, floors, walls, ladders, long fences, long hazards, character-like statues, hero objects, or anything that must be pixel-perfect or collision-aligned.

For layered maps with generated props, prefer this in-world reference mockup pipeline:

1. Generate `assets/map/<name>-base.png` as ground-only terrain.
2. Hand the base to `imagine_image_to_image` as a real reference: pass its sandbox `file_path`. Also `read_file` it so you can see it; do not rely on a path string or prompt text as the reference.
3. In the dressed-reference prompt, explicitly say: use the provided base image as the visual reference, preserve its camera/framing/dimensions/terrain/road/water/boundaries, and generate an in-world dressed reference mockup.
4. The dressed reference must show proposed props as natural game-world objects placed on the base. It must not contain circles, arrows, outlines, labels, text, callouts, legends, highlighted boxes, or other annotation graphics.
5. The dressed reference should contain at most 9 distinct visible prop/object candidates unless the user explicitly asks for more. Prefer the objects that will become final generated props, collision blockers, interactables, or occluders.
6. Generate `assets/map/<name>-dressed-reference.png` from the visible base. Treat this as a reference mockup, not the final runtime map.
7. Generate one-by-one props or a prop pack based on the dressed reference.
8. Place extracted props over the original base and compose a flattened preview.
9. Validate that base, dressed reference, and preview dimensions match.

## Scripts

Use the extract script after generating a solid-magenta prop sheet (built-in magenta cleanup is enough for most sheets):

```bash
python3 .grok/skills/generate2dmap/scripts/extract_prop_pack.py \
  --input assets/props/raw/<name>-sheet.png \
  --rows 3 --cols 3 \
  --labels <comma-separated-labels> \
  --output-dir assets/props \
  --manifest assets/props/<name>-prop-pack.json \
  --component-mode largest
```

Compose a QA preview:

```bash
python3 .grok/skills/generate2dmap/scripts/compose_layered_preview.py \
  --base assets/map/<name>-base.png \
  --placements data/<name>-props.json \
  --output assets/map/<name>-layered-preview.png
```
